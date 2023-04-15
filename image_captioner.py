import tensorflow as tf
import einops
import numpy as np
import os
import re
import string

from utils import *

class Tokenizer():
	def __init__(self, dataset, weights_dir) -> None:
		self.dataset = dataset
		self.weights_dir = weights_dir
		self.vocabulary_size = 5000
		self.create_tokenizer()
  
	def create_tokenizer(self):
		self.tokenizer = tf.keras.layers.TextVectorization(max_tokens=self.vocabulary_size,
														   standardize=self.standardize,
														   ragged=True)
		self.set_vocab()

	def set_vocab(self):
		vocab_path = os.path.join(self.weights_dir, self.dataset, 'tokenizer', 'vocabulary', 'vocab.txt')
		# vocab_path = './XAI/Iterative/weights/flickr8k/tokenizer/vocabulary/vocab.txt'
		self.tokenizer.set_vocabulary(vocab_path)
	
	# def get_vocab(self):
	#	 return self.tokenizer.get_vocabulary()

	def get_tokenizer(self):
		return self.tokenizer

	def standardize(self, s):
		s = tf.strings.lower(s)
		s = tf.strings.regex_replace(s, f'[{re.escape(string.punctuation)}]', '')
		s = tf.strings.join(['[START]', s, '[END]'], separator=' ')
		return s

class SeqEmbedding(tf.keras.layers.Layer):
	def __init__(self, vocab_size, max_length, depth):
		super().__init__()
		self.pos_embedding = tf.keras.layers.Embedding(input_dim=max_length, output_dim=depth)

		self.token_embedding = tf.keras.layers.Embedding(
				input_dim=vocab_size,
				output_dim=depth,
				mask_zero=True)
		
		self.add = tf.keras.layers.Add()

	def call(self, seq):
		seq = self.token_embedding(seq) # (batch, seq, depth)

		x = tf.range(tf.shape(seq)[1])	# (seq)
		x = x[tf.newaxis, :]	# (1, seq)
		x = self.pos_embedding(x)	# (1, seq, depth)

		return self.add([seq,x])

class CausalSelfAttention(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super().__init__()
		self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
		# Use Add instead of + so the keras mask propagates through.
		self.add = tf.keras.layers.Add() 
		self.layernorm = tf.keras.layers.LayerNormalization()
	
	def call(self, x):
		attn = self.mha(query=x, value=x,
										use_causal_mask=True)
		x = self.add([x, attn])
		return self.layernorm(x)

class CrossAttention(tf.keras.layers.Layer):
	def __init__(self,**kwargs):
		super().__init__()
		self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
		self.add = tf.keras.layers.Add() 
		self.layernorm = tf.keras.layers.LayerNormalization()
	
	def call(self, x, y, **kwargs):
		attn, attention_scores = self.mha(
						 query=x, value=y,
						 return_attention_scores=True)
		
		self.last_attention_scores = attention_scores

		x = self.add([x, attn])
		return self.layernorm(x)

class FeedForward(tf.keras.layers.Layer):
	def __init__(self, units, dropout_rate=0.1):
		super().__init__()
		self.seq = tf.keras.Sequential([
				tf.keras.layers.Dense(units=2*units, activation='relu'),
				tf.keras.layers.Dense(units=units),
				tf.keras.layers.Dropout(rate=dropout_rate),
		])

		self.layernorm = tf.keras.layers.LayerNormalization()
	
	def call(self, x):
		x = x + self.seq(x)
		return self.layernorm(x)

class DecoderLayer(tf.keras.layers.Layer):
	def __init__(self, units, num_heads=1, dropout_rate=0.1):
		super().__init__()
		
		self.self_attention = CausalSelfAttention(num_heads=num_heads,
																							key_dim=units,
																							dropout=dropout_rate)
		self.cross_attention = CrossAttention(num_heads=num_heads,
																					key_dim=units,
																					dropout=dropout_rate)
		self.ff = FeedForward(units=units, dropout_rate=dropout_rate)
			

	def call(self, inputs, training=False):
		in_seq, out_seq = inputs

		# Text input
		out_seq = self.self_attention(out_seq)

		out_seq = self.cross_attention(out_seq, in_seq)
		
		self.last_attention_scores = self.cross_attention.last_attention_scores

		out_seq = self.ff(out_seq)

		return out_seq

class TokenOutput(tf.keras.layers.Layer):
	def __init__(self, tokenizer, banned_tokens=('', '[UNK]', '[START]'), dataset='flickr8k', weights_dir='', **kwargs):
		super().__init__()
		self.dataset = dataset
		self.weights_dir = weights_dir
		self.tokenizer = tokenizer
		self.dense = tf.keras.layers.Dense(
				units=self.tokenizer.vocabulary_size(), **kwargs)
		self.banned_tokens = banned_tokens

		self.bias = None
		self.counts = None

	def adapt(self):
		self.read_counts()
		total = self.counts.sum()
		p = self.counts/total
		p[self.counts==0] = 1.0
		log_p = np.log(p)	# log(1) == 0

		entropy = -(log_p*p).sum()

		print(type(self.counts), self.counts)
		print(f"Uniform entropy: {np.log(self.tokenizer.vocabulary_size()):0.2f}")
		print(f"Marginal entropy: {entropy:0.2f}")

		self.bias = log_p
		self.bias[self.counts==0] = -1e9

		return self.counts

	def read_counts(self):
		counts_path = os.path.join(self.weights_dir, self.dataset, 'token_output', 'counts.txt')
		with open(counts_path, 'r') as f:
				counts = f.readlines()
		counts = ''.join(counts)
		self.counts = np.fromstring(counts, dtype=float, sep='\n')

	def call(self, x):
		x = self.dense(x)
		# TODO(b/250038731): Fix this.
		# An Add layer doesn't work because of the different shapes.
		# This clears the mask, that's okay because it prevents keras from rescaling
		# the losses.
		return x + self.bias

class Captioner(tf.keras.Model):
	@classmethod
	def add_method(cls, fun):
		setattr(cls, fun.__name__, fun)
		return fun

	def __init__(self, dataset='flickr8k', weights_dir='', num_layers=2, units=256, max_length=50, num_heads=2, dropout_rate=0.1):
		super().__init__()

		self.dataset = dataset
		self.weights_dir = weights_dir

		self.create_tokenizer()
		self.create_output_layer()
		self.create_features_extractor()

		self.word_to_index = tf.keras.layers.StringLookup(
				mask_token="",
				vocabulary=self.tokenizer.get_vocabulary())
		self.index_to_word = tf.keras.layers.StringLookup(
				mask_token="",
				vocabulary=self.tokenizer.get_vocabulary(),
				invert=True) 

		self.seq_embedding = SeqEmbedding(
				vocab_size=self.tokenizer.vocabulary_size(),
				depth=units,
				max_length=max_length)

		self.decoder_layers = [
				DecoderLayer(units, num_heads=num_heads, dropout_rate=dropout_rate)
				for n in range(num_layers)]

	def create_tokenizer(self):
		self.tokenizer = Tokenizer(dataset=self.dataset, weights_dir=self.weights_dir).get_tokenizer()

	def create_output_layer(self):
		self.output_layer = TokenOutput(self.tokenizer, banned_tokens=('', '[UNK]', '[START]'), dataset=self.dataset, weights_dir=self.weights_dir)
		self.output_layer.adapt()

	def create_features_extractor(self):
		mobilenet = tf.keras.applications.MobileNetV3Small(
				input_shape=IMAGE_SHAPE,
				include_top=False,
				include_preprocessing=True)
		mobilenet.trainable=False
		self.feature_extractor = mobilenet

	def call(self, inputs):
		image, txt = inputs

		if image.shape[-1] == 3:
			# Apply the feature-extractor, if you get an RGB image.
			image = self.feature_extractor(image)
		
		# Flatten the feature map
		image = einops.rearrange(image, 'b h w c -> b (h w) c')


		if txt.dtype == tf.string:
			# Apply the tokenizer if you get string inputs.
			txt = self.tokenizer(txt)

		txt = self.seq_embedding(txt)

		# Look at the image
		for dec_layer in self.decoder_layers:
			txt = dec_layer(inputs=(image, txt))
			
		txt = self.output_layer(txt)

		return txt

	def tokenize_txt(self, txt): # Moda: XXX
		# if txt.dtype == tf.string:
			# Apply the tokenizer if you get string inputs.
		txt = self.tokenizer(txt)

		# txt = self.seq_embedding(txt)
		return txt

	def simple_gen(self, image, temperature=1):
		initial = self.word_to_index([['[START]']]) # (batch, sequence)
		img_features = self.feature_extractor(image[tf.newaxis, ...])

		tokens = initial # (batch, sequence)
		for n in range(50):
				preds = self((img_features, tokens)).numpy()	# (batch, sequence, vocab)
				preds = preds[:,-1, :]	#(batch, vocab)
				if temperature==0:
						next = tf.argmax(preds, axis=-1)[:, tf.newaxis]	# (batch, 1)
				else:
						next = tf.random.categorical(preds/temperature, num_samples=1)	# (batch, 1)
				tokens = tf.concat([tokens, next], axis=1) # (batch, sequence) 

				if next[0] == self.word_to_index('[END]'):
						break
		words = self.index_to_word(tokens[0, 1:-1])
		result = tf.strings.reduce_join(words, axis=-1, separator=' ')
		return result.numpy().decode()

class IC():
  def __init__(self, dataset='flickr8k', weights_dir='', num_layers=2, units=256, \
            max_length=50, num_heads=2, dropout_rate=0.5) -> None:
    self.dataset = dataset
    self.weights_dir = weights_dir
    self.feature_extractor = None
    self.tokenizer = None
    self.output_layer = None
    
    self.check_dataset()

    self.model = Captioner(dataset=self.dataset, weights_dir=self.weights_dir, num_layers=num_layers, units=units, \
                          max_length=max_length, num_heads=num_heads, dropout_rate=dropout_rate)
    self.compile_model()
    self.load_model_weights()

  def check_dataset(self):
    datasets = {'flickr8k', 'conceptual_captions', 'coco'}
    dataset = self.dataset.lower()
    if dataset in datasets:
        self.dataset = dataset
    else:
        print("No model is supported for this dataset!")
        print(f"Available datasets are: {datasets}")
        self.dataset = None
  
  def compile_model(self):
    self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
          loss=masked_loss,
          metrics=[masked_acc])
	
  def load_model_weights(self):
    weights_path = os.path.join(self.weights_dir, self.dataset, 'IC', f'{self.dataset}.model')
    self.model.load_weights(weights_path)

  def get_model(self):
    return self.model
  
#   def caption_image(self, image_path=''):
#     image = load_image(image_path)
#     result = self.model.simple_gen(image, temperature=0.0)
#     return result
  
  def caption_image(self, image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, IMAGE_SHAPE[:-1])
    result = self.model.simple_gen(image, temperature=0.0)
    return result

#############################################################################################################################3

# model = IC(dataset='flickr8k', weights_dir='.\XAI\Iterative\weights') #.get_model()

# image_path = './XAI/Iterative/surf.jpg'
# result = model.caption_image(image_path)

# print(result)

if __name__ == '__main__':
	c = IC(dataset='flickr8k', weights_dir='.\weights')
	import matplotlib.pyplot as plt
	image = plt.imread('.\\data\\test\\flickr8k\\save_dir\\0_1_blurred.jpg')

	print(c.caption_image(image).lower())

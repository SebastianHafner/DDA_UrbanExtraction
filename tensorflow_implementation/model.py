import tensorflow as tf
from experiment_manager.config import utils as config_utils
from tensorflow.python.keras import layers
from tensorflow.python.keras import models


class UNet(tf.keras.Model):

	def __init__(self):
		super(UNet, self).__init__()

		# encoder block 1
		self.encoder_double_convolution1 = double_convolution_block(num_filters=64)
		self.max_pool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))

		# encoder block 2
		self.encoder_double_convolution2 = double_convolution_block(num_filters=128)
		self.max_pool2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))

		# encoder block 3
		self.encoder_double_convolution3 = double_convolution_block(num_filters=256)
		self.max_pool3 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))

		# encoder block 4
		self.encoder_double_convolution4 = double_convolution_block(num_filters=512)
		self.max_pool4 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))

		# center
		self.center_double_convolution = double_convolution_block(num_filters=512)

		# decoder block 1
		self.transposed_convolution1 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')
		self.decoder_double_convolution1 = double_convolution_block(256)

		# decoder block 2
		self.transposed_convolution2 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')
		self.decoder_double_convolution2 = double_convolution_block(128)

		# decoder block 3
		self.transposed_convolution3 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')
		self.decoder_double_convolution3 = double_convolution_block(64)

		# decoder block 4
		self.transposed_convolution4 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')
		self.decoder_double_convolution4 = double_convolution_block(64)

		# out convolution
		self.out_convolution = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')

	def call(self, input_tensor):

		# encoder
		# encoder block 1
		encoder1 = self.encoder_double_convolution1(input_tensor)
		encoder1_pool = self.max_pool1(encoder1)
		# encoder block 2
		encoder2 = self.encoder_double_convolution2(encoder1_pool)
		encoder2_pool = self.max_pool2(encoder2)
		# encoder block 3
		encoder3 = self.encoder_double_convolution3(encoder2_pool)
		encoder3_pool = self.max_pool3(encoder3)
		# encoder block 4
		encoder4 = self.encoder_double_convolution4(encoder3_pool)
		encoder4_pool = self.max_pool4(encoder4)

		# center
		center = self.center_double_convolution(encoder4_pool)

		# decoder
		# decoder block 1
		decoder1_up = self.transposed_convolution1(center)
		decoder1_concat = layers.concatenate([encoder4, decoder1_up], axis=-1)
		decoder1 = self.decoder_double_convolution1(decoder1_concat)
		# decoder block 2
		decoder2_up = self.transposed_convolution2(decoder1)
		decoder2_concat = layers.concatenate([encoder3, decoder2_up], axis=-1)
		decoder2 = self.decoder_double_convolution2(decoder2_concat)
		# decoder block 3
		decoder3_up = self.transposed_convolution3(decoder2)
		decoder3_concat = layers.concatenate([encoder2, decoder3_up], axis=-1)
		decoder3 = self.decoder_double_convolution3(decoder3_concat)
		# decoder block 4
		decoder4_up = self.transposed_convolution4(decoder3)
		decoder4_concat = layers.concatenate([encoder1, decoder4_up], axis=-1)
		decoder4 = self.decoder_double_convolution4(decoder4_concat)

		output_tensor = self.out_convolution(decoder4)

		return output_tensor


def double_convolution_block(num_filters):
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same'))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Activation('relu'))
	model.add(tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same'))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Activation('relu'))
	return model

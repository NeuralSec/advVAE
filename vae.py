from keras.layers import Lambda, Input, Dense, Conv2D, Conv2DTranspose, Flatten
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import keras.losses


class VAE:
	def __init__(self, input_shape, latent_dim):
		# reparameterization trick
		# instead of sampling from Q(z|X), sample eps = N(0,I)
		# z = z_mean + sqrt(var)*eps
		def sampling(args):
			"""Reparameterization trick by sampling fr an isotropic unit Gaussian.
			# Arguments
				args (tensor): mean and log of variance of Q(z|X)
			# Returns
				z (tensor): sampled latent vector
			"""
			z_mean, z_log_var = args
			batch = K.shape(z_mean)[0]
			dim = K.int_shape(z_mean)[1]
			# by default, random_normal has mean=0 and std=1.0
			epsilon = K.random_normal(shape=(batch, dim))
			return z_mean + K.exp(0.5 * z_log_var) * epsilon

		# VAE model = encoder + decoder
		# build encoder model
		self.inputs = Input(shape=input_shape, name='encoder_input')
		x = Conv2D(filters=32, kernel_size=3, strides=(2,2), activation='relu')(self.inputs)
		x = Conv2D(filters=64, kernel_size=3, strides=(2,2), activation='relu')(x)
		x = Flatten()(x)
		z_mean = Dense(latent_dim, name='z_mean')(x)
		z_log_var = Dense(latent_dim, name='z_log_var')(x)
		# use reparameterization trick to push the sampling out as input
		# note that "output_shape" isn't necessary with the TensorFlow backend
		z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
		# instantiate encoder model
		self.encoder = Model(self.inputs, [z_mean, z_log_var, z], name='encoder')
		self.encoder.summary()
		
		# build decoder model
		latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
		x = Dense(7*7*32, activation='relu')(latent_inputs)
		x = Reshape(target_shape=(7, 7, 32))(x)
		x = Conv2DTranspose(filters=64, kernel_size=3, strides=(2,2), padding='SAME', activation='relu')(x)
		x = Conv2DTranspose(filters=32, kernel_size=3, strides=(2,2), padding="SAME", activation='relu')(x)
		outputs = Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding="SAME")(x)
		
		# instantiate decoder model
		self.decoder = Model(latent_inputs, outputs, name='decoder')
		self.decoder.summary()
		
		# instantiate VAE model
		self.outputs = self.decoder(self.encoder(self.inputs)[2])
		self.vae = Model(self.inputs, self.outputs, name='vae_mlp')
		
		def vae_loss(y_true, y_pred):
			reconstruction_loss = mse(y_true, y_pred)
			kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
			kl_loss = K.sum(kl_loss, axis=-1)
			kl_loss *= -0.5
			return K.mean(reconstruction_loss + kl_loss)

		self.vae.compile(optimizer='adam', loss=vae_loss, metrics=['mae'])
		self.vae.summary()

	def train(self, x, batch_size=32, epochs=10, val_ratio=0.1):
		self.vae.fit(x, x, epochs=epochs, batch_size=batch_size, validation_split=val_ratio, shuffle=True)
		return self.vae, self.encoder, self.decoder
	
	






from keras.layers import Lambda, Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, Concatenate
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import keras.losses

class VAE:
	def __init__(self, image_size, intermediate_dim, latent_dim):
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
		original_dim = image_size * image_size
		input_shape = (original_dim, )
		self.inputs = Input(shape=input_shape, name='encoder_input')
		x = Dense(intermediate_dim, activation='relu')(self.inputs)
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
		x = Dense(intermediate_dim, activation='relu')(latent_inputs)
		outputs = Dense(original_dim, activation='sigmoid')(x)
		
		# instantiate decoder model
		self.decoder = Model(latent_inputs, outputs, name='decoder')
		self.decoder.summary()
		
		# instantiate VAE model
		self.outputs = self.decoder(self.encoder(self.inputs)[2])
		self.vae = Model(self.inputs, self.outputs, name='vae_mlp')
		
		def vae_loss(y_true, y_pred):
			reconstruction_loss = mse(y_true, y_pred)
			reconstruction_loss *= original_dim
			kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
			kl_loss = K.sum(kl_loss, axis=-1)
			kl_loss *= -0.5
			return K.mean(reconstruction_loss + kl_loss)

		self.vae.compile(optimizer='adam', loss=vae_loss, metrics=['mae'])
		self.vae.summary()

	def train(self, x, batch_size=32, epochs=10, val_ratio=0.1):
		self.vae.fit(x, x, epochs=epochs, batch_size=batch_size, validation_split=val_ratio, shuffle=True)
		return self.vae, self.encoder, self.decoder

class CVAE:
	def __init__(self, image_size, cond_dim, intermediate_dim, latent_dim):
		# reparameterization trick
		# instead of sampling from Q(z|X), sample eps = N(0,I)
		# z = z_mean + sqrt(var)*eps
		def sampling(args):
			"""Reparameterization trick by sampling from an isotropic unit Gaussian.
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
		original_dim = image_size * image_size
		input_shape = (original_dim, )
		cond_shape = (cond_dim, )
		self.inputs = Input(shape=input_shape, name='encoder_input')
		self.cond = Input(shape=cond_shape, name='condition')
		concat_inputs = Concatenate(axis=-1)([self.inputs, self.cond])
		x = Dense(intermediate_dim, activation='relu')(concat_inputs)
		z_mean = Dense(latent_dim, name='z_mean')(x)
		z_log_var = Dense(latent_dim, name='z_log_var')(x)
		# use reparameterization trick to push the sampling out as input
		# note that "output_shape" isn't necessary with the TensorFlow backend
		z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
		# instantiate encoder model
		self.encoder = Model([self.inputs, self.cond], [z_mean, z_log_var, z], name='encoder')
		self.encoder.summary()
		
		# build decoder model
		latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
		x = Concatenate(axis=-1)([latent_inputs, self.cond])
		x = Dense(intermediate_dim, activation='relu')(x)
		outputs = Dense(original_dim, activation='sigmoid')(x)
		
		# instantiate decoder model
		self.decoder = Model([latent_inputs, self.cond], outputs, name='decoder')
		self.decoder.summary()
		
		# instantiate VAE model
		self.outputs = self.decoder([self.encoder([self.inputs, self.cond])[2], self.cond])
		self.cvae = Model([self.inputs, self.cond], self.outputs, name='vae_mlp')
		
		def vae_loss(y_true, y_pred):
			reconstruction_loss = mse(y_true, y_pred)
			reconstruction_loss *= original_dim
			kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
			kl_loss = K.sum(kl_loss, axis=-1)
			kl_loss *= -0.5
			return K.mean(reconstruction_loss + kl_loss)

		self.cvae.compile(optimizer='adam', loss=vae_loss, metrics=['mae'])
		self.cvae.summary()

	def train(self, x, cond, batch_size=32, epochs=10, val_ratio=0.1):
		self.cvae.fit([x, cond], x, epochs=epochs, batch_size=batch_size, validation_split=val_ratio, shuffle=True)
		return self.cvae, self.encoder, self.decoder

class ConvVAE:
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
			def mean_squared_error(y_t, y_p):
				return K.mean(K.square(y_p - y_t), axis=[-3,-2,-1])
			reconstruction_loss = mean_squared_error(y_true, y_pred)
			kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
			kl_loss = K.sum(kl_loss, axis=-1)
			kl_loss *= -0.5
			return K.mean(reconstruction_loss + kl_loss)

		self.vae.compile(optimizer='adam', loss=vae_loss, metrics=['mae'])
		self.vae.summary()

	def train(self, x, batch_size=32, epochs=10, val_ratio=0.1):
		self.vae.fit(x, x, epochs=epochs, batch_size=batch_size, validation_split=val_ratio, shuffle=True)
		return self.vae, self.encoder, self.decoder

class advVAE:
	def __init__(self, vae_encoder, vae_decoder, classifier):
		self.keras_vae_encoder = vae_encoder
		self.keras_vae_decoder = vae_decoder
		self.classifier = classifier
		for layer in self.keras_vae_encoder.layers:
			layer.trainable = False
		for layer in self.classifier.layers:
			layer.trainable = False
		self.inputs = self.keras_vae_encoder.inputs
		self.ouputs = self.keras_vae_decoder(self.keras_vae_encoder(self.inputs)[2])
		self.adv_vae = Model(inputs=self.inputs, outputs=self.ouputs)
		flatten_img = self.adv_vae(self.inputs)
		reshaped_img = Reshape(target_shape=(28, 28, 1))(flatten_img)
		classification_results = self.classifier(reshaped_img)
		self.adv_vae_classifier = Model(inputs=self.inputs, outputs=[classification_results, self.ouputs])
		def adv_loss(y_true, y_pred):
			return 1/(1+K.categorical_crossentropy(y_true, y_pred))
		self.adv_vae_classifier.compile(optimizer='adam', loss=[adv_loss,'mse'], metrics=['acc'])
		self.adv_vae_classifier.summary()
		self.adv_vae.summary()

	def attack(self, x, y, batch_size=32, epochs=10, val_ratio=0.1):
		self.adv_vae_classifier.fit(x, [y, x], epochs=epochs, batch_size=batch_size, validation_split=val_ratio, shuffle=True)
		return self.adv_vae, self.keras_vae_decoder

class advCVAE:
	def __init__(self, cvae_encoder, cvae_decoder, classifier):
		self.keras_cvae_encoder = cvae_encoder
		self.keras_cvae_decoder = cvae_decoder
		self.classifier = classifier
		for layer in self.keras_cvae_encoder.layers:
			layer.trainable = False
		for layer in self.classifier.layers:
			layer.trainable = False
		self.inputs = self.keras_cvae_encoder.inputs
		self.ouputs = self.keras_cvae_decoder([self.keras_cvae_encoder(self.inputs)[2], self.inputs[1]])
		self.adv_cvae = Model(inputs=self.inputs, outputs=self.ouputs)
		flatten_img = self.adv_cvae(self.inputs)
		reshaped_img = Reshape(target_shape=(28, 28, 1))(flatten_img)
		classification_results = self.classifier(reshaped_img)
		self.adv_cvae_classifier = Model(inputs=self.inputs, outputs=[classification_results, self.ouputs])
		def adv_loss(y_true, y_pred):
			return 1/(1+K.categorical_crossentropy(y_true, y_pred))
		self.adv_cvae_classifier.compile(optimizer='adam', loss=[adv_loss,'mse'], metrics=['acc'])
		self.adv_cvae_classifier.summary()
		self.adv_cvae.summary()

	def attack(self, x, y, batch_size=32, epochs=10, val_ratio=0.1):
		self.adv_cvae_classifier.fit([x, y], [y, x], epochs=epochs, batch_size=batch_size, validation_split=val_ratio, shuffle=True)
		return self.adv_cvae, self.keras_cvae_decoder



		

	






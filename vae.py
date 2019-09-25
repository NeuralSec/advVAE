import tensorflow as tf
from keras.layers import Lambda, Input, Dense, Dropout, Reshape, Concatenate, Average, Add, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, Flatten, MaxPooling2D, UpSampling2D, Activation
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.optimizers import RMSprop
from keras.utils import plot_model
from keras import backend as K
import keras.losses
import numpy as np
from tqdm import tqdm

#################################################################################################################################################
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

#################################################################################################################################################

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

#################################################################################################################################################

class ConvVAE:
	def __init__(self, input_shape, intermediate_dim, latent_dim):
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
		self.inputs = Input(shape=input_shape)  # adapt this if using `channels_first` image data format
		x = Conv2D(filters=3, kernel_size=(2,2), strides=1, activation='relu', padding='same')(self.inputs)
		x = Conv2D(filters=32, kernel_size=(2,2), strides=(2,2), activation='relu', padding='same')(x)
		x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')(x)
		x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')(x)
		x = Flatten()(x)
		x = Dense(intermediate_dim)(x)
		z_mean = Dense(latent_dim, name='z_mean')(x)
		z_log_var = Dense(latent_dim, name='z_log_var')(x)
		z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
		self.encoder = Model(self.inputs, [z_mean, z_log_var, z], name='encoder')
		self.encoder.summary()

		latent_inputs = Input(shape=(latent_dim,))
		x = Dense(intermediate_dim, activation='relu')(latent_inputs)
		x = Dense(32*16*16, activation='relu')(x)
		x = Reshape((16, 16, 32))(x)
		x = Conv2DTranspose(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')(x)
		x = Conv2DTranspose(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')(x)
		x = Conv2DTranspose(filters=32, kernel_size=(2,2), strides=(2,2), activation='relu', padding='valid')(x)
		decoded = Conv2DTranspose(filters=3, kernel_size=1, strides=1, activation='sigmoid', padding='valid')(x)
		
		# instantiate decoder model
		self.decoder = Model(latent_inputs, decoded, name='decoder')
		self.decoder.summary()
		
		# instantiate VAE model
		self.outputs = self.decoder(self.encoder(self.inputs)[2])
		self.vae = Model(self.inputs, self.outputs, name='vae_mlp')
		
		def vae_loss(y_true, y_pred):
			def mean_squared_error(y_t, y_p):
				return K.mean(K.square(y_p - y_t), axis=[-3,-2,-1])
			reconstruction_loss = 32*32*3*K.sum(binary_crossentropy(y_true, y_pred), axis=[-2,-1])
			#reconstruction_loss = 32*32*3*mean_squared_error(y_true, y_pred)
			kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
			kl_loss = K.sum(kl_loss, axis=-1)
			kl_loss *= -0.5
			return K.mean(reconstruction_loss + kl_loss)

		self.vae.compile(optimizer='adam', loss=vae_loss, metrics=['mae'])
		self.vae.summary()

	def train(self, x, batch_size=32, epochs=10, val_ratio=0.1):
		self.vae.fit(x, x, epochs=epochs, batch_size=batch_size, validation_split=val_ratio, shuffle=True)
		return self.vae, self.encoder, self.decoder

#################################################################################################################################################

class VAEGAN:
	def __init__(self, input_shape, intermediate_dim, latent_dim, gamma=1e-4):
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

		def sampling_normal(latent_in):
			batch =  K.shape(latent_in)[0]
			dim = K.shape(latent_in)[1]
			return K.random_normal(shape=(batch, dim))

		# build encoder model
		encoder_inputs = Input(shape=input_shape, name='encoder_inputs')  # adapt this if using `channels_first` image data format
		x_e = Conv2D(filters=3, kernel_size=(2,2), strides=1, activation='relu', padding='same')(encoder_inputs)
		x_e = Conv2D(filters=32, kernel_size=(2,2), strides=(2,2), activation='relu', padding='same')(x_e)
		x_e = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')(x_e)
		x_e = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')(x_e)
		x_e = Flatten()(x_e)
		x_e = Dense(intermediate_dim)(x_e)
		z_mean = Dense(latent_dim, name='z_mean')(x_e)
		z_log_var = Dense(latent_dim, name='z_log_var')(x_e)
		z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
		self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='Encoder')

		# build decoder model
		latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
		x_d = Dense(intermediate_dim, activation='relu')(latent_inputs)
		x_d = Dense(32*16*16, activation='relu')(x_d)
		x_d = Reshape((16, 16, 32))(x_d)
		x_d = Conv2DTranspose(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')(x_d)
		x_d = Conv2DTranspose(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')(x_d)
		x_d = Conv2DTranspose(filters=32, kernel_size=(2,2), strides=(2,2), activation='relu', padding='valid')(x_d)
		decoded = Conv2DTranspose(filters=3, kernel_size=1, strides=1, activation='sigmoid', padding='valid', name='decoder_output')(x_d)
		self.decoder = Model(latent_inputs, decoded, name='Decoder')
		
		# build VAE model
		vae_outputs = self.decoder(self.encoder(encoder_inputs)[2])
		self.vae = Model(encoder_inputs, vae_outputs, name='VAE')

		# build discriminator
		discrim_input = Input(shape=input_shape, name='discriminator_input')
		x_c = Conv2D(filters=32, kernel_size=(3,3), strides=1, activation='relu', padding='same')(discrim_input)
		x_c = BatchNormalization()(x_c)
		x_c = Conv2D(filters=32, kernel_size=(3,3), strides=1, activation='relu', padding='same')(x_c)
		x_c = BatchNormalization()(x_c)
		x_c = MaxPooling2D(pool_size=(2,2))(x_c)
		x_c = Dropout(0.2)(x_c)
		x_c = Conv2D(filters=64, kernel_size=(3,3), strides=1, activation='relu', padding='same')(discrim_input)
		x_c = BatchNormalization()(x_c)
		middle_conv = Conv2D(filters=64, kernel_size=(3,3), strides=1, activation='relu', padding='same', name='discriminator_intermidiate_output')(x_c)
		x_c = BatchNormalization()(middle_conv)
		x_c = MaxPooling2D(pool_size=(2,2))(x_c)
		x_c = Dropout(0.2)(x_c)
		x_c = Flatten()(x_c)
		x_c = Dense(256, activation='relu')(x_c)
		x_c = Dense(1)(x_c)
		discrim_output = Activation('sigmoid', name='discriminator_output')(x_c)
		self.discriminator = Model(discrim_input, [discrim_output, middle_conv], name='Discriminator')
		
		# build vaegan _ng: no gradients of the vae, _nge: no gradients of the encoder 
		encode_real = self.encoder(encoder_inputs)[2]
		encode_real_nge = Lambda(K.stop_gradient)(encode_real)
		
		sample_noise = Lambda(sampling_normal, output_shape=(latent_dim,), name='sampled_noise')(encode_real)
		sample_noise_nge = Lambda(K.stop_gradient)(sample_noise)
		
		decode_real = self.decoder(encode_real)
		decode_real_nge = self.decoder(encode_real_nge)
		
		decode_noise = self.decoder(sample_noise)
		decode_noise_nge = self.decoder(sample_noise_nge)
		
		self.vae_nge = Model(encoder_inputs, [decode_real_nge, decode_noise_nge], name='vae_encoder_fixed')
		
		[cons_from_real_nge, cons_from_noise_nge] = self.vae_nge(encoder_inputs)
		cons_from_real_ng = Lambda(K.stop_gradient)(cons_from_real_nge)
		cons_from_noise_ng = Lambda(K.stop_gradient)(cons_from_noise_nge)
		
		# No fixed
		[discrim_real_score, l_real_x] = self.discriminator(encoder_inputs)
		[discrim_tilde_score, l_tiled_x] = self.discriminator(decode_real)
		[discrim_noise_score, l_noise] = self.discriminator(decode_noise)

		# encoder fixes
		[discrim_tilde_score_nge, l_tiled_x_nge] = self.discriminator(cons_from_real_nge)
		[discrim_noise_score_nge, l_noise_nge] = self.discriminator(cons_from_noise_nge)

		# vae fixed
		[discrim_tilde_score_ng, l_tiled_x_ng] = self.discriminator(cons_from_real_ng)
		[discrim_noise_score_ng, l_noise_ng] = self.discriminator(cons_from_noise_ng)
		
		self.vaegan = Model(encoder_inputs, [discrim_real_score, discrim_tilde_score_ng, discrim_noise_score_ng], name='vaegan_vae_fixed')

		# define losses
		def NLLNormal(pred, target):
			c = -0.5 * tf.log(2 * np.pi)
			multiplier = 1.0 / (2.0 * 1)
			tmp = tf.square(pred - target)
			tmp *= -multiplier
			tmp += c
			return tmp

		# losses
		ll_loss = K.sum(NLLNormal(l_tiled_x, l_real_x), axis=[-3,-2,-1])
		ll_loss = K.mean(ll_loss)
		ll_loss_nge = K.sum(NLLNormal(l_tiled_x_nge, l_real_x), axis=[-3,-2,-1])
		ll_loss_nge = K.mean(ll_loss_nge)

		kl_loss = 1 + encode_real[1] - K.square(encode_real[0]) - K.exp(encode_real[1])
		kl_loss = K.sum(kl_loss, axis=-1)
		kl_loss *= -0.5
		kl_loss = K.mean(kl_loss)
		
		encoder_loss = kl_loss/latent_dim - ll_loss/(32*32*64)
		decoder_loss =  -K.mean(K.relu(1-discrim_noise_score) + K.relu(1-discrim_tilde_score)) - gamma*ll_loss_nge
		gan_loss = K.mean(K.relu(1+discrim_real_score) + K.relu(1-discrim_tilde_score_ng) + K.relu(1-discrim_noise_score_ng))

		self.encoder.add_loss(encoder_loss)
		self.vae_nge.add_loss(decoder_loss)
		self.vaegan.add_loss(gan_loss)
		self.encoder.compile(optimizer='adam')
		self.vae_nge.compile(optimizer='adam')
		self.vaegan.compile(optimizer='adam')
		print('\n********************** Encoder Summary *****************************************')
		self.encoder.summary()
		print('\n********************** Decoder Summary *****************************************')
		self.vae_nge.summary()
		print('\n********************** VAEGAN Summary *****************************************')
		self.vaegan.summary()
		self.latent_dim = latent_dim
		
	def train(self, x, batch_size=32, epochs=10, val_ratio=0.1):
		history = LossHistory()
		self.vaegan.fit(x, epochs=1, batch_size=batch_size, validation_split=0.1)
		for epoch in range(epochs):
			print(f'Training epoch: {epoch+1}/{epochs}')
			for batch_ind in tqdm(range(int(x.shape[0]/batch_size))):
				start = batch_ind * batch_size
				end = start + batch_size
				x_batch = x[start:end]
				
				self.vae_nge.trainable = False
				self.vaegan.trainable = False
				self.encoder.trainable = True
				self.encoder.fit(x_batch, epochs=1, batch_size=batch_size, validation_split=0, verbose=0, callbacks=[history])
				if batch_ind%100 == 0:
					print(f'Encoder loss: {history.losses}')

				self.encoder.trainable = False
				self.vaegan.trainable = False
				self.vae_nge.trainable = True
				self.vae_nge.fit(x_batch, epochs=1, batch_size=batch_size, validation_split=0, verbose=0, callbacks=[history])
				if batch_ind%100 == 0:
					print(f'Decoder losses: {history.losses}')

				self.encoder.trainable = False
				self.vae_nge.trainable = False
				self.vaegan.trainable = True
				self.vaegan.fit(x_batch, epochs=1, batch_size=batch_size, validation_split=0, verbose=0, callbacks=[history])
				if batch_ind%100 == 0:
					print(f'Discriminator losses: {history.losses}')
		return self.vae, self.encoder, self.decoder

#################################################################################################################################################

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

#################################################################################################################################################

class ConvCVAE:
	def __init__(self, input_shape, cond_dim, intermediate_dim, latent_dim):
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
		cond_shape = (cond_dim, )
		self.cond = Input(shape=cond_shape, name='condition')
		self.inputs = Input(shape=input_shape)  # adapt this if using `channels_first` image data format
		x = Conv2D(filters=3, kernel_size=(2,2), strides=1, activation='relu', padding='same')(self.inputs)
		x = Conv2D(filters=32, kernel_size=(2,2), strides=(2,2), activation='relu', padding='same')(x)
		x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')(x)
		x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')(x)
		x = Flatten()(x)
		concat_inputs = Concatenate(axis=-1)([x, self.cond])
		x = Dense(intermediate_dim)(concat_inputs)
		z_mean = Dense(latent_dim, name='z_mean')(x)
		z_log_var = Dense(latent_dim, name='z_log_var')(x)
		z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
		self.encoder = Model([self.inputs, self.cond], [z_mean, z_log_var, z], name='encoder')
		self.encoder.summary()

		latent_inputs = Input(shape=(latent_dim,))
		x = Concatenate(axis=-1)([latent_inputs, self.cond])
		x = Dense(intermediate_dim, activation='relu')(latent_inputs)
		x = Dense(32*16*16, activation='relu')(x)
		x = Reshape((16, 16, 32))(x)
		x = Conv2DTranspose(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')(x)
		x = Conv2DTranspose(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')(x)
		x = Conv2DTranspose(filters=32, kernel_size=(2,2), strides=(2,2), activation='relu', padding='valid')(x)
		decoded = Conv2DTranspose(filters=3, kernel_size=1, strides=1, activation='sigmoid', padding='valid')(x)
		
		# instantiate decoder model
		self.decoder = Model([latent_inputs, self.cond], decoded, name='decoder')
		self.decoder.summary()
		
		# instantiate VAE model
		self.outputs = self.decoder([self.encoder([self.inputs, self.cond])[2], self.cond])
		self.cvae = Model([self.inputs, self.cond], self.outputs, name='vae_mlp')
		
		def vae_loss(y_true, y_pred):
			def mean_squared_error(y_t, y_p):
				return K.mean(K.square(y_p - y_t), axis=[-3,-2,-1])
			reconstruction_loss = 32*32*3*K.sum(binary_crossentropy(y_true, y_pred), axis=[-2,-1])
			#reconstruction_loss = 32*32*3*mean_squared_error(y_true, y_pred)
			kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
			kl_loss = K.sum(kl_loss, axis=-1)
			kl_loss *= -0.5
			return K.mean(reconstruction_loss + kl_loss)

		self.cvae.compile(optimizer='adam', loss=vae_loss, metrics=['mae'])
		self.cvae.summary()

	def train(self, x, cond, batch_size=32, epochs=10, val_ratio=0.1):
		self.cvae.fit([x, cond], x, epochs=epochs, batch_size=batch_size, validation_split=val_ratio, shuffle=True)
		return self.cvae, self.encoder, self.decoder

#################################################################################################################################################

class advVAE:
	def __init__(self, vae_encoder, vae_decoder, classifier, c=1., is_targeted=True, for_mnist=True):
		self.keras_vae_encoder = vae_encoder
		self.keras_vae_decoder = vae_decoder
		self.classifier = classifier
		for layer in self.keras_vae_encoder.layers:
			layer.trainable = False
		for layer in self.classifier.layers:
			layer.trainable = False
		self.inputs = self.keras_vae_encoder.inputs
		self.ouputs = self.keras_vae_decoder(self.keras_vae_encoder(self.inputs)[2])
		self.adv_vae = Model(inputs=self.inputs, outputs=self.ouputs, name='adv_vae')
		ex = self.adv_vae(self.inputs)
		if for_mnist:
			ex = Reshape(target_shape=(28, 28, 1))(ex)
		classification_results = self.classifier(ex)
		self.adv_vae_classifier = Model(inputs=self.inputs, outputs=[classification_results, self.ouputs], name='adv_vae_classifier')
		def non_targeted_loss(y_true, y_pred):
			real = tf.reduce_sum(y_true * y_pred, 1)
			other = tf.reduce_max((1 - y_true) * y_pred - (y_true * 10000), 1)
			return c*tf.reduce_sum(tf.maximum(0.0, real - other))
		def targeted_loss(y_true, y_pred):
			real = tf.reduce_sum(y_true * y_pred, 1)
			other = tf.reduce_max((1 - y_true) * y_pred - (y_true * 10000), 1)
			return c*tf.reduce_sum(tf.maximum(0.0, other - real))
		if is_targeted:
			self.adv_vae_classifier.compile(optimizer='adam', loss=[targeted_loss,'mse'], metrics=['mse'])
		else:
			self.adv_vae_classifier.compile(optimizer='adam', loss=[non_targeted_loss,'mse'], metrics=['mse'])
		print('***********************advVAE with classifier summary************************')
		self.adv_vae_classifier.summary()
		print('***********************advVAE summary***************************')
		self.adv_vae.summary()

	def attack(self, x, y, batch_size=32, epochs=10, val_ratio=0.1):
		self.adv_vae_classifier.fit(x, [y, x], epochs=epochs, batch_size=batch_size, validation_split=val_ratio, shuffle=True)
		return self.adv_vae, self.keras_vae_decoder

#################################################################################################################################################

class advCVAE:
	def __init__(self, cvae_encoder, cvae_decoder, classifier, c=1., is_targeted=True, for_mnist=True):
		self.keras_cvae_encoder = cvae_encoder
		self.keras_cvae_decoder = cvae_decoder
		self.classifier = classifier
		for layer in self.keras_cvae_encoder.layers:
			layer.trainable = False
		for layer in self.classifier.layers:
			layer.trainable = False
		self.inputs = self.keras_cvae_encoder.inputs
		self.ouputs = self.keras_cvae_decoder([self.keras_cvae_encoder(self.inputs)[2], self.inputs[1]])
		self.adv_cvae = Model(inputs=self.inputs, outputs=self.ouputs, name='adv_Cvae')
		ex = self.adv_cvae(self.inputs)
		if for_mnist:
			ex = Reshape(target_shape=(28, 28, 1))(ex)
		classification_results = self.classifier(ex)
		self.adv_cvae_classifier = Model(inputs=self.inputs, outputs=[classification_results, self.ouputs], name='adv_Cvae_classifier')
		def non_targeted_loss(y_true, y_pred):
			real = tf.reduce_sum(y_true * y_pred, 1)
			other = tf.reduce_max((1 - y_true) * y_pred - (y_true * 10000), 1)
			return c*tf.reduce_sum(tf.maximum(0.0, real - other))
		def targeted_loss(y_true, y_pred):
			real = tf.reduce_sum(y_true * y_pred, 1)
			other = tf.reduce_max((1 - y_true) * y_pred - (y_true * 10000), 1)
			return c*tf.reduce_sum(tf.maximum(0.0, other - real))
		if is_targeted:
			self.adv_cvae_classifier.compile(optimizer='adam', loss=[targeted_loss,'mse'], metrics=['mse'])
		else:
			self.adv_cvae_classifier.compile(optimizer='adam', loss=[non_targeted_loss,'mse'], metrics=['mse'])
		print('***********************advCVAE with classifier summary************************')
		self.adv_cvae_classifier.summary()
		print('***********************advCVAE with classifier summary************************')
		self.adv_cvae.summary()

	def attack(self, x, cond, y, batch_size=32, epochs=10, val_ratio=0.1):
		self.adv_cvae_classifier.fit([x, cond], [y, x], epochs=epochs, batch_size=batch_size, validation_split=val_ratio, shuffle=True)
		return self.adv_cvae, self.keras_cvae_decoder

#################################################################################################################################################

class advEgnosticVAE:
	def __init__(self, vae_encoders, vae_decoder, classifier, c=1., is_targeted=True, for_mnist=True):
		self.keras_vae_encoders = vae_encoders
		self.keras_vae_decoder = vae_decoder
		self.classifier = classifier
		for encoder in self.keras_vae_encoders:
			for layer in encoder.layers:
				layer.trainable = False
		for layer in self.classifier.layers:
			layer.trainable = False
		self.inputs = self.keras_vae_encoders[0].inputs
		encoder_outputs = [encoder(self.inputs)[2] for encoder in self.keras_vae_encoders]
		print(f'******************{encoder_outputs}*******************')
		averaged_latent_z = Average()(encoder_outputs)
		self.ouputs = self.keras_vae_decoder(averaged_latent_z)
		self.adv_vae = Model(inputs=self.inputs, outputs=self.ouputs, name='adv_vae')
		ex = self.adv_vae(self.inputs)
		if for_mnist:
			ex = Reshape(target_shape=(28, 28, 1))(ex)
		classification_results = self.classifier(ex)
		self.adv_vae_classifier = Model(inputs=self.inputs, outputs=[classification_results, self.ouputs], name='adv_vae_classifier')
		def non_targeted_loss(y_true, y_pred):
			real = tf.reduce_sum(y_true * y_pred, 1)
			other = tf.reduce_max((1 - y_true) * y_pred - (y_true * 10000), 1)
			return c*tf.reduce_sum(tf.maximum(0.0, real - other))
		def targeted_loss(y_true, y_pred):
			real = tf.reduce_sum(y_true * y_pred, 1)
			other = tf.reduce_max((1 - y_true) * y_pred - (y_true * 10000), 1)
			return c*tf.reduce_sum(tf.maximum(0.0, other - real))
		if is_targeted:
			self.adv_vae_classifier.compile(optimizer='adam', loss=[targeted_loss,'mse'], metrics=['mse'])
		else:
			self.adv_vae_classifier.compile(optimizer='adam', loss=[non_targeted_loss,'mse'], metrics=['mse'])
		print('***********************advVAE with classifier summary************************')
		self.adv_vae_classifier.summary()
		print('***********************advVAE summary***************************')
		self.adv_vae.summary()

	def attack(self, x, y, batch_size=32, epochs=10, val_ratio=0.1):
		self.adv_vae_classifier.fit(x, [y, x], epochs=epochs, batch_size=batch_size, validation_split=val_ratio, shuffle=True)
		return self.adv_vae, self.keras_vae_decoder

	






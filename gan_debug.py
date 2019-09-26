import tensorflow as tf
from keras.layers import Lambda, Input, Dense, Dropout, Reshape, Concatenate, Average, Add, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, Flatten, MaxPooling2D, UpSampling2D, Activation
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K
import keras.losses
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
from tqdm import tqdm
import utils
import matplotlib.pyplot as plt

def adam_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

def create_generator(output_shape):
	# build decoder model
	latent_inputs = Input(shape=(100,), name='decoder_input')
	x_d = Dense(256, activation=LeakyReLU(0.2))(latent_inputs)
	x_d = Dense(512, activation=LeakyReLU(0.2))(x_d)
	x_d = Dense(1024, activation=LeakyReLU(0.2))(x_d)
	x_d = Dense(output_shape[0]*output_shape[1]*output_shape[2], activation='sigmoid', name='decoder_output')(x_d)
	decoded = Reshape(target_shape=output_shape)(x_d)
	gen = Model(latent_inputs, decoded, name='Gen')
	gen.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
	return gen

def create_discriminator(input_shape):
	# build discriminator
	discrim_input = Input(shape=input_shape, name='discriminator_input')
	x_c = Flatten(name='F1')(discrim_input)
	x_c = Dense(1024, activation=LeakyReLU(0.2), name='D1')(x_c)
	x_c = Dropout(0.3)(x_c)
	x_c = Dense(512, activation=LeakyReLU(0.2), name='D2')(x_c)
	x_c = Dropout(0.3)(x_c)
	x_c = Dense(256, activation=LeakyReLU(0.2), name='D3')(x_c)
	x_c = Dense(1)(x_c)
	discrim_output = Activation('sigmoid', name='discriminator_output')(x_c)
	discriminator = Model(discrim_input, discrim_output, name='Discriminator')
	discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer(), metrics=['acc'])
	return discriminator

def create_gan(gen, dis):	
	# build gan
	gan_input = Input(shape=(100,), name='gan_input')
	gan_out = dis(gen(gan_input))
	gan =Model(gan_input, gan_out, name='GAN')
	gan.compile(loss='binary_crossentropy', optimizer='adam')
	return gan

GEN = create_generator((28,28,1))
DIS = create_discriminator((28,28,1))
GAN = create_gan(GEN, DIS)

mnist_X_train, mnist_y_train, _, _ = utils.load_dataset(dataset='mnist')
y_labels = np.argmax(mnist_y_train, axis=-1)
inds = np.where(y_labels == 0)
mnist_X_train = mnist_X_train[inds]
epochs = 400
batch_size=128

for epoch in range(epochs):
	print(f'Training epoch: {epoch+1}/{epochs}')
	for batch_ind in tqdm(range(int(mnist_X_train.shape[0]/batch_size))):
		start = batch_ind * batch_size
		end = start + batch_size
		x_batch = mnist_X_train[start:end]
		noise_batch = np.random.normal(0,1, [batch_size, 100])
		generated_batch = GEN.predict(noise_batch)
		concat_batch= np.concatenate([x_batch, generated_batch])
		y_dis=np.zeros(2*batch_size)
		y_dis[:batch_size]=0.9
		DIS.trainable=True
		DIS.train_on_batch(concat_batch, y_dis)

		noise_batch = np.random.normal(0,1, [batch_size, 100])
		y_gen = np.ones(batch_size)
		DIS.trainable=False
		GAN.train_on_batch(noise_batch, y_gen)
	if epoch == 0 or epoch%50 == 0:
		imgs = GEN.predict(noise_batch[:10]).reshape(-1, 28, 28)
		fig, axes = plt.subplots(nrows=1, ncols=10)
		for i in range(10):
			axes[i].imshow(imgs[i].reshape(28, 28), interpolation='nearest')
		plt.show()
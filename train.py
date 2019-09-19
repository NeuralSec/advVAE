import numpy as np
from keras.datasets import mnist
from keras.models import Model, load_model
import argparse
import utils
from vae import VAE, ConvVAE, CVAE, ConvCVAE, advVAE, advCVAE
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

DATA ='mnist'
BATCH_SIZE = 32
EPOCHS = 20
TARGETED = False
TARGET_CLASS = 0

COND_DIM = 10
MNIST_VAE_DIM = 2		# Latent dimension of mnist VAE
CIFAR_VAE_DIM = 64		# Latent dimension of cifar10 VAE
MNIST_INTER_DIM = 512
CIFAR_INTER_DIM = 128
if DATA =='mnist':
	INPUT_SHAPE = (28,28,1)
elif DATA =='cifar10':
	INPUT_SHAPE = (32,32,3)
TRAIN_VAE = False	# train a vae for begining or load a trained one
TRAIN_advVAE = False
TRAIN_CVAE = False
TRAIN_advCVAE = True

if __name__ == '__main__':
	# load mnist dataset for training and testing
	mnist_X_train, mnist_y_train, mnist_X_test, mnist_y_test = utils.load_dataset(dataset='mnist')
	mnist_X_train = np.reshape(mnist_X_train, (-1, 28**2))
	mnist_X_test = np.reshape(mnist_X_test, (-1, 28**2))
	cifar10_X_train, cifar10_y_train, cifar10_X_test, cifar10_y_test = utils.load_dataset(dataset='cifar10')
	print('===== Loading shadow models =======')
	mnist_substitute = load_model('mnist_substitute.h5')
	mnist_substitute.layers.pop()
	cifar10_substitute = load_model('cifar10_substitute.h5')
	cifar10_substitute.layers.pop()
	print('===== Loading victim models =======')
	mnist_classifier = load_model('mnist_model.h5')
	cifar10_classifier = load_model('cifar10_model.h5')
	
	#print('===== Model summary =======')
	#mnist_substitute.summary()
	#mnist_classifier.summary()
	#cifar10_substitute.summary()
	#cifar10_classifier.summary()

	if DATA == 'mnist':
		if TRAIN_VAE:
			model = VAE(28, MNIST_INTER_DIM, MNIST_VAE_DIM)
			model.train(mnist_X_train, batch_size=BATCH_SIZE, epochs=EPOCHS, val_ratio=0.1)
			model.vae.save(f'snapshots/mnist-vae-{MNIST_VAE_DIM}d.h5')
			model.encoder.save(f'snapshots/mnist-vae-encoder-{MNIST_VAE_DIM}d.h5')
			model.decoder.save(f'snapshots/mnist-vae-decoder-{MNIST_VAE_DIM}d.h5')
			victim = VAE(28, MNIST_INTER_DIM+88, MNIST_VAE_DIM)
			victim.train(mnist_X_train, batch_size=BATCH_SIZE, epochs=EPOCHS, val_ratio=0.1)
			victim.vae.save(f'snapshots/victim-mnist-vae-{MNIST_VAE_DIM}d.h5')
			victim.encoder.save(f'snapshots/victim-mnist-vae-encoder-{MNIST_VAE_DIM}d.h5')
			victim.decoder.save(f'snapshots/victim-mnist-vae-decoder-{MNIST_VAE_DIM}d.h5')

		#loading trained vae for adv training
		if TRAIN_advVAE:
			print('===== Loading VAE =======')
			vae = load_model(f'./snapshots/mnist-vae-{MNIST_VAE_DIM}d.h5', compile=False)
			vae_encoder = load_model(f'./snapshots/mnist-vae-encoder-{MNIST_VAE_DIM}d.h5', compile=False)
			vae_decoder = load_model(f'./snapshots/mnist-vae-decoder-{MNIST_VAE_DIM}d.h5', compile=False)
			print(f'{MNIST_VAE_DIM}-D VAE loaded.')
			# training based on classifier with logits
			advvae = advVAE(vae_encoder, vae_decoder, mnist_substitute, c=0.1, is_targeted=TARGETED, for_mnist=True)
			if TARGETED:
				y_labels = np.zeros(mnist_y_train.shape)
				y_labels[:,TARGET_CLASS] = 1
			else:
				y_labels = mnist_y_train
			adv_vae, adv_decoder = advvae.attack(mnist_X_train, y_labels, batch_size=32, epochs=EPOCHS, val_ratio=0.1)
			adv_vae.save(f'snapshots/mnist-adv-vae-{MNIST_VAE_DIM}d.h5')
			adv_decoder.save(f'snapshots/mnist-adv-decoder-{MNIST_VAE_DIM}d.h5')

		if TRAIN_CVAE:
			cvae = CVAE(28, COND_DIM, MNIST_INTER_DIM, MNIST_VAE_DIM)
			cvae.train(mnist_X_train, mnist_y_train, batch_size=32, epochs=EPOCHS, val_ratio=0.1)
			cvae.cvae.save(f'snapshots/mnist-cvae-{MNIST_VAE_DIM}d.h5')
			cvae.encoder.save(f'snapshots/mnist-cvae-encoder-{MNIST_VAE_DIM}d.h5')
			cvae.decoder.save(f'snapshots/mnist-cvae-decoder-{MNIST_VAE_DIM}d.h5')

		if TRAIN_advCVAE:
			print('===== Loading CVAE =======')
			cvae = load_model(f'./snapshots/mnist-cvae-{MNIST_VAE_DIM}d.h5', compile=False)
			cvae_encoder = load_model(f'./snapshots/mnist-cvae-encoder-{MNIST_VAE_DIM}d.h5', compile=False)
			cvae_decoder = load_model(f'./snapshots/mnist-cvae-decoder-{MNIST_VAE_DIM}d.h5', compile=False)
			print(f'{MNIST_VAE_DIM}-D CVAE loaded.')
			advcvae = advCVAE(cvae_encoder, cvae_decoder, mnist_substitute)
			if TARGETED:
				y_labels = np.zeros(mnist_y_train.shape)
				y_labels[:,TARGET_CLASS] = 1
			else:
				y_labels = mnist_y_train
			adv_cvae, adv_cdecoder = advcvae.attack(x=mnist_X_train, cond=mnist_y_train, y=y_labels, batch_size=32, epochs=EPOCHS, val_ratio=0.1)
			adv_cvae.save(f'snapshots/mnist-adv-cvae-{MNIST_VAE_DIM}d.h5')
			adv_cdecoder.save(f'snapshots/mnist-adv-cdecoder-{MNIST_VAE_DIM}d.h5')

	elif DATA == 'cifar10':
		if TRAIN_VAE:
			model = ConvVAE(INPUT_SHAPE, CIFAR_INTER_DIM, CIFAR_VAE_DIM)
			model.train(cifar10_X_train, batch_size=BATCH_SIZE, epochs=EPOCHS, val_ratio=0.1)
			model.vae.save(f'snapshots/cifar10-vae-{CIFAR_VAE_DIM}d.h5')
			model.encoder.save(f'snapshots/cifar10-vae-encoder-{CIFAR_VAE_DIM}d.h5')
			model.decoder.save(f'snapshots/cifar10-vae-decoder-{CIFAR_VAE_DIM}d.h5')
			victim = ConvVAE(INPUT_SHAPE, CIFAR_INTER_DIM*2, CIFAR_VAE_DIM)
			victim.train(cifar10_X_train, batch_size=BATCH_SIZE, epochs=EPOCHS, val_ratio=0.1)
			victim.vae.save(f'snapshots/victim-cifar10-vae-{CIFAR_VAE_DIM}d.h5')
			victim.encoder.save(f'snapshots/victim-cifar10-vae-encoder-{CIFAR_VAE_DIM}d.h5')
			victim.decoder.save(f'snapshots/victim-cifar10-vae-decoder-{CIFAR_VAE_DIM}d.h5')

		if TRAIN_advVAE:
			print('===== Loading VAE =======')
			vae = load_model(f'./snapshots/cifar10-vae-{CIFAR_VAE_DIM}d.h5', compile=False)
			vae_encoder = load_model(f'./snapshots/cifar10-vae-encoder-{CIFAR_VAE_DIM}d.h5', compile=False)
			vae_decoder = load_model(f'./snapshots/cifar10-vae-decoder-{CIFAR_VAE_DIM}d.h5', compile=False)
			print(f'{CIFAR_VAE_DIM}-D VAE loaded.')
			advvae = advVAE(vae_encoder, vae_decoder, cifar10_substitute, c=0.05, is_targeted=TARGETED, for_mnist=False)
			if TARGETED:
				y_labels = np.zeros(cifar10_y_train.shape)
				y_labels[:,TARGET_CLASS] = 1
			else:
				y_labels = cifar10_y_train
			adv_vae, adv_decoder = advvae.attack(cifar10_X_train, y_labels, batch_size=32, epochs=EPOCHS, val_ratio=0.1)
			adv_vae.save(f'snapshots/cifar10-adv-vae-{CIFAR_VAE_DIM}d.h5')
			adv_decoder.save(f'snapshots/cifar10-adv-decoder-{CIFAR_VAE_DIM}d.h5')

		if TRAIN_CVAE:
			cvae = ConvCVAE(INPUT_SHAPE, COND_DIM, CIFAR_INTER_DIM, CIFAR_VAE_DIM)
			cvae.train(cifar10_X_train, cifar10_y_train, batch_size=32, epochs=EPOCHS, val_ratio=0.1)
			cvae.cvae.save(f'snapshots/cifar10-cvae-{CIFAR_VAE_DIM}d.h5')
			cvae.encoder.save(f'snapshots/cifar10-cvae-encoder-{CIFAR_VAE_DIM}d.h5')
			cvae.decoder.save(f'snapshots/cifar10-cvae-decoder-{CIFAR_VAE_DIM}d.h5')

		if TRAIN_advCVAE:
			print('===== Loading CVAE =======')
			cvae = load_model(f'./snapshots/cifar10-cvae-{CIFAR_VAE_DIM}d.h5', compile=False)
			cvae_encoder = load_model(f'./snapshots/cifar10-cvae-encoder-{CIFAR_VAE_DIM}d.h5', compile=False)
			cvae_decoder = load_model(f'./snapshots/cifar10-cvae-decoder-{CIFAR_VAE_DIM}d.h5', compile=False)
			print(f'{CIFAR_VAE_DIM}-D CVAE loaded.')
			advcvae = advCVAE(cvae_encoder, cvae_decoder, mnist_substitute)
			if TARGETED:
				y_labels = np.zeros(cifar10_y_train.shape)
				y_labels[:,TARGET_CLASS] = 1
			else:
				y_labels = cifar10_y_train
			adv_cvae, adv_cdecoder = advcvae.attack(x=cifar10_X_train, cond=cifar10_y_train, y=y_labels, batch_size=32, epochs=EPOCHS, val_ratio=0.1)
			adv_cvae.save(f'snapshots/cifar10-adv-cvae-{CIFAR_VAE_DIM}d.h5')
			adv_cdecoder.save(f'snapshots/cifar10-adv-cdecoder-{CIFAR_VAE_DIM}d.h5')






		
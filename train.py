import numpy as np
from keras.datasets import mnist
from keras.models import Model, load_model
import argparse
import utils
from vae import VAE, ConvVAE, CVAE, advVAE, advCVAE
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

INPUT_SHAPE = (28,28,1)
IMG_SIZE = INPUT_SHAPE[0]
BATCH_SIZE = 32
COND_DIM = 10
VAE_DIM = 200			# Latent dimension of VAE which encodes the conditioning dataset
INTER_DIM = 512
EPOCHS = 100
TRAIN_VAE = False	# train a vae for begining or load a trained one
DEBUG =True

if __name__ == '__main__':
	# load mnist dataset for training and testing
	mnist_X_train, mnist_y_train, mnist_X_test, mnist_y_test = utils.load_dataset(dataset='mnist')
	cifar10_X_train, cifar10_y_train, cifar10_X_test, cifar10_y_test = utils.load_dataset(dataset='cifar10')
	#mnist_X_train = np.reshape(mnist_X_train, (-1, IMG_SIZE**2))
	#mnist_X_test = np.reshape(mnist_X_test, (-1, IMG_SIZE**2))

	if DEBUG == False:
		if TRAIN_VAE == True:
			model = VAE(IMG_SIZE, INTER_DIM, VAE_DIM)
			model.train(mnist_X_train, batch_size=BATCH_SIZE, epochs=EPOCHS, val_ratio=0.1)
			model.vae.save(f'snapshots/trained-vae-{VAE_DIM}d.h5')
			model.encoder.save(f'snapshots/trained-vae-encoder-{VAE_DIM}d.h5')
			model.decoder.save(f'snapshots/trained-vae-decoder-{VAE_DIM}d.h5')
			victim = VAE(IMG_SIZE, INTER_DIM+88, VAE_DIM)
			victim.train(mnist_X_train, batch_size=BATCH_SIZE, epochs=EPOCHS, val_ratio=0.1)
			victim.vae.save(f'snapshots/victim-vae-{VAE_DIM}d.h5')
			victim.encoder.save(f'snapshots/victim-vae-encoder-{VAE_DIM}d.h5')
			victim.decoder.save(f'snapshots/victim-vae-decoder-{VAE_DIM}d.h5')
			cvae = CVAE(IMG_SIZE, COND_DIM, INTER_DIM, VAE_DIM)
			cvae.train(mnist_X_train, mnist_y_train, batch_size=32, epochs=10, val_ratio=0.1)
			cvae.cvae.save(f'snapshots/trained-cvae-{VAE_DIM}d.h5')
			cvae.encoder.save(f'snapshots/trained-cvae-encoder-{VAE_DIM}d.h5')
			cvae.decoder.save(f'snapshots/trained-cvae-decoder-{VAE_DIM}d.h5')

		#loading trained vae for adv training
		elif TRAIN_VAE == False:
			print('===== Loading victim model =======')
			classifier = load_model('mnist_model.h5')
			print('===== Loading VAE =======')
			vae = load_model(f'./snapshots/trained-vae-{VAE_DIM}d.h5', compile=False)
			vae_encoder = load_model(f'./snapshots/trained-vae-encoder-{VAE_DIM}d.h5', compile=False)
			vae_decoder = load_model(f'./snapshots/trained-vae-decoder-{VAE_DIM}d.h5', compile=False)
			print(f'{VAE_DIM}-D VAE loaded.')
			vae.summary()
			advvae = advVAE(vae_encoder, vae_decoder, classifier)
			adv_vae, adv_decoder = advvae.attack(mnist_X_train, mnist_y_train, batch_size=32, epochs=10, val_ratio=0.1)
			adv_vae.save(f'snapshots/adv-vae-{VAE_DIM}d.h5')
			adv_decoder.save(f'snapshots/adv-decoder-{VAE_DIM}d.h5')

			print('===== Loading CVAE =======')
			cvae = load_model(f'./snapshots/trained-cvae-{VAE_DIM}d.h5', compile=False)
			cvae_encoder = load_model(f'./snapshots/trained-cvae-encoder-{VAE_DIM}d.h5', compile=False)
			cvae_decoder = load_model(f'./snapshots/trained-cvae-decoder-{VAE_DIM}d.h5', compile=False)
			print(f'{VAE_DIM}-D CVAE loaded.')
			cvae.summary()
			advcvae = advCVAE(cvae_encoder, cvae_decoder, classifier)
			adv_cvae, adv_cdecoder = advcvae.attack(mnist_X_train, mnist_y_train, batch_size=32, epochs=10, val_ratio=0.1)
			adv_cvae.save(f'snapshots/adv-cvae-{VAE_DIM}d.h5')
			adv_cdecoder.save(f'snapshots/adv-cdecoder-{VAE_DIM}d.h5')

	else:
		model = ConvVAE(INPUT_SHAPE, VAE_DIM)
		model.train(cifar10_X_train, batch_size=BATCH_SIZE, epochs=EPOCHS, val_ratio=0.1)
		model.vae.save(f'snapshots/cifar10-convo-vae-{VAE_DIM}d.h5')
		model.encoder.save(f'snapshots/cifar10-convo-vae-encoder-{VAE_DIM}d.h5')
		model.decoder.save(f'snapshots/cifar10-convo-vae-decoder-{VAE_DIM}d.h5')


		
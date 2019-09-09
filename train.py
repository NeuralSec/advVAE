import numpy as np
from keras.datasets import mnist
from keras.models import Model, load_model
import argparse
import utils
from vae import VAE, advVAE
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

INPUT_SHAPE = (28,28,1)
IMG_SIZE = INPUT_SHAPE[0]
BATCH_SIZE = 32
VAE_DIM = 2			# Latent dimension of VAE which encodes the conditioning dataset
EPOCHS = 10
TRAIN_VAE = True	# train a vae for begining or load a trained one

if __name__ == '__main__':
	# load mnist dataset for training and testing
	mnist_X_train, mnist_y_train, mnist_X_test, mnist_y_test = utils.load_dataset(dataset='mnist')
	mnist_X_train = np.reshape(mnist_X_train, (-1, IMG_SIZE**2))
	mnist_X_test = np.reshape(mnist_X_test, (-1, IMG_SIZE**2))

	if TRAIN_VAE == True:
		model = VAE(INPUT_SHAPE, VAE_DIM)
		model.train(mnist_X_train, batch_size=BATCH_SIZE, epochs=EPOCHS, val_ratio=0.1)
		model.vae.save(f'snapshots/trained-vae-{VAE_DIM}d.h5')
		model.encoder.save(f'snapshots/trained-vae-encoder-{VAE_DIM}d.h5')
		model.decoder.save(f'snapshots/trained-vae-decoder-{VAE_DIM}d.h5')

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
		advvae.attack(mnist_X_train, mnist_y_train, batch_size=32, epochs=10, val_ratio=0.1)
		advvae.adv_vae.save(f'snapshots/adv-vae-{VAE_DIM}d.h5')

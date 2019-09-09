import numpy as np
from keras.datasets import mnist
import argparse
import utils
from vae import VAE

INPUT_SHAPE = (28,28,1)
IMG_SIZE = 28
BATCH_SIZE = 32
VAE_DIM = 2			# Latent dimension of VAE which encodes the conditioning dataset
TRAINING = False		# True: train the vae and the cvae. False: load the vae and the cvae

if __name__ == '__main__':
	# load mnist dataset for training and testing
	mnist_X_train, mnist_y_train, mnist_X_test, mnist_y_test = utils.load_dataset(dataset='mnist')
	model = VAE(INPUT_SHAPE, VAE_DIM)
	model.train(mnist_X_train, batch_size=32, epochs=10, val_ratio=0.1)
	model.vae.save(f'snapshost/trained_vae.h5')
	model.encoder.save(f'snapshost/trained_vae_encoder.h5')
	model.decoder.save(f'snapshost/trained_vae_decoder.h5')

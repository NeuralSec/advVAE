import numpy as np
from keras.datasets import mnist
import argparse
import utils
from vae import VAE
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

INPUT_SHAPE = (28,28,1)
BATCH_SIZE = 32
VAE_DIM = 2			# Latent dimension of VAE which encodes the conditioning dataset
EPOCHS = 10

if __name__ == '__main__':
	# load mnist dataset for training and testing
	mnist_X_train, mnist_y_train, mnist_X_test, mnist_y_test = utils.load_dataset(dataset='mnist')
	model = VAE(INPUT_SHAPE, VAE_DIM)
	model.train(mnist_X_train, batch_size=BATCH_SIZE, epochs=EPOCHS, val_ratio=0.1)
	model.vae.save(f'snapshots/trained_vae.h5')
	model.encoder.save(f'snapshots/trained_vae_encoder.h5')
	model.decoder.save(f'snapshots/trained_vae_decoder.h5')

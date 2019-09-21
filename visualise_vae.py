import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model
import numpy as np
import utils
import os
from train import DATA, MNIST_VAE_DIM, CIFAR_VAE_DIM, TARGETED, TARGET_CLASS, VAE_NUM
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

if DATA == 'mnist':
	VAE_DIM = MNIST_VAE_DIM
elif DATA == 'cifar10':
	VAE_DIM = CIFAR_VAE_DIM

def plot(vae, vae_encoder, victim_vae, victim_encoder, advvae, adv_decoder, X_test, y_test, dataset='mnist'):
	# Plotting
	outputs = vae.predict(X_test[:10])
	latent_codes = vae_encoder.predict(X_test[:10])[2]
	victim_outputs = victim_vae.predict(X_test[:10])
	victim_codes = victim_encoder.predict(X_test[:10])[2]
	adv_outputs = advvae.predict(X_test[:10])
	victim_advs = adv_decoder.predict(victim_codes)
	if dataset=='mnist':
		random_codes = np.random.normal(0,1,(10,2))
	if dataset=='cifar10':
		random_codes = np.random.normal(0,1,(10,64))
	random_advs = adv_decoder.predict(random_codes)
	fig, axes = plt.subplots(nrows=6, ncols=10)
	for i in range(10):
		if dataset=='mnist':
			axes[0][i].imshow(X_test[:10][i].reshape(28,28), cmap='gray')
			axes[1][i].imshow(outputs[i].reshape(28,28), cmap='gray')
			axes[2][i].imshow(victim_outputs[i].reshape(28,28), cmap='gray')
			axes[3][i].imshow(adv_outputs[i].reshape(28,28), cmap='gray')
			axes[4][i].imshow(random_advs[i].reshape(28,28), cmap='gray')
			axes[5][i].imshow(victim_advs[i].reshape(28,28), cmap='gray')
			axes[5][i].set_xlabel(f'{np.argmax(y_test[:10][i])}')
		if dataset=='cifar10':
			axes[0][i].imshow(X_test[:10][i])
			axes[1][i].imshow(outputs[i])
			axes[2][i].imshow(victim_outputs[i])
			axes[3][i].imshow(adv_outputs[i])
			axes[4][i].imshow(random_advs[i])
			axes[5][i].imshow(victim_advs[i])
			axes[5][i].set_xlabel(f'{np.argmax(y_test[:10][i])}')

	axes[0][5].set_title('Inputs')
	axes[1][5].set_title('Shadiow VAE (Benign) Outputs')
	axes[2][5].set_title('Victim VAE Outputs')
	axes[3][5].set_title('White-Box-Encoder Attacks')
	axes[4][5].set_title('Random Generated Attacks')
	axes[5][5].set_title('Black-Box-Encoder Attacks')
	for i in range(10):
		[axes[j][i].axis('off') for j in range(6)]
	plt.show()

mnist_X_train_cnn, mnist_y_train, mnist_X_test_cnn, mnist_y_test = utils.load_dataset(dataset='mnist')
mnist_X_train = np.reshape(mnist_X_train_cnn, (-1, 28**2))
mnist_X_test = np.reshape(mnist_X_test_cnn, (-1, 28**2))
cifar10_X_train, cifar10_y_train, cifar10_X_test, cifar10_y_test = utils.load_dataset(dataset='cifar10')

substitute = load_model(f'{DATA}_substitute.h5')
classifier = load_model(f'{DATA}_model.h5')
victim_vae = load_model(f'snapshots/victim-{DATA}-vae-{VAE_DIM}d.h5', compile=False)
victim_encoder = load_model(f'snapshots/victim-{DATA}-vae-encoder-{VAE_DIM}d.h5', compile=False)
vae = load_model(f'snapshots/{DATA}-vae-{VAE_DIM}d.h5', compile=False)
vae_encoder = load_model(f'snapshots/{DATA}-vae-encoder-{VAE_DIM}d.h5', compile=False)
vae_decoder = load_model(f'snapshots/{DATA}-vae-decoder-{VAE_DIM}d.h5', compile=False)
advvae = load_model(f'snapshots/{DATA}-adv-vae-{VAE_DIM}d.h5', compile=False)
adv_decoder = load_model(f'snapshots/{DATA}-adv-decoder-{VAE_DIM}d.h5', compile=False)
adv_decoder.summary()
egnostic_advvae_decoder = load_model(f'snapshots/{DATA}-egnostic_adv-decoder-{VAE_DIM}d-{VAE_NUM}encoders.h5', compile=False)
egnostic_advvae_decoder.summary()


# Evaluation
if DATA == 'mnist':
	if TARGETED:
		y_labels = np.zeros(mnist_y_test.shape)
		y_labels[:,TARGET_CLASS] = 1
		print('**********************Black-box targeted attacks eval (mnist)******************************')
		print(f'Predicted labels of the first ten examples (black-box-classifier setting): {np.argmax(classifier.predict(advvae.predict(mnist_X_test[:10]).reshape((-1,28,28,1))), axis=-1)}')
		print(f'Accuracy in the black-box-classifier setting: {classifier.evaluate(advvae.predict(mnist_X_test).reshape((-1,28,28,1)), y_labels)}')
		print(f'Accuracy in the black-box-encoder setting: {substitute.evaluate(adv_decoder.predict(victim_encoder.predict(mnist_X_test)[2]).reshape((-1,28,28,1)), y_labels)}')
		print(f'Accuracy in the black-box-encoder setting (encoder egnostic): {substitute.evaluate(egnostic_advvae_decoder.predict(victim_encoder.predict(mnist_X_test)[2]).reshape((-1,28,28,1)), y_labels)}')
		print(f'Accuracy in the double-black-box setting: {classifier.evaluate(adv_decoder.predict(victim_encoder.predict(mnist_X_test)[2]).reshape((-1,28,28,1)), y_labels)}')
		print(f'Accuracy in the double-black-box setting (encoder egnostic): {classifier.evaluate(egnostic_advvae_decoder.predict(victim_encoder.predict(mnist_X_test)[2]).reshape((-1,28,28,1)), y_labels)}')
		
		print('**********************White-box targeted attacks eval (mnist)******************************')
		print(f'Predicted labels of the first ten examples (double-white-box setting): {np.argmax(substitute.predict(advvae.predict(mnist_X_test[:10]).reshape((-1,28,28,1))), axis=-1)}')
		print(f'Accuracy in the double-white-box setting: {substitute.evaluate(advvae.predict(mnist_X_test).reshape((-1,28,28,1)), y_labels)}')
		
		# Plotting
		plot(vae, vae_encoder, victim_vae, victim_encoder, advvae, adv_decoder, mnist_X_test, y_labels, dataset='mnist')

	else:
		print('**********************Black-box non-targeted attacks eval (mnist)******************************')
		print(f'Predicted labels of the first ten examples (black-box-classifier setting): {np.argmax(classifier.predict(advvae.predict(mnist_X_test[:10]).reshape((-1,28,28,1))), axis=-1)}')
		print(f'Accuracy in the black-box-classifier setting: {classifier.evaluate(advvae.predict(mnist_X_test).reshape((-1,28,28,1)), mnist_y_test)}')
		print(f'Accuracy in the black-box-encoder setting: {substitute.evaluate(adv_decoder.predict(victim_encoder.predict(mnist_X_test)[2]).reshape((-1,28,28,1)), mnist_y_test)}')
		print(f'Accuracy in the black-box-encoder setting (encoder egnostic): {substitute.evaluate(egnostic_advvae_decoder.predict(victim_encoder.predict(mnist_X_test)[2]).reshape((-1,28,28,1)), mnist_y_test)}')
		print(f'Accuracy in the double-black-box setting: {classifier.evaluate(adv_decoder.predict(victim_encoder.predict(mnist_X_test)[2]).reshape((-1,28,28,1)), mnist_y_test)}')
		print(f'Accuracy in the double-black-box setting (encoder egnostic): {classifier.evaluate(egnostic_advvae_decoder.predict(victim_encoder.predict(mnist_X_test)[2]).reshape((-1,28,28,1)), mnist_y_test)}')
		
		print('**********************White-box non-targeted attacks eval (mnist)******************************')
		print(f'Predicted labels of the first ten examples (double-white-box setting): {np.argmax(substitute.predict(advvae.predict(mnist_X_test[:10]).reshape((-1,28,28,1))), axis=-1)}')
		print(f'Accuracy in the double-white-box setting: {substitute.evaluate(advvae.predict(mnist_X_test).reshape((-1,28,28,1)), mnist_y_test)}')
	
		# Plotting
		plot(vae, vae_encoder, victim_vae, victim_encoder, advvae, egnostic_advvae_decoder, mnist_X_test, mnist_y_test, dataset='mnist')


elif DATA == 'cifar10':
	if TARGETED:
		y_labels = np.zeros(cifar10_y_test.shape)
		y_labels[:,TARGET_CLASS] = 1
		print('**********************Black-box targeted attacks eval (cifar10)******************************')
		print(f'Predicted labels of the first ten examples (black-box-classifier setting): {np.argmax(classifier.predict(advvae.predict(cifar10_X_test[:10])), axis=-1)}')
		print(f'Accuracy in the black-box-classifier setting: {classifier.evaluate(advvae.predict(cifar10_X_test), y_labels)}')
		print(f'Accuracy in the black-box-encoder setting: {substitute.evaluate(adv_decoder.predict(victim_encoder.predict(cifar10_X_test)[2]), y_labels)}')
		print(f'Accuracy in the black-box-encoder setting (encoder egnostic): {substitute.evaluate(egnostic_advvae_decoder.predict(victim_encoder.predict(cifar10_X_test)[2]), y_labels)}')
		print(f'Accuracy in the double-black-box setting: {classifier.evaluate(adv_decoder.predict(victim_encoder.predict(cifar10_X_test)[2]), y_labels)}')
		print(f'Accuracy in the double-black-box setting (encoder egnostic): {classifier.evaluate(egnostic_advvae_decoder.predict(victim_encoder.predict(cifar10_X_test)[2]), y_labels)}')
		
		print('**********************White-box targeted attacks eval (cifar10)******************************')
		print(f'Predicted labels of the first ten examples (double-white-box setting): {np.argmax(substitute.predict(advvae.predict(cifar10_X_test[:10])), axis=-1)}')
		print(f'Accuracy in the double-white-box setting: {substitute.evaluate(advvae.predict(cifar10_X_test), y_labels)}')
		
		# Plotting
		plot(vae, vae_encoder, victim_vae, victim_encoder, advvae, adv_decoder, cifar10_X_test, y_labels, dataset='cifar10')

	else:
		print('**********************Black-box non-targeted attacks eval (cifar10)******************************')
		print(f'Predicted labels of the first ten examples (black-box-classifier setting): {np.argmax(classifier.predict(advvae.predict(cifar10_X_test[:10])), axis=-1)}')
		print(f'Accuracy in the black-box-classifier setting: {classifier.evaluate(advvae.predict(cifar10_X_test), cifar10_y_test)}')
		print(f'Accuracy in the black-box-encoder setting: {substitute.evaluate(adv_decoder.predict(victim_encoder.predict(cifar10_X_test)[2]), cifar10_y_test)}')
		print(f'Accuracy in the black-box-encoder setting (encoder egnostic): {substitute.evaluate(egnostic_advvae_decoder.predict(victim_encoder.predict(cifar10_X_test)[2]), cifar10_y_test)}')
		print(f'Accuracy in the double-black-box setting: {classifier.evaluate(adv_decoder.predict(victim_encoder.predict(cifar10_X_test)[2]), cifar10_y_test)}')
		print(f'Accuracy in the double-black-box setting (encoder egnostic): {classifier.evaluate(egnostic_advvae_decoder.predict(victim_encoder.predict(cifar10_X_test)[2]), cifar10_y_test)}')
		
		print('**********************White-box non-targeted attacks eval (cifar10)******************************')
		print(f'Predicted labels of the first ten examples (double-white-box setting): {np.argmax(substitute.predict(advvae.predict(cifar10_X_test[:10])), axis=-1)}')
		print(f'Accuracy in the double-white-box setting: {substitute.evaluate(advvae.predict(cifar10_X_test), cifar10_y_test)}')
	
		# Plotting
		plot(vae, vae_encoder, victim_vae, victim_encoder, advvae, adv_decoder, cifar10_X_test, cifar10_y_test, dataset='cifar10')



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model
import numpy as np
import utils
import os
from train import DATA, MNIST_VAE_DIM, CIFAR_VAE_DIM, TARGETED, TARGET_CLASS
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

if DATA == 'mnist':
	VAE_DIM = MNIST_VAE_DIM
elif DATA == 'cifar10':
	VAE_DIM = CIFAR_VAE_DIM

mnist_X_train_cnn, mnist_y_train, mnist_X_test_cnn, mnist_y_test = utils.load_dataset(dataset='mnist')
mnist_X_train = np.reshape(mnist_X_train_cnn, (-1, 28**2))
mnist_X_test = np.reshape(mnist_X_test_cnn, (-1, 28**2))
cifar10_X_train, cifar10_y_train, cifar10_X_test, cifar10_y_test = utils.load_dataset(dataset='cifar10')

victim = load_model(f'snapshots/victim-{DATA}-vae-{VAE_DIM}d.h5', compile=False)
victim.summary()
victim_encoder = load_model(f'snapshots/victim-{DATA}-vae-encoder-{VAE_DIM}d.h5', compile=False)
cvae = load_model(f'snapshots/{DATA}-cvae-{VAE_DIM}d.h5', compile=False)
cvae_encoder = load_model(f'snapshots/{DATA}-cvae-encoder-{VAE_DIM}d.h5', compile=False)
cvae_decoder = load_model(f'snapshots/{DATA}-cvae-decoder-{VAE_DIM}d.h5', compile=False)
advCvae = load_model(f'snapshots/{DATA}-adv-cvae-{VAE_DIM}d.h5', compile=False)
adv_Cdecoder = load_model(f'snapshots/{DATA}-adv-cdecoder-{VAE_DIM}d.h5', compile=False)
adv_Cdecoder.summary()
substitute = load_model(f'{DATA}_substitute.h5')
substitute.summary()
classifier = load_model(f'{DATA}_model.h5')
classifier.summary()

# Evaluation
if DATA == 'mnist':
	if TARGETED:
		y_labels = np.zeros(mnist_y_test.shape)
		y_labels[:,TARGET_CLASS] = 1
		print('**********************Black-box targeted attacks eval (mnist)******************************')
		print(np.argmax(classifier.predict(advCvae.predict([mnist_X_test[:10], mnist_y_test[:10]]).reshape((-1,28,28,1))), axis=-1))
		print(classifier.evaluate(advCvae.predict([mnist_X_test, mnist_y_test]).reshape((-1,28,28,1)), y_labels))
		print(classifier.evaluate(adv_Cdecoder.predict([victim_encoder.predict(mnist_X_test)[2], mnist_y_test]).reshape((-1,28,28,1)), y_labels))
		print('**********************White-box targeted attacks eval (mnist)******************************')
		print(np.argmax(substitute.predict(advCvae.predict([mnist_X_test[:10], mnist_y_test[:10]]).reshape((-1,28,28,1))), axis=-1))
		print(substitute.evaluate(advCvae.predict([mnist_X_test, mnist_y_test]).reshape((-1,28,28,1)), y_labels))
		print(substitute.evaluate(adv_Cdecoder.predict([victim_encoder.predict(mnist_X_test)[2], mnist_y_test]).reshape((-1,28,28,1)), y_labels))
	else:
		print('**********************Black-box non-targeted attacks eval (mnist)******************************')
		print(np.argmax(classifier.predict(advCvae.predict([mnist_X_test[:10], mnist_y_test[:10]]).reshape((-1,28,28,1))), axis=-1))
		print(classifier.evaluate(advCvae.predict([mnist_X_test, mnist_y_test]).reshape((-1,28,28,1)), mnist_y_test))
		print(classifier.evaluate(adv_Cdecoder.predict([victim_encoder.predict(mnist_X_test)[2], mnist_y_test]).reshape((-1,28,28,1)), mnist_y_test))
		print('**********************White-box non-targeted attacks eval (mnist)******************************')
		print(np.argmax(substitute.predict(advCvae.predict([mnist_X_test[:10], mnist_y_test[:10]]).reshape((-1,28,28,1))), axis=-1))
		print(substitute.evaluate(advCvae.predict([mnist_X_test, mnist_y_test]).reshape((-1,28,28,1)), mnist_y_test))
		print(substitute.evaluate(adv_Cdecoder.predict([victim_encoder.predict(mnist_X_test)[2], mnist_y_test]).reshape((-1,28,28,1)), mnist_y_test))

	# Plotting
	outputs = cvae.predict([mnist_X_test[:10], mnist_y_test[:10]])
	victim_outputs = victim.predict(mnist_X_test[:10])
	adv_outputs = advCvae.predict([mnist_X_test[:10], mnist_y_test[:10]])
	victim_codes = victim_encoder.predict(mnist_X_test[:10])[2]
	victim_advs = adv_Cdecoder.predict([victim_codes, mnist_y_test[:10]])
	random_codes = np.random.normal(0,1,(10,2))
	random_advs = adv_Cdecoder.predict([random_codes, mnist_y_test[:10]])
	fig, axes = plt.subplots(nrows=6, ncols=10)
	for i in range(10):
		axes[0][i].imshow(mnist_X_test[:10][i].reshape(28,28), cmap='gray')
		axes[1][i].imshow(outputs[i].reshape(28,28), cmap='gray')
		axes[2][i].imshow(victim_outputs[i].reshape(28,28), cmap='gray')
		axes[3][i].imshow(adv_outputs[i].reshape(28,28), cmap='gray')
		axes[4][i].imshow(random_advs[i].reshape(28,28), cmap='gray')
		axes[5][i].imshow(victim_advs[i].reshape(28,28), cmap='gray')

	axes[0][5].set_title('Inputs')
	axes[1][5].set_title('Benign CVAE Outputs')
	axes[2][5].set_title('Blackbox Victim VAE Outputs')
	axes[3][5].set_title('Adversarial CVAE Outputs')
	axes[4][5].set_title('Randomly Generated Adversarial CVAE Outputs')
	axes[5][5].set_title('Adversarial CVAE Outputs based on Victim Encoder')
	for i in range(10):
		[axes[j][i].axis('off') for j in range(6)]
	plt.show()

elif DATA == 'cifar10':
	if TARGETED:
		y_labels = np.zeros(cifar10_y_test.shape)
		y_labels[:,TARGET_CLASS] = 1
		print('**********************Black-box targeted attacks eval (cifar10)******************************')
		print(np.argmax(classifier.predict(advCvae.predict([cifar10_X_test[:10], cifar10_y_test[:10]])), axis=-1))
		print(classifier.evaluate(advCvae.predict([cifar10_X_test, cifar10_y_test]), y_labels))
		print(classifier.evaluate(adv_Cdecoder.predict([victim_encoder.predict(cifar10_X_test)[2], cifar10_y_test]), y_labels))
		print('**********************White-box targeted attacks eval (cifar10)******************************')
		print(np.argmax(substitute.predict(advCvae.predict([cifar10_X_test[:10], cifar10_y_test[:10]])), axis=-1))
		print(substitute.evaluate(advCvae.predict([cifar10_X_test, cifar10_y_test]), y_labels))
		print(substitute.evaluate(adv_Cdecoder.predict([victim_encoder.predict(cifar10_X_test)[2], cifar10_y_test]), y_labels))
	else:
		print('**********************Black-box non-targeted attacks eval (cifar10)******************************')
		print(np.argmax(classifier.predict(advCvae.predict([cifar10_X_test[:10], cifar10_y_test[:10]])), axis=-1))
		print(classifier.evaluate(advCvae.predict([cifar10_X_test, cifar10_y_test]), cifar10_y_test))
		print(classifier.evaluate(adv_Cdecoder.predict([victim_encoder.predict(cifar10_X_test)[2], cifar10_y_test]), cifar10_y_test))
		print('**********************White-box non-targeted attacks eval (cifar10)******************************')
		print(np.argmax(substitute.predict(advCvae.predict([cifar10_X_test[:10], cifar10_y_test[:10]])), axis=-1))
		print(substitute.evaluate(advCvae.predict([cifar10_X_test, cifar10_y_test]), cifar10_y_test))
		print(substitute.evaluate(adv_Cdecoder.predict([victim_encoder.predict(cifar10_X_test)[2], cifar10_y_test]), cifar10_y_test))
	# Plotting
	outputs = cvae.predict([cifar10_X_test[:10], cifar10_y_test[:10]])
	victim_outputs = victim.predict(cifar10_X_test[:10])
	adv_outputs = advCvae.predict([cifar10_X_test[:10], cifar10_y_test[:10]])
	victim_codes = victim_encoder.predict(cifar10_X_test[:10])[2]
	victim_advs = adv_Cdecoder.predict([victim_codes, cifar10_y_test[:10]])
	random_codes = np.random.normal(0,1,(10,64))
	random_advs = adv_Cdecoder.predict([random_codescifar10_y_test[:10]])
	fig, axes = plt.subplots(nrows=6, ncols=10)
	for i in range(10):
		axes[0][i].imshow(cifar10_X_test[:10][i])
		axes[1][i].imshow(outputs[i])
		axes[2][i].imshow(victim_outputs[i])
		axes[3][i].imshow(adv_outputs[i])
		axes[4][i].imshow(random_advs[i])
		axes[5][i].imshow(victim_advs[i])

	axes[0][5].set_title('Inputs')
	axes[1][5].set_title('Benign CVAE Outputs')
	axes[2][5].set_title('Blackbox Victim VAE Outputs')
	axes[3][5].set_title('Adversarial CVAE Outputs')
	axes[4][5].set_title('Randomly Generated Adversarial CVAE Outputs')
	axes[5][5].set_title('Adversarial CVAE Outputs based on Victim Encoder')
	for i in range(10):
		[axes[j][i].axis('off') for j in range(6)]
	plt.show()

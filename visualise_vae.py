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

if TARGETED:
	TITLE = f'Targeted to {TARGET_CLASS}'
else:
	TITLE = 'Non-Targeted'
LOOSE = False


def classifier_only_eval_set_selection(X, y, model):
	pred_labels = np.argmax(model.predict(X), axis=-1)
	true_labels = np.argmax(y, axis=-1)
	inds = np.where(pred_labels == true_labels)
	return X[inds][:1000], y[inds][:1000]

def mnist_eval_set_selection(X, y, model, vae):
	pred_labels = np.argmax(model.predict(vae.predict(X).reshape((-1,28,28,1))), axis=-1)
	true_labels = np.argmax(y, axis=-1)
	inds = np.where(pred_labels == true_labels)
	return X[inds][:1000], y[inds][:1000]

def cifar10_eval_set_selection(X, y, model, vae):
	pred_labels = np.argmax(model.predict(vae.predict(X)), axis=-1)
	true_labels = np.argmax(y, axis=-1)
	inds = np.where(pred_labels == true_labels)
	return X[inds][:1000], y[inds][:1000]

def plot(vae, vae_encoder, victim_vae, victim_encoder, advvae, adv_decoder, classifier, X_test, y_test, title='', dataset='mnist'):
	# Plotting
	benign_vae_outputs = vae.predict(X_test[:10])
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
			X_test = X_test.reshape(-1, 28, 28, 1)
			benign_vae_outputs = benign_vae_outputs.reshape(-1, 28, 28, 1)
			victim_outputs = victim_outputs.reshape(-1, 28, 28, 1)
			adv_outputs = adv_outputs.reshape(-1, 28, 28, 1)
			random_advs = random_advs.reshape(-1, 28, 28, 1)
			victim_advs = victim_advs.reshape(-1, 28, 28, 1)
			axes[0][i].imshow(X_test[:10][i].reshape(28, 28), cmap='gray')
			axes[1][i].imshow(benign_vae_outputs[i].reshape(28, 28), cmap='gray')
			axes[2][i].imshow(victim_outputs[i].reshape(28, 28), cmap='gray')
			axes[3][i].imshow(adv_outputs[i].reshape(28, 28), cmap='gray')
			axes[4][i].imshow(random_advs[i].reshape(28, 28), cmap='gray')
			axes[5][i].imshow(victim_advs[i].reshape(28, 28), cmap='gray')
		
		if dataset=='cifar10':
			axes[0][i].imshow(X_test[:10][i])
			axes[1][i].imshow(benign_vae_outputs[i])
			axes[2][i].imshow(victim_outputs[i])
			axes[3][i].imshow(adv_outputs[i])
			axes[4][i].imshow(random_advs[i])
			axes[5][i].imshow(victim_advs[i])
	for ax in axes:
		[a.set_xticks([]) for a in ax]
		[a.set_yticks([]) for a in ax]
	cols = []
	cols.append([f'{t}' for t in np.argmax(y_test[:10], axis=-1)])
	cols.append([f'{t}' for t in np.argmax(classifier.predict(benign_vae_outputs), axis=-1)])
	cols.append([f'{t}' for t in np.argmax(classifier.predict(victim_outputs), axis=-1)])
	cols.append([f'{t}' for t in np.argmax(classifier.predict(adv_outputs), axis=-1)])
	cols.append([f'{t}' for t in np.argmax(classifier.predict(random_advs), axis=-1)])
	cols.append([f'{t}' for t in np.argmax(classifier.predict(victim_advs), axis=-1)])
	rows = ['Original', 'Shadow VAE (Benign) Outputs', 'Victim VAE Outputs', 'White-Box-Encoder Attacks', 'Random Generated Attacks', 'Black-Box-Encoder Attacks']
	for ax, col in zip(axes, cols):
		for a, c in zip(ax, col):
			a.set_title(c)
	for ax, row in zip(axes[:,0], rows):
		ax.set_ylabel(row, rotation=60, size=6)
	fig.suptitle(title, fontsize=16)
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
	# Select the examples that can be correctly classified after being reconstructed by the benign vae as the evaluation datasets.
	if LOOSE:
		classifier_test_X, classifier_test_y = classifier_only_eval_set_selection(mnist_X_test, mnist_y_test, classifier)
		substitute_test_X, substitute_test_y = classifier_only_eval_set_selection(mnist_X_test, mnist_y_test, substitute)
	else:
		classifier_test_X, classifier_test_y = mnist_eval_set_selection(mnist_X_test, mnist_y_test, classifier, vae)
		substitute_test_X, substitute_test_y = mnist_eval_set_selection(mnist_X_test, mnist_y_test, substitute, vae)
	if TARGETED:
		classifier_y_labels = np.zeros(classifier_test_y.shape)
		classifier_y_labels[:,TARGET_CLASS] = 1
		substitute_y_labels = np.zeros(substitute_test_y.shape)
		substitute_y_labels[:,TARGET_CLASS] = 1
	else:
		classifier_y_labels = classifier_test_y
		substitute_y_labels = substitute_test_y
	
	# predictions
	black_box_classifier_pred_benign = np.argmax(classifier.predict(vae.predict(classifier_test_X).reshape((-1,28,28,1))), axis=-1)
	double_white_box_pred_benign = np.argmax(substitute.predict(vae.predict(substitute_test_X).reshape((-1,28,28,1))), axis=-1)

	black_box_classifier_pred = classifier.predict(advvae.predict(classifier_test_X).reshape((-1,28,28,1)))
	black_box_encoder_pred = substitute.predict(advvae.predict(substitute_test_X).reshape((-1,28,28,1)))
	double_black_box_pred = classifier.predict(adv_decoder.predict(victim_encoder.predict(classifier_test_X)[2]).reshape((-1,28,28,1)))
	double_white_box_pred = substitute.predict(advvae.predict(substitute_test_X).reshape((-1,28,28,1)))

	# Evaluations (benign)
	print(f'\n**********************Benign eval (mnist)******************************')
	utils.evaluations(np.argmax(classifier_y_labels, axis=-1), black_box_classifier_pred_benign, name='black_box_benign')
	utils.evaluations(np.argmax(substitute_y_labels, axis=-1), double_white_box_pred_benign, name='white_box_benign')
	# Evaluations (adversarials)
	print(f'\n**********************Black-box targeted:{TARGETED} attacks eval (mnist)******************************')
	utils.evaluations(np.argmax(classifier_y_labels, axis=-1), np.argmax(black_box_classifier_pred, axis=-1), name='black_box_classifier')
	utils.evaluations(np.argmax(substitute_y_labels, axis=-1), np.argmax(black_box_encoder_pred, axis=-1), name='black_box_encoder')
	utils.evaluations(np.argmax(classifier_y_labels, axis=-1), np.argmax(double_black_box_pred, axis=-1), name='double_black_box')
	print(f'\n**********************White-box targeted:{TARGETED} attacks eval (mnist)******************************')
	utils.evaluations(np.argmax(substitute_y_labels, axis=-1), np.argmax(double_white_box_pred, axis=-1), name='double_white_box')

	# Plotting on black-box and white-box test sets
	plot(vae, vae_encoder, victim_vae, victim_encoder, advvae, adv_decoder, classifier, classifier_test_X, classifier_y_labels, title=f'Black-box Attacks ({TITLE})', dataset='mnist')
	plot(vae, vae_encoder, victim_vae, victim_encoder, advvae, adv_decoder, substitute, substitute_test_X, substitute_y_labels, title=f'White-box Attacks ({TITLE})', dataset='mnist')

elif DATA == 'cifar10':
	# Select the examples that can be correctly classified after being reconstructed by the benign vae as the evaluation datasets.
	if LOOSE:
		classifier_test_X, classifier_test_y = classifier_only_eval_set_selection(cifar10_X_test, cifar10_y_test, classifier)
		substitute_test_X, substitute_test_y = classifier_only_eval_set_selection(cifar10_X_test, cifar10_y_test, substitute)
	else:
		classifier_test_X, classifier_test_y = cifar10_eval_set_selection(cifar10_X_test, cifar10_y_test, classifier, vae)
		substitute_test_X, substitute_test_y = cifar10_eval_set_selection(cifar10_X_test, cifar10_y_test, substitute, vae)
	if TARGETED:
		classifier_y_labels = np.zeros(classifier_test_y.shape)
		classifier_y_labels[:,TARGET_CLASS] = 1
		substitute_y_labels = np.zeros(substitute_test_y.shape)
		substitute_y_labels[:,TARGET_CLASS] = 1
	else:
		classifier_y_labels = classifier_test_y
		substitute_y_labels = substitute_test_y
	
	# predictions
	black_box_classifier_pred_benign = np.argmax(classifier.predict(vae.predict(classifier_test_X)), axis=-1)
	double_white_box_pred_benign = np.argmax(substitute.predict(vae.predict(substitute_test_X)), axis=-1)

	black_box_classifier_pred = classifier.predict(advvae.predict(classifier_test_X))
	black_box_encoder_pred = substitute.predict(advvae.predict(substitute_test_X))
	double_black_box_pred = classifier.predict(adv_decoder.predict(victim_encoder.predict(classifier_test_X)[2]))
	double_white_box_pred = substitute.predict(advvae.predict(substitute_test_X))

	# Evaluations (benign)
	print(f'\n**********************Benign eval (cifar10)******************************')
	utils.evaluations(np.argmax(classifier_y_labels, axis=-1), black_box_classifier_pred_benign, name='black_box_benign')
	utils.evaluations(np.argmax(substitute_y_labels, axis=-1), double_white_box_pred_benign, name='white_box_benign')
	# Evaluations (adversarials)
	print(f'\n**********************Black-box targeted:{TARGETED} attacks eval (cifar10)******************************')
	utils.evaluations(np.argmax(classifier_y_labels, axis=-1), np.argmax(black_box_classifier_pred, axis=-1), name='black_box_classifier')
	utils.evaluations(np.argmax(substitute_y_labels, axis=-1), np.argmax(black_box_encoder_pred, axis=-1), name='black_box_encoder')
	utils.evaluations(np.argmax(classifier_y_labels, axis=-1), np.argmax(double_black_box_pred, axis=-1), name='double_black_box')
	print(f'\n**********************White-box targeted:{TARGETED} attacks eval (cifar10)******************************')
	utils.evaluations(np.argmax(substitute_y_labels, axis=-1), np.argmax(double_white_box_pred, axis=-1), name='double_white_box')

	# Plotting on black-box and white-box test sets
	plot(vae, vae_encoder, victim_vae, victim_encoder, advvae, adv_decoder, classifier, classifier_test_X, classifier_y_labels, title=f'Black-box Attacks ({TITLE})', dataset='cifar10')
	plot(vae, vae_encoder, victim_vae, victim_encoder, advvae, adv_decoder, substitute, substitute_test_X, substitute_y_labels, title=f'White-box Attacks ({TITLE})', dataset='cifar10')


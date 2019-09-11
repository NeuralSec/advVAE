import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model
import numpy as np
import utils
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
VAE_DIM =2
IMG_SIZE = 28

mnist_X_train, mnist_y_train, mnist_X_test, mnist_y_test = utils.load_dataset(dataset='mnist')
mnist_X_train = np.reshape(mnist_X_train, (-1, IMG_SIZE**2))
mnist_X_test = np.reshape(mnist_X_test, (-1, IMG_SIZE**2))

victim = load_model(f'snapshots/victim-vae-{VAE_DIM}d.h5', compile=False)
victim_encoder = load_model(f'snapshots/victim-vae-encoder-{VAE_DIM}d.h5', compile=False)
cvae = load_model(f'snapshots/trained-cvae-{VAE_DIM}d.h5', compile=False)
cvae_encoder = load_model(f'snapshots/trained-cvae-encoder-{VAE_DIM}d.h5', compile=False)
cvae_decoder = load_model(f'snapshots/trained-cvae-decoder-{VAE_DIM}d.h5', compile=False)
advcvae = load_model(f'snapshots/adv-cvae-{VAE_DIM}d.h5', compile=False)
adv_cdecoder = load_model(f'snapshots/adv-cdecoder-{VAE_DIM}d.h5', compile=False)
classifier = load_model('mnist_model.h5')

# Evaluation
cond_data = mnist_y_test
print(classifier.evaluate(advcvae.predict([mnist_X_test, cond_data]).reshape((-1,28,28,1)), mnist_y_test))
print(classifier.evaluate(adv_cdecoder.predict([victim_encoder.predict(mnist_X_test)[2], cond_data]).reshape((-1,28,28,1)), mnist_y_test))

# Plotting
outputs = cvae.predict([mnist_X_test[:10], cond_data[:10]])
victim_outputs = victim.predict(mnist_X_test[:10])
latent_codes = cvae_encoder.predict([mnist_X_test[:10], cond_data[:10]])[2]
adv_outputs = advcvae.predict([mnist_X_test[:10], cond_data[:10]])

victim_codes = victim_encoder.predict(mnist_X_test[:10])[2]
victim_advs = adv_cdecoder.predict([victim_codes, cond_data[:10]])
random_codes = np.random.normal(0,1,(10,2))
random_advs = adv_cdecoder.predict([random_codes, cond_data[:10]])
fig, axes = plt.subplots(nrows=6, ncols=10)
for i in range(10):
	axes[0][i].imshow(mnist_X_test[:10][i].reshape(28,28), cmap='gray')
	axes[1][i].imshow(outputs[i].reshape(28,28), cmap='gray')
	axes[2][i].imshow(victim_outputs[i].reshape(28,28), cmap='gray')
	axes[3][i].imshow(adv_outputs[i].reshape(28,28), cmap='gray')
	axes[4][i].imshow(random_advs[i].reshape(28,28), cmap='gray')
	axes[5][i].imshow(victim_advs[i].reshape(28,28), cmap='gray')

axes[0][5].set_title('Inputs')
axes[1][5].set_title('Local CVAE Outputs')
axes[2][5].set_title('Blackbox Victim VAE Outputs')
axes[3][5].set_title('Adversarial Outputs')
axes[4][5].set_title('Random Adversarial Outputs')
axes[5][5].set_title('Victim Encoder Adversarial Outputs')
plt.show()

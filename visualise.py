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

vae = load_model(f'snapshots/trained-vae-{VAE_DIM}d.h5', compile=False)
vae_encoder = load_model(f'snapshots/trained-vae-encoder-{VAE_DIM}d.h5', compile=False)
vae_decoder = load_model(f'snapshots/trained-vae-decoder-{VAE_DIM}d.h5', compile=False)
advvae = load_model(f'snapshots/adv-vae-{VAE_DIM}d.h5', compile=False)
classifier = load_model('mnist_model.h5')
print(classifier.evaluate(advvae.predict(mnist_X_test).reshape((-1,28,28,1)), mnist_y_test))

outputs = vae.predict(mnist_X_test[:10])
latent_codes = vae_encoder.predict(mnist_X_test[:10])[2]
adv_outputs = advvae.predict(mnist_X_test[:10])

fig, axes = plt.subplots(nrows=4, ncols=10)
for i in range(10):
	axes[0][i].imshow(mnist_X_test[:10][i].reshape(28,28), cmap='gray')
	axes[1][i].scatter(latent_codes[i][0], latent_codes[i][1])
	axes[2][i].imshow(outputs[i].reshape(28,28), cmap='gray')
	axes[3][i].imshow(adv_outputs[i].reshape(28,28), cmap='gray')
axes[0][5].set_title('Inputs')
axes[1][5].set_title('Latent space')
axes[2][5].set_title('Outputs')
axes[2][5].set_title('Adversarial Outputs')
plt.show()

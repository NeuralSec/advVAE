from keras.datasets import mnist, cifar10
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_dataset(dataset='mnist'):
	if dataset == 'cifar10':
		(X_train, y_train), (X_test, y_test) = cifar10.load_data()
	elif dataset == 'mnist':
		(X_train, y_train), (X_test, y_test) = mnist.load_data()
		X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
		X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)		
	X_train = np.float32(X_train/255.)
	X_test = np.float32(X_test/255.)
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)
	print('data loaded, training x:{0}, y:{1}; test x:{2}, y{3}.'.format(X_train.shape,
																		 X_test.shape,
																		 y_train.shape,
																		 y_test.shape))
	return X_train, y_train, X_test, y_test

def load_imagenet_data(img_path='imagenet_data'):
	imgs = []
	for (root, dirs, files) in os.walk(img_path):
		if files:
			for f in files:
				print(f)
				path = os.path.join(root, f)
				img = image.load_img(path, target_size=(224, 224))
				imgs.append(image.img_to_array(img))
	return np.float32(np.stack(imgs)/255.)

def load_perturbation(data_type='train'):
	if data_type =='train':
		data = [np.load(f'./adv_mnist_data/training perturbation and original data of {i}.npz') for i in range(10)]
	elif data_type =='test':
		data = [np.load(f'./adv_mnist_data/testing perturbation and original data of {i}.npz') for i in range(10)]
	all_perturbations = np.concatenate([data[i]['delta'] for i in range(10)], axis=0)
	all_x = np.concatenate([data[i]['x'] for i in range(10)], axis=0)
	all_y = np.concatenate([data[i]['y'] for i in range(10)], axis=0)
	return all_perturbations, all_x, all_y

def evaluations(y_ture, y_pred, average='macro', name=''):
	accracy = accuracy_score(y_ture, y_pred)
	precision = precision_score(y_ture, y_pred, average=average)
	recall = recall_score(y_ture, y_pred, average=average)
	f1 = f1_score(y_ture, y_pred, average=average)
	print(f'The accuracy ín {name} setting is {accracy}')
	print(f'The precision ín {name} setting is {precision}')
	print(f'The recall ín {name} setting is {recall}')
	print(f'The f1_score ín {name} setting is {f1}')
	return accracy, precision, recall, f1
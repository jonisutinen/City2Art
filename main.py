from os import listdir, path
from random import random
import tensorflow as tf
from keras.initializers import RandomNormal
from keras.layers import Activation, Concatenate, Conv2D, Conv2DTranspose, LeakyReLU
from keras.models import Input, Model
from keras.optimizers import Adam
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from matplotlib import pyplot
from numpy import asarray, load, ones, zeros
from numpy.random import randint
import loadimages

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	tf.config.experimental.set_memory_growth(gpus[0], True)
	tf.config.experimental.set_visible_devices(gpus[0], 'GPU')


def define_discriminator(image_shape):
	"""
	:param image_shape: image shape
	:return: model
	"""
	init = RandomNormal(stddev=0.02)
	in_image = Input(shape=image_shape)

	d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.2)(d)

	d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)

	d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)

	d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)

	d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)

	patch_out = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
	model = Model(in_image, patch_out)
	model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
	return model


def resnet_block(n_filters, input_layer):
	"""
	:param n_filters:
	:param input_layer:
	:return: g
	"""
	init = RandomNormal(stddev=0.02)
	g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(input_layer)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Concatenate()([g, input_layer])
	return g


def define_generator(shape, n_resnet=9):
	"""
	:param shape:
	:param n_resnet: default 9
	:return: model
	"""
	init = RandomNormal(stddev=0.02)
	in_image = Input(shape=shape)

	g = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(in_image)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)

	g = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)

	g = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)

	for _ in range(n_resnet):
		g = resnet_block(256, g)

	g = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)

	g = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)

	g = Conv2D(3, (7, 7), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)

	out_image = Activation('tanh')(g)
	model = Model(in_image, out_image)
	return model


def define_composite_model(generator_1, discriminator, generator_2, shape):
	"""
	:param generator_1:
	:param discriminator:
	:param generator_2:
	:param shape:
	:return: model
	"""
	generator_1.trainable = True
	discriminator.trainable = False
	generator_2.trainable = False
	input_gen = Input(shape=shape)
	gen1_out = generator_1(input_gen)
	output_d = discriminator(gen1_out)
	input_id = Input(shape=shape)
	output_id = generator_1(input_id)
	output_f = generator_2(gen1_out)
	gen2_out = generator_2(input_id)
	output_b = generator_1(gen2_out)
	model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
	return model


def load_real_samples(filename):
	"""
	:param filename:
	:return: [X1,X2]
	"""
	data = load(filename)
	X1, X2 = data['arr_0'], data['arr_1']
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]


def generate_real_samples(dataset, n_samples, patch_shape):
	"""
	:param dataset:
	:param n_samples:
	:param patch_shape:
	:return: return X, y
	"""
	ix = randint(0, dataset.shape[0], n_samples)
	X = dataset[ix]
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return X, y


def generate_fake_samples(g_model, dataset, patch_shape):
	"""
	:param g_model:
	:param dataset:
	:param patch_shape:
	:return: X, y
	"""
	X = g_model.predict(dataset)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y


def save_models(step, g_model_BtoA):
	"""
	:param step:
	:param g_model_BtoA:
	"""
	filename = 'models/g_model_BtoA_%06d.h5' % (step + 1)
	g_model_BtoA.save(filename)
	print('>Saved: %s' % filename)


def summarize_performance(step, g_model, trainX, name, n_samples=5):
	"""
	:param step:
	:param g_model:
	:param trainX:
	:param name:
	:param n_samples: default 5
	"""
	X_in, _ = generate_real_samples(trainX, n_samples, 0)
	X_out, _ = generate_fake_samples(g_model, X_in, 0)
	X_in = (X_in + 1) / 2.0
	X_out = (X_out + 1) / 2.0
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_in[i])
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_out[i])
	filename = '%s_generated_plot_%06d.png' % (name, (step + 1))
	pyplot.savefig('generated/' + filename)
	pyplot.close()


def update_image_pool(pool, images, max_size=50):
	"""
	:param pool:
	:param images:
	:param max_size: default 50
	:return: array of selected
	"""
	selected = list()
	for image in images:
		if len(pool) < max_size:
			pool.append(image)
			selected.append(image)
		elif random() < 0.5:
			selected.append(image)
		else:
			ix = randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = image
	return asarray(selected)


def train(discriminator_a, generator_ab, generator_ba, composite_ba, dataset):
	"""
	:param discriminator_a:
	:param generator_ab:
	:param generator_ba:
	:param composite_ba:
	:param dataset:
	"""
	n_epochs, n_batch, = 500, 1
	n_patch = discriminator_a.output_shape[1]
	trainA, trainB = dataset
	poolA = list()
	bat_per_epo = int(len(trainA) / n_batch)
	n_steps = bat_per_epo * n_epochs
	for i in range(n_steps):
		print("Epochs: {0}/{1}".format(i, n_steps))
		X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
		X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
		X_fakeA, y_fakeA = generate_fake_samples(generator_ba, X_realB, n_patch)
		X_fakeA = update_image_pool(poolA, X_fakeA)

		g_loss, _, _, _, _ = composite_ba.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
		d_a_loss1 = discriminator_a.train_on_batch(X_realA, y_realA)
		d_a_loss2 = discriminator_a.train_on_batch(X_fakeA, y_fakeA)

		print('>%d, dA[%.3f,%.3f] g[%.3f]' % (i + 1, d_a_loss1, d_a_loss2, g_loss))
		if (i + 1) % 100 == 0:
			summarize_performance(i, generator_ba, trainB, 'BtoA')
		if (i + 1) % 2500 == 0:
			save_models(i, generator_ba)


def main():
	if not path.isfile('city2art_256.npz'):
		loadimages.main()
	dataset = load_real_samples('city2art_256.npz')
	print('Loaded', dataset[0].shape, dataset[1].shape)
	image_shape = dataset[0].shape[1:]
	generator_ab = define_generator(image_shape)
	discriminator_a = define_discriminator(image_shape)
	generator_ba = define_generator(image_shape)
	"""
	# load model
	model = 'models/g_model_BtoA_009620.h5'
	if (path.isfile(model)):
		cust = {'InstanceNormalization': InstanceNormalization}
		generator_ba = tf.keras.models.load_model(model, cust)
		print("Model loaded")
	"""
	composite_ba = define_composite_model(generator_ba, discriminator_a, generator_ab, image_shape)
	train(discriminator_a, generator_ab, generator_ba, composite_ba, dataset)


if __name__ == "__main__":
	main()

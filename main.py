from os import listdir, path
from random import random
import tensorflow as tf
from tensorflow import keras
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


#define globals
kernel_init = RandomNormal(mean=0.0, stddev=0.02)
epochs = 500
batch_size = 1
res_blocks = 9 #9 residual blocks for image size 256x256 (or higher), if image size 128x128 -> use 6 residual blocks

def downsample(x, filters, activation, kernel_size=(3, 3), strides=(2, 2), padding='same'):
    """
    docstring
    """
    x = Conv2D(filters,kernel_size,strides=strides,kernel_initializer=kernel_init,padding=padding)(x)
    x = InstanceNormalization(axis=-1)(x)
    if activation:
        x = activation(x)
    
    return x

def residual_block(n_filters, input_layer):
	"""
	:param n_filters:
	:param input_layer:
	:return: x
	"""
	x = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=kernel_init)(input_layer)
	x = InstanceNormalization(axis=-1)(x)
	x = Activation('relu')(x)
	x = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=kernel_init)(x)
	x = InstanceNormalization(axis=-1)(x)
	x = Concatenate()([x, input_layer])
	return x

def upsample(x, filters,activation,kernel_size=(3, 3),strides=(2, 2),padding="same"):
    """
    docstring
    """
    x = Conv2DTranspose(filters,kernel_size,strides=strides,kernel_initializer=kernel_init,padding=padding)(x)
    x = InstanceNormalization(axis=-1)(x)
    if activation:
        x = activation(x)

    return x


def define_discriminator(image_shape):
	"""
    The discriminator architecture is: 
    C64(no InstanceNorm)-C128-C256-C512
    leaky ReLUs with slope of 0.2

	:param image_shape: image shape
	:return: model
	"""
	in_image = Input(shape=image_shape)
	filters = 64
    #C64(no InstanceNorm)
	d = Conv2D(filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=kernel_init)(in_image)
	d = LeakyReLU(alpha=0.2)(d)

	filters2 = filters
	for down_block in range(3):
		filters2 *= 2
		if down_block < 2:
			d = downsample(d,filters=filters2,activation=LeakyReLU(alpha=0.2),kernel_size=(4, 4),strides=(2, 2)) #C128-C256
		else:
			d = downsample(d,filters=filters2,activation=LeakyReLU(alpha=0.2),kernel_size=(4, 4),strides=(1, 1)) #C512

	patch_out = Conv2D(1, (4, 4), padding='same', kernel_initializer=kernel_init)(d)
	model = Model(in_image, patch_out)
	model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
	return model


def define_generator(image_shape):
	"""
    Generator architecture (9 residual blocks for image size 256x256):
	c7s1-64 - d128 - d256 - R256 - R256 - R256 - R256 - R256 - R256 - R256 - R256 - R256 - u128 - u64 - c7s1-3

    :param image_shape:
	:return: model
	"""
	in_image = Input(shape=image_shape)
	filters = 64
	down_blocks=2
	up_blocks=2

	#c7s1-64
	g = Conv2D(filters, (7, 7), padding='same', kernel_initializer=kernel_init)(in_image)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
    
    #Downsample blocks (d128 - d256)
	for _ in range(down_blocks):
		filters *= 2
		g = downsample(g, filters=filters,activation=Activation('relu'))

    #Residual blocks (R256 x9)
	for _ in range(res_blocks):
		g = residual_block(256, g)

    #Upsample blocks (u128 - u64)
	for _ in range(up_blocks):
		filters //= 2
		g = upsample(g, filters,activation=Activation('relu'))
    
    #c7s1-3
	g = Conv2D(3, (7, 7), padding='same', kernel_initializer=kernel_init)(g)
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
	print('Saved: %s' % filename)


def summarize_performance(step, g_model, trainX, n_samples=5):
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
	filename = 'generated/generated_art_%06d.png' % (step + 1)
	pyplot.savefig(filename)
	print('Saved: %s' % filename)
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
	n_patch = discriminator_a.output_shape[1]
	trainA, trainB = dataset
	poolA = list()
	bat_per_epo = int(len(trainA) / batch_size)
	n_steps = bat_per_epo * epochs
	for i in range(n_steps):
		print("Epochs: {0}/{1}".format(i, n_steps))
		X_realA, y_realA = generate_real_samples(trainA, batch_size, n_patch)
		X_realB, y_realB = generate_real_samples(trainB, batch_size, n_patch)
		X_fakeA, y_fakeA = generate_fake_samples(generator_ba, X_realB, n_patch)
		X_fakeA = update_image_pool(poolA, X_fakeA)

		g_loss, _, _, _, _ = composite_ba.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
		d_a_loss1 = discriminator_a.train_on_batch(X_realA, y_realA)
		d_a_loss2 = discriminator_a.train_on_batch(X_fakeA, y_fakeA)

		print('>%d, dA[%.3f,%.3f] g[%.3f]' % (i + 1, d_a_loss1, d_a_loss2, g_loss))
		if (i + 1) % 100 == 0:
			summarize_performance(i, generator_ba, trainB)
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
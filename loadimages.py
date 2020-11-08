from os import listdir
from keras.preprocessing.image import img_to_array, load_img
from numpy import asarray, savez_compressed, vstack

def load_images(path, size=(256,256)):
	"""
    :param path: path to images
    :param size: size on images, default = 256,256
    :return: array of data list
    """
	data_list = list()
	for filename in listdir(path):
		data_list.append(img_to_array(load_img(path + filename, target_size=size)))
	return asarray(data_list)

def main():

	path = 'city2art/'
	stackedA = vstack((load_images(path + 'trainA/'), load_images(path + 'testA/')))
	print('StackedA: ', stackedA.shape)
	stackedB = vstack((load_images(path + 'trainB/'), load_images(path + 'testB/')))
	print('StackedB: ', stackedB.shape)

	filename = 'city2art_256.npz'
	savez_compressed(filename, stackedA, stackedB)
	print('Saved dataset: ', filename)

if __name__ == "__main__":
    main()

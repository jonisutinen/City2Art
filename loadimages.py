from os import listdir
from numpy import asarray, vstack
from keras.preprocessing.image import img_to_array, load_img
from numpy import savez_compressed

# load all images in a directory into memory
def load_images(path, size=(256,256)):
	"""
    :param path: path to images
    :param size: size on images, default = 256,256
    :return: array of data list
    """
	data_list = list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# store
		data_list.append(pixels)
	return asarray(data_list)

def main():
	# dataset path
	path = 'city2art/'
	# load dataset A
	dataA1 = load_images(path + 'trainA/')
	dataAB = load_images(path + 'testA/')
	dataA = vstack((dataA1, dataAB))
	print('Loaded dataA: ', dataA.shape)
	# load dataset B
	dataB1 = load_images(path + 'trainB/')
	dataB2 = load_images(path + 'testB/')
	dataB = vstack((dataB1, dataB2))
	print('Loaded dataB: ', dataB.shape)
	# save as compressed numpy array
	filename = 'city2art_256.npz'
	savez_compressed(filename, dataA, dataB)
	print('Saved dataset: ', filename)

if __name__ == "__main__":
    main()
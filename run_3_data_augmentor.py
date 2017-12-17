import os
import sys
import cv2
import numpy as np


label_dict = {}
CLASSES = 15
IMG_SIZE = 200
SPI = 4          # samples per image


def crop_4(img, crop_size):
	return np.asarray([
		img[:crop_size, :crop_size],
		img[:crop_size, -crop_size:],
		img[-crop_size:, :crop_size],
		img[-crop_size:, -crop_size:],
		], dtype=np.uint8)

def set_label_dictionary():
	counter = 0
	global label_dict
	for classname in os.listdir('training'):
		label_dict[classname] = counter
		counter += 1

def randomize_data(features_data, labels):
	permutation = np.random.permutation(len(labels))
	return (features_data[permutation], labels[permutation])


def serialize_dataset(dataset_type, first_sample, last_sample):

	global label_dict
	print('Serializing ' + dataset_type + ' set', flush=True)

	counter = 0
	dataset_size = last_sample - first_sample
	data = np.empty((CLASSES * dataset_size * SPI, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
	labels = np.empty((CLASSES * dataset_size * SPI, 1), dtype=np.uint8)

	for classname in os.listdir('training'):
		for image_index in range(first_sample, last_sample):

			# read the image
			imgname = '%d.jpg' % image_index
			img = cv2.imread(os.path.join('training', classname, imgname), 0)

			# store cropped instances
			data[counter:counter + SPI] = crop_4(img, IMG_SIZE)
			labels[counter:counter + SPI] = np.asarray(SPI * [label_dict[classname]],
				dtype = np.uint8).reshape((SPI, 1))
			counter += SPI

	data, labels = randomize_data(data, labels)

	labels.dump(os.path.join('run_3_data', 'raw_data', dataset_type + '_labels.dat'))
	for i, img in enumerate(data):
		imgname = '%.6d.jpg' % i
		cv2.imwrite(os.path.join('run_3_data', 'raw_data', dataset_type, imgname), img)



if __name__ == '__main__':
    
	if len(sys.argv) != 4:
		sys.exit('Insert the number of training/validation/test images')

	try:
		training_size = int(sys.argv[1])
		validation_size = int(sys.argv[2])
		test_size = int(sys.argv[3])
		if training_size + validation_size + test_size != 100:
			sys.exit('Invalid division of data')
	except ValueError:
		sys.exit('Wrong type of arguments')

	set_label_dictionary()

	serialize_dataset('training', 0, training_size)
	serialize_dataset('validation', training_size, training_size + validation_size)
	serialize_dataset('test', training_size + validation_size, 100)
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


label_dict = {}
CLASSES = 15
IMG_SIZE = 200
SPI = 20          # samples per image


def crop_5(img, crop_size):
	return np.asarray([
		img[:crop_size, :crop_size],
		img[:crop_size, -crop_size:],
		img[-crop_size:, :crop_size],
		img[-crop_size:, -crop_size:],
		img[img.shape[0]//2-crop_size//2:img.shape[0]//2+crop_size//2,
		    img.shape[1]//2-crop_size//2:img.shape[1]//2+crop_size//2]
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


def serialize_dataset_augment(dataset_type, first_sample, last_sample):

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

			# 5 cropped instances
			raw_cropped = crop_5(img, IMG_SIZE)
			aug_cropped = crop_5(img, IMG_SIZE)

			# mirror the images
			raw_flipped = flip_images(raw_cropped)
			aug_flipped = flip_images(aug_cropped)

			# add a little noise
			for i in range(len(aug_cropped)):
				aug_cropped[i] = maniuplate_image(aug_cropped[i])
				aug_flipped[i] = maniuplate_image(aug_flipped[i])

			
			data[counter : counter + SPI//4] = raw_cropped
			data[counter + SPI//4 : counter + SPI//2] = raw_flipped
			data[counter + SPI//2 : counter + 3*SPI//4] = aug_cropped
			data[counter + 3*SPI//4 : counter + SPI] = aug_flipped

			labels[counter:counter + SPI] = np.asarray(SPI * [label_dict[classname]],
				dtype = np.uint8).reshape((SPI, 1))

			counter += SPI


	data, labels = randomize_data(data, labels)

	labels.dump(os.path.join('run_3_data', 'raw_data', dataset_type + '_labels.dat'))
	for i, img in enumerate(data):
		imgname = '%.6d.jpg' % i
		cv2.imwrite(os.path.join('run_3_data', 'raw_data', dataset_type, imgname), img)


def serialize_dataset(dataset_type, first_sample, last_sample):

	global label_dict
	print('Serializing ' + dataset_type + ' set', flush=True)

	counter = 0
	dataset_size = last_sample - first_sample
	data = np.empty((CLASSES * dataset_size * 10, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
	labels = np.empty((CLASSES * dataset_size * 10, 1), dtype=np.uint8)

	for classname in os.listdir('training'):
		for image_index in range(first_sample, last_sample):

			# read the image
			imgname = '%d.jpg' % image_index
			img = cv2.imread(os.path.join('training', classname, imgname), 0)

			# 5 cropped instances
			raw_cropped = crop_5(img, IMG_SIZE)

			# mirror the images
			raw_flipped = flip_images(raw_cropped)

			
			data[counter : counter + 5] = raw_cropped
			data[counter + 5 : counter + 10] = raw_flipped

			labels[counter : counter + 10] = np.asarray(10 * [label_dict[classname]],
				dtype = np.uint8).reshape((10, 1))

			counter += 10


	data, labels = randomize_data(data, labels)

	labels.dump(os.path.join('run_3_data', 'raw_data', dataset_type + '_labels.dat'))
	for i, img in enumerate(data):
		imgname = '%.6d.jpg' % i
		cv2.imwrite(os.path.join('run_3_data', 'raw_data', dataset_type, imgname), img)


def maniuplate_image(img):

	if np.random.randint(0, 2) < 1:
		return salt_and_pepper(img, 0.5, 0.007)	
	return cv2.blur(img, (3, 3))


def flip_images(images):
	return np.flip(np.copy(images), axis=2)
    

def salt_and_pepper(image, salt_vs_pepper, amount):

    rows, cols = image.shape

    num_salt = np.ceil(amount * image.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper))
    
 
    # add Salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    image[coords[0], coords[1]] = 255

    # Add Pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    image[coords[0], coords[1]] = 0
    
    return image



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

	serialize_dataset_augment('training', 0, training_size)
	serialize_dataset('validation', training_size, training_size + validation_size)
	serialize_dataset('test', training_size + validation_size, 100)
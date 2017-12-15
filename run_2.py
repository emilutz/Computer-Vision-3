import os
import cv2
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn import svm
from scipy import spatial


label_dict = {}
CLASSES = 15
SAMPLES_PER_CLASS = 100


#====================[ K-MEANS PROCESSING ]====================#


def set_label_dictionary():
	counter = 0
	global label_dict
	for classname in os.listdir('training'):
		label_dict[classname] = counter
		counter += 1


def get_training_patch_samples(training_size, samples_per_image, patch_size):

	X = np.empty((CLASSES * training_size * samples_per_image, patch_size, patch_size))

	for classname in os.listdir('training'):
		for image_index in range(training_size):

			# read the image
			imgname = '%d.jpg' % image_index
			img = cv2.imread(os.path.join('training', classname, imgname), 0)

			# sampling random patches from the image
			row_coords = np.random.randint(0, img.shape[0] - patch_size, samples_per_image)
			col_coords = np.random.randint(0, img.shape[1] - patch_size, samples_per_image)

			# putting the generated coordinates together
			coords = np.dstack((row_coords, col_coords))[0]

			# store the patches in the patches matrix
			for pair_index, (x, y) in enumerate(coords):
				X[label_dict[classname] * training_size * samples_per_image 
				+ image_index * samples_per_image + pair_index] = img[x:x+patch_size, y:y+patch_size]

	return X


def vectorize_patches(patch_data):
	sh = patch_data.shape
	return patch_data.reshape(sh[0], sh[1] * sh[2])


def normalize_patches(patch_data):

	mean = np.mean(patch_data, axis=1).reshape((len(patch_data), 1))
	std = np.std(patch_data, axis=1).reshape((len(patch_data), 1))

	# avoid division by a std of 0
	with np.errstate(divide='ignore', invalid='ignore'):
		patch_data = np.true_divide(patch_data - mean, std)
		patch_data[~np.isfinite(patch_data)] = 0

	return patch_data 


#====================[ CLASSIFICATION PROCESSING ]====================#


def one_dimensional_fittings(length, patch, stride):
	return (length - patch) // stride + 1


def feature_vector(img, patch_size, stride, clusterer, clusters):

	# see how many patches can be sampled
	height_fittings = one_dimensional_fittings(img.shape[0], patch_size, stride)
	width_fittings  = one_dimensional_fittings(img.shape[1], patch_size, stride)
	patches = np.empty((height_fittings * width_fittings, patch_size, patch_size))

	# collect the patches
	for r in range(0, img.shape[0] - patch_size + 1, stride):
		for c in range(0, img.shape[1] - patch_size + 1, stride):
			patches[r//stride * width_fittings + c//stride] = img[r:r+patch_size, c:c+patch_size]

    # vectorize the patches
	patches = vectorize_patches(patches)

	# normalize the patches
	patches = normalize_patches(patches)

	# cluster-classify the patches
	quantization = clusterer.predict(patches)

	# create feature vector
	feature_vector = np.zeros((1, clusters))
	for q in quantization:
		feature_vector[0, q] += 1

	return feature_vector


def read_and_convert_data(first_sample, last_sample, patch_size, stride, clusterer, clusters):

	counter = 0
	X = np.zeros((CLASSES * (last_sample - first_sample), clusters))
	y = np.zeros((CLASSES * (last_sample - first_sample),), dtype=np.int32)

	for classname in os.listdir('training'):
		for i in range(first_sample, last_sample):

			imgname = '%d.jpg' % i
			img = cv2.imread(os.path.join('training', classname, imgname), 0)
			
			X[counter] = feature_vector(img, patch_size, stride, clusterer, clusters)
			y[counter] = label_dict[classname]
			counter += 1
	
	return (X, y)			


def randomize_data(features_data, labels):
	permutation = np.random.permutation(len(labels))
	return (features_data[permutation], labels[permutation])


#====================[ TEST SET ]====================#


def predict_on_test(patch_size, stride, clusterer, clusters, clf):

	TEST_SIZE = 2985
	counter = 0
	X_test = np.empty((TEST_SIZE, clusters))

	# clustering for features
	for image_index in range(2988):
		try:
			imgname = '%d.jpg' % image_index
			img = cv2.imread(os.path.join('testing', imgname), 0)
		 
			X_test[counter] = feature_vector(img, patch_size, stride, clusterer, clusters)
			counter += 1
		except AttributeError:
			pass 

	# classify the images
	predictions = clf.predict(X_test)

	reverse_label_dict = {}
	for key, value in label_dict.items():
		reverse_label_dict[value] = key

	# serialize the results
	with open('run_2.txt', 'w') as f:
		for pred in predictions:
			f.write(str(reverse_label_dict[pred]) + '\n')


#====================[ MAIN FLOW ]====================#


if __name__ == '__main__':
    
	if len(sys.argv) != 6:
		sys.exit('Wrong number of arguments')

	try:
		samples_per_image = int(sys.argv[1])
		patch_size = int(sys.argv[2])
		stride = int(sys.argv[3])
		clusters = int(sys.argv[4])
		training_percentage = float(sys.argv[5])
		if training_percentage < 0 or training_percentage > 1:
			sys.exit('Invalid percentage of training data')
		else:
			training_images_number = int(training_percentage * SAMPLES_PER_CLASS)
	except ValueError:
		sys.exit('Wrong type of arguments')

	print('Setting parameters...')
	print('    patch size = {0}x{0}'.format(patch_size))
	print('    stride     = {0}'.format(stride))
	print('    clusters   = {0}'.format(clusters), flush=True)

	set_label_dictionary()

	print('Reading training data...')
	print('    using {0:.1f}% of total data'.format(100 * training_percentage))
	print('    sampling {0} patches from each image'.format(samples_per_image), flush=True)
	X_clustering = get_training_patch_samples(training_images_number, samples_per_image, patch_size)

	print('Vectorizing patches...', flush=True)
	X_clustering = vectorize_patches(X_clustering)

	print('Normalizing patches...', flush=True)
	X_clustering = normalize_patches(X_clustering)

	print('Running K-means...')
	print('    data samples : {0}'.format(len(X_clustering)))
	print('    centroids    : {0}'.format(clusters), flush=True)
	clusterer = KMeans(n_clusters=clusters).fit(X_clustering)

	print('Creating training set for SVM...', flush=True)
	X_svm, y_svm = read_and_convert_data(0, training_images_number, patch_size, stride, clusterer, clusters)

	# for x in X_svm:
	# 	print(x)
	# 	plt.bar(list(range(clusters)), x, width=0.5)
	# 	plt.show()

	print('Randomizing data...', flush=True)
	X_svm, y_svm = randomize_data(X_svm, y_svm)

	print('Running SVM...', flush=True)
	clf = svm.LinearSVC()
	clf.fit(X_svm, y_svm)

	print('Loading test data...', flush=True)
	X_test, y_test = read_and_convert_data(training_images_number, SAMPLES_PER_CLASS, patch_size, stride, clusterer, clusters)

	training_accuracy = accuracy_score(clf.predict(X_svm), y_svm) * 100
	test_accuracy = accuracy_score(clf.predict(X_test), y_test) * 100

	print('\nTraining accuracy : {0:.2f}%'.format(training_accuracy), flush=True)
	print('Test accuracy     : {0:.2f}%\n'.format(test_accuracy), flush=True)

	predict_on_test(patch_size, stride, clusterer, clusters, clf)
	sys.exit()

	print('Done !', flush=True)




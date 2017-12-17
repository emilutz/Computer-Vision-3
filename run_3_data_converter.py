import os
import cv2
import numpy as np
import tensorflow as tf


IMG_SIZE = 200


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tf(data, labels, dataset_type):

	writing_path = os.path.join('run_3_data', 'tf_data', dataset_type + '.tfrecords')
	writer = tf.python_io.TFRecordWriter(writing_path)

	for index in range(len(labels)):
		data_sample = data[index].tostring()
		example = tf.train.Example(features=tf.train.Features(feature={
		    'label': _int64_feature(int(labels[index])),
		    'image': _bytes_feature(data_sample)}))
		writer.write(example.SerializeToString())
		
	writer.close()


if __name__ == '__main__':
    
	# read raw data
	for dataset_type in ['training', 'validation', 'test']:
		print('Serializing ' + dataset_type)

		labels = np.load(os.path.join('run_3_data', 'raw_data',
					dataset_type + '_labels.dat'))
		image_names = os.listdir(os.path.join('run_3_data', 'raw_data',
					dataset_type))
		images = np.empty((len(labels), IMG_SIZE, IMG_SIZE), dtype=np.uint8)
		
		counter = 0
		for imgname in image_names:
			images[counter] = cv2.imread(os.path.join(
				'run_3_data', 'raw_data', dataset_type, imgname), 0)
			counter += 1

		convert_to_tf(images, labels, dataset_type)


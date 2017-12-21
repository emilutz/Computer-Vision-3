import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from run_3_data_reader import *


CLASSES = 15
IMG_SIZE = 200
class_to_label = {}
label_to_class = {}

def set_label_dictionaries():
	counter = 0
	global label_dict
	for classname in os.listdir('training'):
		class_to_label[classname] = counter
		label_to_class[counter] = classname
		counter += 1

def load_weights(weight_file):
	weights = np.load(weight_file)
	# keys = sorted(weights.keys())
	# for i, k in enumerate(keys):
	# 	print(i, k, np.shape(weights[k]))
	return weights

def load_validation():
	images = os.listdir(os.path.join('run_3_data', 'raw_data', 'validation'))
	val_images = np.empty((len(images), IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)

	counter = 0
	for imgname in images:
		img = cv2.imread(os.path.join(
			'run_3_data', 'raw_data', 'validation', imgname), 0)
		val_images[counter] = img.reshape((IMG_SIZE, IMG_SIZE, 1))
		counter += 1

	val_labels = np.load(os.path.join(
		'run_3_data', 'raw_data', 'validation_labels.dat'))

	return (val_images, val_labels)


#=====[ HYPER-PARAMETERS ]=====#

BATCH_SIZE = 8
EPOCHS     = 2

VALIDATION_BATCH = 16

#==============================#


set_label_dictionaries() 
tf.logging.set_verbosity(tf.logging.INFO)

vgg_params = load_weights(os.path.join('run_3_data', 'vgg_data', 'vgg16_weights.npz'))
val_images, val_labels = load_validation()


with tf.device('/cpu:0'):

	# decide the dataset input type
	is_training = tf.placeholder(tf.bool, name="is_training")
	# testing data
	images_tst = tf.placeholder(tf.uint8, shape=[VALIDATION_BATCH, IMG_SIZE, IMG_SIZE, 1])
	labels_tst = tf.placeholder(tf.int32, shape=[VALIDATION_BATCH, 1])

	# read training data from queues
	images_trn, labels_trn = inputs('training', BATCH_SIZE, EPOCHS)

	# choose the input
	images = tf.cond(is_training, lambda: images_trn, lambda: images_tst)
	labels = tf.cond(is_training, lambda: labels_trn, lambda: labels_tst)

	# subtract the mean
	images = (tf.cast(images, tf.float32) - 127.5)
	# channelize images
	images = tf.image.grayscale_to_rgb(images)
	# encode labels using one hot
	labels = tf.one_hot(labels, CLASSES)


with tf.device('/gpu:0'):
	

	#==================================[ CONV 1 ]==================================#


	with tf.variable_scope('conv1_1') as scope:

		conv_weights = vgg_params['conv1_1_W']
		kernel = tf.get_variable('weights', initializer=conv_weights, dtype=tf.float32)

		conv_biases = vgg_params['conv1_1_b']
		biases = tf.get_variable('biases', initializer=conv_biases, dtype=tf.float32)

		conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
		out = tf.nn.bias_add(conv, biases)
		activated1_1 = tf.nn.relu(out, name='out')


	with tf.variable_scope('conv1_2') as scope:

		conv_weights = vgg_params['conv1_2_W']
		kernel = tf.get_variable('weights', initializer=conv_weights, dtype=tf.float32)

		conv_biases = vgg_params['conv1_2_b']
		biases = tf.get_variable('biases', initializer=conv_biases, dtype=tf.float32)

		conv = tf.nn.conv2d(activated1_1, kernel, [1, 1, 1, 1], padding='SAME')
		out = tf.nn.bias_add(conv, biases)
		activated1_2 = tf.nn.relu(out, name='out')


	pool1 = tf.nn.max_pool(activated1_2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool1')


	#==================================[ CONV 2 ]==================================#


	with tf.variable_scope('conv2_1') as scope:

		conv_weights = vgg_params['conv2_1_W']
		kernel = tf.get_variable('weights', initializer=conv_weights, dtype=tf.float32)

		conv_biases = vgg_params['conv2_1_b']
		biases = tf.get_variable('biases', initializer=conv_biases, dtype=tf.float32)

		conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
		out = tf.nn.bias_add(conv, biases)
		activated2_1 = tf.nn.relu(out, name='out')


	with tf.variable_scope('conv2_2') as scope:

		conv_weights = vgg_params['conv2_2_W']
		kernel = tf.get_variable('weights', initializer=conv_weights, dtype=tf.float32)

		conv_biases = vgg_params['conv2_2_b']
		biases = tf.get_variable('biases', initializer=conv_biases, dtype=tf.float32)

		conv = tf.nn.conv2d(activated2_1, kernel, [1, 1, 1, 1], padding='SAME')
		out = tf.nn.bias_add(conv, biases)
		activated2_2 = tf.nn.relu(out, name='out')


	pool2 = tf.nn.max_pool(activated2_2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool2')


	#==================================[ CONV 3 ]==================================#


	with tf.variable_scope('conv3_1') as scope:

		conv_weights = vgg_params['conv3_1_W']
		kernel = tf.get_variable('weights', initializer=conv_weights, dtype=tf.float32)

		conv_biases = vgg_params['conv3_1_b']
		biases = tf.get_variable('biases', initializer=conv_biases, dtype=tf.float32)

		conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
		out = tf.nn.bias_add(conv, biases)
		activated3_1 = tf.nn.relu(out, name='out')


	with tf.variable_scope('conv3_2') as scope:

		conv_weights = vgg_params['conv3_2_W']
		kernel = tf.get_variable('weights', initializer=conv_weights, dtype=tf.float32)

		conv_biases = vgg_params['conv3_2_b']
		biases = tf.get_variable('biases', initializer=conv_biases, dtype=tf.float32)

		conv = tf.nn.conv2d(activated3_1, kernel, [1, 1, 1, 1], padding='SAME')
		out = tf.nn.bias_add(conv, biases)
		activated3_2 = tf.nn.relu(out, name='out')


	with tf.variable_scope('conv3_3') as scope:

		conv_weights = vgg_params['conv3_3_W']
		kernel = tf.get_variable('weights', initializer=conv_weights, dtype=tf.float32)

		conv_biases = vgg_params['conv3_3_b']
		biases = tf.get_variable('biases', initializer=conv_biases, dtype=tf.float32)

		conv = tf.nn.conv2d(activated3_2, kernel, [1, 1, 1, 1], padding='SAME')
		out = tf.nn.bias_add(conv, biases)
		to_test = out
		activated3_3 = tf.nn.relu(out, name='out')


	pool3 = tf.nn.max_pool(activated3_3,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool3')


	#==================================[ DENSE 1 ]==================================#


	img_shape = pool3.get_shape()
	dense1_shape = int(np.prod(pool3.get_shape()[1:]))

	pool3_flat = tf.reshape(pool3, [-1, dense1_shape])

	with tf.variable_scope('fc1') as scope:

		weights = tf.get_variable(name='weights', trainable=True,
			                      initializer=tf.truncated_normal([dense1_shape, 64],
		                                                          dtype=tf.float32,
		                                                          mean=0,
		                                                          stddev=1e-1))
		biases = tf.get_variable(name='biases', trainable=True,
			                     initializer=tf.constant(1.0,
		                                                 shape=[64],
		                                                 dtype=tf.float32))

		dense1 = tf.nn.bias_add(tf.matmul(pool3_flat, weights), biases)
		activated1 = tf.nn.relu(dense1, name='out')


	#==================================[ DENSE 2 ]==================================#


	with tf.variable_scope('fc2') as scope:

		weights = tf.get_variable(name='weights', trainable=True,
			                      initializer=tf.truncated_normal([64, CLASSES],
		                                                          dtype=tf.float32,
		                                                          mean=0,
		                                                          stddev=1e-1))
		biases = tf.get_variable(name='biases', trainable=True,
			                     initializer=tf.constant(1.0,
		                                                 shape=[CLASSES],
		                                                 dtype=tf.float32))

		logits = tf.nn.bias_add(tf.matmul(activated1, weights), biases)


	loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits))

	optimizer = tf.train.AdadeltaOptimizer(learning_rate=1e-3)

	train_op = optimizer.minimize(loss)


# Create the graph, etc.
init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

# Create a session for running operations in the Graph.
sess = tf.Session()

# Initialize the variables (like the epoch counter).
sess.run(init_op)

# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# Collect data for Tensorboard
with tf.device('/cpu:0'):

	tf.summary.image('act_1', activated1_1[:,:,:,:3])
	tf.summary.image('act_2', activated1_2[:,:,:,:3])
	tf.summary.scalar('loss', loss)

	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter('run_3_tensorboard', sess.graph)


step = 0
try:
	while not coord.should_stop():
		# Run training steps or whatever
		step += 1
		print(step, flush=True)

		feed_dict = {
			is_training : True,
			images_tst : val_images[:VALIDATION_BATCH],
			labels_tst : val_labels[:VALIDATION_BATCH]
		}
		_, summary, training_loss, tt_img = \
		    sess.run([train_op, merged, loss, to_test], feed_dict=feed_dict)

		train_writer.add_summary(summary, step)
		print('Training Loss   : {0:.2f}'.format(training_loss))

		min_t = np.min(tt_img[0])
		max_t = np.max(tt_img[0])
		mean_t = np.mean(tt_img[0])

		print('mean : ', mean_t)
		print('std : ', np.std(tt_img[0]))
		print('min : ', min_t)
		print('max : ', max_t)

		plt.imshow((tt_img[0,:,:,:3] - min_t) / (max_t - min_t), cmap='gray')
		plt.show()

		if step % 20 == 0:

			validation_loss = 0
			validation_wrong = 0
			validation_total = 0
			for s in range(0, len(val_images), VALIDATION_BATCH):
				try:
					# print(s / len(val_images))
					feed_dict = {
						is_training : False,
						images_tst : val_images[s : s + VALIDATION_BATCH],
						labels_tst : val_labels[s : s + VALIDATION_BATCH]
					}
					val_loss, true, pred = sess.run([loss, labels, logits], feed_dict=feed_dict)
				
					validation_loss += val_loss
					validation_total += len(true)
					validation_wrong += np.count_nonzero(
						np.argmax(true[:,0,:], axis=1) - np.argmax(pred, axis=1))

				except ValueError:
					pass

			validation_loss /= len(val_images) // VALIDATION_BATCH
			print('Validation Loss : {0:.2f}'.format(validation_loss))
			print('Validation Accuracy : {0:.2f}%'.format(100 - 
				(validation_wrong / validation_total) * 100))


except tf.errors.OutOfRangeError:
	print('\nDone training -- epoch limit reached\n')
finally:
	# When done, ask the threads to stop.
	coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()
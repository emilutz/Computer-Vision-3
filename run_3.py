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


def load_test():
	images = os.listdir(os.path.join('run_3_data', 'raw_data', 'test'))
	val_images = np.empty((len(images), IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)

	counter = 0
	for imgname in images:
		img = cv2.imread(os.path.join(
			'run_3_data', 'raw_data', 'test', imgname), 0)
		val_images[counter] = img.reshape((IMG_SIZE, IMG_SIZE, 1))
		counter += 1

	val_labels = np.load(os.path.join(
		'run_3_data', 'raw_data', 'test_labels.dat'))

	return (val_images, val_labels)


#=====[ HYPER-PARAMETERS ]=====#

BATCH_SIZE = 32
EPOCHS     = 1000

VALIDATION_BATCH = 32

#==============================#


set_label_dictionaries() 
tf.logging.set_verbosity(tf.logging.INFO)

vgg_params = load_weights(os.path.join('run_3_data', 'vgg_data', 'vgg16_weights.npz'))
val_images, val_labels = load_validation()
# sys.exit()

with tf.device('/cpu:0'):

	# decide the dataset input type
	is_training = tf.placeholder(tf.bool, name="is_training")
	# testing data
	images_tst = tf.placeholder(tf.uint8, shape=[VALIDATION_BATCH, IMG_SIZE, IMG_SIZE, 1])
	labels_tst = tf.placeholder(tf.int32, shape=[VALIDATION_BATCH, 1])
	# dropout probability
	prob = tf.placeholder_with_default(1.0, shape=())

	# rotation angle
	angle = tf.placeholder(tf.float32, shape=[1], name='rotation_angle')

	# read training data from queues
	images_trn, labels_trn = inputs('training', BATCH_SIZE, EPOCHS)

	# choose the input
	images = tf.cond(is_training, lambda: images_trn, lambda: images_tst)
	labels = tf.cond(is_training, lambda: labels_trn, lambda: labels_tst)

	# subtract the mean
	images = (tf.cast(images, tf.float32) - 127.5)

	# rotate the images
	images = tf.contrib.image.rotate(images, angle)

	# channelize images
	images = tf.image.grayscale_to_rgb(images)

	# encode labels using one hot
	labels = tf.one_hot(labels, CLASSES)


with tf.device('/gpu:0'):
	

	#==================================[ CONV 1 ]==================================#

	to_test = images

	with tf.variable_scope('conv1_1') as scope:

		conv_weights = vgg_params['conv1_1_W']
		kernel = tf.get_variable('weights', initializer=conv_weights, dtype=tf.float32, trainable=False)

		conv_biases = vgg_params['conv1_1_b']
		biases = tf.get_variable('biases', initializer=conv_biases, dtype=tf.float32, trainable=False)

		conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
		out = tf.nn.bias_add(conv, biases)
		activated1_1 = tf.nn.relu(out, name='out')


	with tf.variable_scope('conv1_2') as scope:

		conv_weights = vgg_params['conv1_2_W']
		kernel = tf.get_variable('weights', initializer=conv_weights, dtype=tf.float32, trainable=False)

		conv_biases = vgg_params['conv1_2_b']
		biases = tf.get_variable('biases', initializer=conv_biases, dtype=tf.float32, trainable=False)

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
		kernel = tf.get_variable('weights', initializer=conv_weights, dtype=tf.float32, trainable=False)

		conv_biases = vgg_params['conv2_1_b']
		biases = tf.get_variable('biases', initializer=conv_biases, dtype=tf.float32, trainable=False)

		conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
		out = tf.nn.bias_add(conv, biases)
		activated2_1 = tf.nn.relu(out, name='out')


	with tf.variable_scope('conv2_2') as scope:

		conv_weights = vgg_params['conv2_2_W']
		kernel = tf.get_variable('weights', initializer=conv_weights, dtype=tf.float32, trainable=False)

		conv_biases = vgg_params['conv2_2_b']
		biases = tf.get_variable('biases', initializer=conv_biases, dtype=tf.float32, trainable=False)

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
		kernel = tf.get_variable('weights', initializer=conv_weights, dtype=tf.float32, trainable=False)

		conv_biases = vgg_params['conv3_1_b']
		biases = tf.get_variable('biases', initializer=conv_biases, dtype=tf.float32, trainable=False)

		conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
		out = tf.nn.bias_add(conv, biases)
		activated3_1 = tf.nn.relu(out, name='out')


	with tf.variable_scope('conv3_2') as scope:

		conv_weights = vgg_params['conv3_2_W']
		kernel = tf.get_variable('weights', initializer=conv_weights, dtype=tf.float32, trainable=False)

		conv_biases = vgg_params['conv3_2_b']
		biases = tf.get_variable('biases', initializer=conv_biases, dtype=tf.float32, trainable=False)

		conv = tf.nn.conv2d(activated3_1, kernel, [1, 1, 1, 1], padding='SAME')
		out = tf.nn.bias_add(conv, biases)
		activated3_2 = tf.nn.relu(out, name='out')


	with tf.variable_scope('conv3_3') as scope:

		conv_weights = vgg_params['conv3_3_W']
		kernel = tf.get_variable('weights', initializer=conv_weights, dtype=tf.float32, trainable=False)

		conv_biases = vgg_params['conv3_3_b']
		biases = tf.get_variable('biases', initializer=conv_biases, dtype=tf.float32, trainable=False)

		conv = tf.nn.conv2d(activated3_2, kernel, [1, 1, 1, 1], padding='SAME')
		out = tf.nn.bias_add(conv, biases)
		activated3_3 = tf.nn.relu(out, name='out')


	pool3 = tf.nn.max_pool(activated3_3,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool3')


	#==================================[ CONV 4 ]==================================#


	with tf.variable_scope('conv4_1') as scope:

		conv_weights = vgg_params['conv4_1_W']
		kernel = tf.get_variable('weights', initializer=conv_weights, dtype=tf.float32, trainable=False)

		conv_biases = vgg_params['conv4_1_b']
		biases = tf.get_variable('biases', initializer=conv_biases, dtype=tf.float32, trainable=False)

		conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
		out = tf.nn.bias_add(conv, biases)
		activated4_1 = tf.nn.relu(out, name='out')


	with tf.variable_scope('conv4_2') as scope:

		conv_weights = vgg_params['conv4_2_W']
		kernel = tf.get_variable('weights', initializer=conv_weights, dtype=tf.float32, trainable=False)

		conv_biases = vgg_params['conv4_2_b']
		biases = tf.get_variable('biases', initializer=conv_biases, dtype=tf.float32, trainable=False)

		conv = tf.nn.conv2d(activated4_1, kernel, [1, 1, 1, 1], padding='SAME')
		out = tf.nn.bias_add(conv, biases)
		activated4_2 = tf.nn.relu(out, name='out')


	with tf.variable_scope('conv4_3') as scope:

		conv_weights = vgg_params['conv4_3_W']
		kernel = tf.get_variable('weights', initializer=conv_weights, dtype=tf.float32, trainable=False)

		conv_biases = vgg_params['conv4_3_b']
		biases = tf.get_variable('biases', initializer=conv_biases, dtype=tf.float32, trainable=False)

		conv = tf.nn.conv2d(activated4_2, kernel, [1, 1, 1, 1], padding='SAME')
		out = tf.nn.bias_add(conv, biases)
		activated4_3 = tf.nn.relu(out, name='out')


	pool4 = tf.nn.max_pool(activated4_3,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool4')


	#==================================[ CONV 5 ]==================================#


	with tf.variable_scope('conv5_1') as scope:

		conv_weights = vgg_params['conv5_1_W']
		kernel = tf.get_variable('weights', initializer=conv_weights, dtype=tf.float32, trainable=False)

		conv_biases = vgg_params['conv5_1_b']
		biases = tf.get_variable('biases', initializer=conv_biases, dtype=tf.float32, trainable=False)

		conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
		out = tf.nn.bias_add(conv, biases)
		activated5_1 = tf.nn.relu(out, name='out')


	with tf.variable_scope('conv5_2') as scope:

		conv_weights = vgg_params['conv5_2_W']
		kernel = tf.get_variable('weights', initializer=conv_weights, dtype=tf.float32, trainable=False)

		conv_biases = vgg_params['conv5_2_b']
		biases = tf.get_variable('biases', initializer=conv_biases, dtype=tf.float32, trainable=False)

		conv = tf.nn.conv2d(activated5_1, kernel, [1, 1, 1, 1], padding='SAME')
		out = tf.nn.bias_add(conv, biases)
		activated5_2 = tf.nn.relu(out, name='out')


	with tf.variable_scope('conv5_3') as scope:

		conv_weights = vgg_params['conv5_3_W']
		kernel = tf.get_variable('weights', initializer=conv_weights, dtype=tf.float32, trainable=False)

		conv_biases = vgg_params['conv5_3_b']
		biases = tf.get_variable('biases', initializer=conv_biases, dtype=tf.float32, trainable=False)

		conv = tf.nn.conv2d(activated5_2, kernel, [1, 1, 1, 1], padding='SAME')
		out = tf.nn.bias_add(conv, biases)
		activated5_3 = tf.nn.relu(out, name='out')


	pool5 = tf.nn.max_pool(activated5_3,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool5')


with tf.device('/gpu:0'):


	#==================================[ DENSE 6 ]==================================#


	img_shape = pool5.get_shape()
	dense6_shape = int(np.prod(pool5.get_shape()[1:]))

	pool5_flat = tf.reshape(pool5, [-1, dense6_shape])

	with tf.variable_scope('fc6') as scope:

		vgg_weights = vgg_params['fc6_W']
		weights = tf.get_variable(name='weights', trainable=False,
			                      initializer=vgg_weights)

		vgg_biases = vgg_params['fc6_b']
		biases = tf.get_variable(name='biases', trainable=False,
			                     initializer=vgg_biases)

		dense6 = tf.nn.bias_add(tf.matmul(pool5_flat, weights), biases)
		activated6 = tf.nn.relu(dense6, name='out')

		drop6 = tf.nn.dropout(activated6, prob, name='drop6')


	#==================================[ DENSE 7 ]==================================#


	with tf.variable_scope('fc7') as scope:

		vgg_weights = vgg_params['fc7_W']
		weights = tf.get_variable(name='weights', trainable=False,
			                      initializer=vgg_weights)

		vgg_biases = vgg_params['fc7_b']
		biases = tf.get_variable(name='biases', trainable=False,
			                     initializer=vgg_biases)

		dense7 = tf.nn.bias_add(tf.matmul(drop6, weights), biases)
		activated7 = tf.nn.relu(dense7, name='out')

		drop7 = tf.nn.dropout(activated7, prob, name='drop7')


	#==================================[ DENSE 8 ]==================================#


	with tf.variable_scope('fc8') as scope:

		vgg_weights = vgg_params['fc8_W']
		weights = tf.get_variable(name='weights', trainable=True,
			                      initializer=vgg_weights)

		vgg_biases = vgg_params['fc8_b']
		biases = tf.get_variable(name='biases', trainable=True,
			                     initializer=vgg_biases)

		dense8 = tf.nn.bias_add(tf.matmul(drop7, weights), biases)
		activated8 = tf.nn.relu(dense8, name='out')


	#==================================[ DENSE 9 ]==================================#


	with tf.variable_scope('fc9') as scope:

		weights = tf.get_variable(name='weights', trainable=True,
			                      initializer=tf.truncated_normal([1000, CLASSES],
		                                                          dtype=tf.float32,
		                                                          mean=0,
		                                                          stddev=1e-1))

		biases = tf.get_variable(name='biases', trainable=True,
			                     initializer=tf.constant(1.0,
		                                                 shape=[CLASSES],
		                                                 dtype=tf.float32))

		logits = tf.nn.bias_add(tf.matmul(activated8, weights), biases)


	loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits))

	optimizer = tf.train.AdadeltaOptimizer(learning_rate=1e-3)

	train_op = optimizer.minimize(loss)



# Add ops to save and restore all the variables.
saver = tf.train.Saver()


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

	tf.summary.image('conv_1', activated1_2[:,:,:,:3])
	tf.summary.image('conv_2', activated2_2[:,:,:,:3])
	tf.summary.image('conv_3', activated3_3[:,:,:,:3])
	tf.summary.image('conv_4', activated4_3[:,:,:,:3])
	tf.summary.image('conv_5', activated5_3[:,:,:,:3])
	tf.summary.histogram('logits hist', logits)
	tf.summary.scalar('loss', loss)

	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter('run_3_tensorboard_pure', sess.graph)


if __name__ == '__main__':
    
	if len(sys.argv) > 1:
		if sys.argv[1] == 'restore':
			print('Restoring model', flush=True)
			saver.restore(sess, "./model_between_years/model.ckpt")

	best_loss = 9999
	step = 0
	try:
		while not coord.should_stop():

			# run training steps or whatever
			step += 1
			print("Step " + str(step), flush=True)

			# generate rotation angle
			rotation_angle = np.asarray([np.random.normal(0, 0.08)])

			feed_dict = {
				is_training : True,
				angle : rotation_angle,
				prob  : 0.5,
				images_tst : val_images[:VALIDATION_BATCH],
				labels_tst : val_labels[:VALIDATION_BATCH]
			}
			_, summary, training_loss, tt_img = \
			    sess.run([train_op, merged, loss, to_test], feed_dict=feed_dict)

			train_writer.add_summary(summary, step)
			print('Training Loss   : {0:.2f}'.format(training_loss))

			if step % 250 == 0:

				validation_loss = 0
				validation_wrong = 0
				validation_total = 0
				for s in range(0, len(val_images), VALIDATION_BATCH):
					try:
						# print(s / len(val_images))
						feed_dict = {
							is_training : False,
							angle : np.asarray([0]),
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

				if validation_loss < best_loss:
					best_loss = validation_loss
					save_path = saver.save(sess, "./model_pure/model.ckpt")
					print("Model saved in file: %s" % save_path)


	except tf.errors.OutOfRangeError:
		print('\nDone training -- epoch limit reached\n')
	finally:
		# When done, ask the threads to stop.
		coord.request_stop()

	# Wait for threads to finish.
	coord.join(threads)
	sess.close()
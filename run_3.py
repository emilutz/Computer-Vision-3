import os
import cv2
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



#=====[ HYPER-PARAMETERS ]=====#

BATCH_SIZE = 30
EPOCHS     = 4

#==============================#


set_label_dictionaries()

with tf.device('/cpu:0'):
	images, labels = inputs('training', BATCH_SIZE, EPOCHS)


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

try:
	while not coord.should_stop():
		# Run training steps or whatever

		image, label = sess.run([images, labels])
		print(label_to_class[label[14]])
		plt.imshow(image[14,:,:,0], cmap='gray')
		plt.show()

except tf.errors.OutOfRangeError:
	print('\nDone training -- epoch limit reached\n')
finally:
	# When done, ask the threads to stop.
	coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()
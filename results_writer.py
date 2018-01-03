import os

def write_results(prediction_tuples):

	# create the reverse label dictionary
	counter = 0
	reverse_label_dict = {}
	for classname in os.listdir('training'):
		reverse_label_dict[counter] = classname.lower()
		counter += 1

	# serialize the results
	with open('run2.txt', 'w') as f:
		for (img, pred) in prediction_tuples:
			f.write('{0}.jpg {1}\n'.format(img, reverse_label_dict[pred]))

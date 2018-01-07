from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from scipy import misc
import scipy.misc
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
from collections import namedtuple
from skimage import io, img_as_float
import re


ClassifierData = namedtuple('ClassifierData', 'data names')


class DataSet(object):
    """

    This class is responsible for loading, 
    storing and adapting data. Loads images
    from directories and converts them to the
    proper vector format.

    """

    def __init__(self):
        """
        Constructor which initializes variables
        responsible for storing images data with
        labels corresponding to them and classes
        dictorionary - to keep them as a integer
        values for saving memory.

        """

        self.images = []
        self.labels = []
        self.classes = dict()

    @staticmethod
    def crop_image(img):
        """
        Static method which allows to crop image.
        Makes a square by center based on lower
        value from height and width.

        Args:
            img (np.ndarray): numpy array of 
                              grayscale image

        Returns:
            img (np.ndarray): cropped image


        """

        # get minimum value to create sqaure 
        # with equal dimensions
        crop_size = min(img.shape)

        # starting point
        start = crop_size//2 - (crop_size//2)

        return img[start:start+crop_size, start:start+crop_size]


    def create_training_set(self, source_dir):
        """
        
        Function which creates the training set
        from given directory.

        Args:
            source_dir (str): directory with files

        """

        # irerate through base directory with directories 
        # of single class type which contains set of images
        # using enumrate to set integer value for each class
        # to save memory - reduce size of labels list
        for class_nr, source in enumerate(glob("%s*" % source_dir)):

            # load set of single class images
            class_set = DataSet.get_files_from_dir(source).data

            set_length = len(class_set)

            # get class name based on directory name
            class_name = os.path.basename(source)

            # store class name in dictionary using given
            # integer value 
            self.classes[class_nr] = class_name

            self.images.extend(class_set)

            # extent labels set with created list multiplied by 
            # images set length, all items in each iteration
            # will be the same class type
            self.labels.extend([class_nr] * set_length)

    @staticmethod
    def get_files_from_dir(source_dir):
        """
        Method for loading single image set from one directory.
        May be the same class - or testing images.

        Args:
            source_dir (str): directory string

        Returns:
            images (list): list of vectorized images
            names (list): names of the files
        """

        images = []
        names = []

        for image_path in glob("%s/*jpg" % source_dir):

            image_object = io.imread(image_path, as_grey=True)

            img = img_as_float(np.asarray(image_object)) 

            img = DataSet.crop_image(img)

            img = misc.imresize(img, (16, 16), mode='F')

            img = img.reshape(-1)

            images.append(img)
            names.append(os.path.split(image_path)[1])


        return ClassifierData(data=images, names=names)

    @staticmethod
    def normalize(data_set):
        """
        
        Static method for normalize the data

        Args:
            data_set (np.array): set of images

        Returns:
            Tuple of:
            data_set (np.array): modified data set
            names (list): means of each feature
            devs (list): std dev of each feature
        """

        means = []
        devs = []
        data_set = np.array(data_set)

        # iterate trough each feature row
        for i in range(len(data_set[0])):

            # get current row
            row = data_set[:, i]

            # calculate the mean and std deviation
            # of each feature separetly 
            mean = np.mean(row)
            std_dev = np.std(row)

            # replace numbers by substracting the
            # mean from each pixel and divide it 
            # by standard deviation
            data_set[:, i] -= mean
            data_set[:, i] /= std_dev
            
            # save it for future use
            # during the prediction
            means.append(mean)
            devs.append(std_dev)

        return data_set, means, devs


class ClassifierTester(object):
    """

    Class for main classifier implementation 
    and operations. It will require to load 
    testing set along with the labels.

    """

    def __init__(self, training_set, labels):
        """
        Constructor function.

        Args:
            training_set (np.array): set of images

            labels(list): list of labels of each image

        """

        self.values = dict()

        self.training_set = training_set
        self.labels = labels

        self.best_model = None
        self.means = []
        self.std_devs = []

    def get_best(self, max_neighbours=20, weights=['uniform', 'distance'], testing_size=0.2):
        """
        
        The most important function - it trains the classifier 
        with given parameters.

        Args:
            max_neigbours (int): max number of neigbours to train
            weights (list): list of weights parameters
            testing_size (int): number which tells how to split
                                data for testing and training sets

        Returns:
            self.best_model (KNeighborsClassifier): trained best model


        """

        for weight in weights:
            self.values[weight] = []

        best_values = {"neighbours": 1, "weights": weights[0]}
        best_accuracy = 0

        self.training_set, self.means, self.std_devs = DataSet.normalize(self.training_set)

        train_sets, test_sets, train_labels, test_labels = train_test_split(
            self.training_set, 
            self.labels, 
            test_size=testing_size,
            random_state=42
        )


        # test classifier for different values
        for neighbours in range(1, max_neighbours+1):
            
            for weight in weights:
                
                # initializate the classifier    
                model = KNeighborsClassifier(n_neighbors=neighbours, weights=weight)

                # train model and get the acurracy
                model.fit(train_sets, train_labels)
                accuracy = model.score(test_sets, test_labels) * 100

                # print informations about accuracies
                text = '#Neighbours: {:02d}, Weight type: {}'.format(neighbours, weight)
                print("{}, Accuracy: {:.2f}%".format(text, accuracy))

                # save results for plot
                self.values[weight].append(accuracy)

                # keep the best model saved
                if accuracy <= best_accuracy:
                    continue

                best_accuracy = accuracy
                best_values['neighbours'] = neighbours
                best_values['weights'] = weight

        # retrain model with best obtained parameters
        self.best_model = KNeighborsClassifier(n_neighbors=best_values['neighbours'], weights=best_values['weights'])
        self.best_model.fit(self.training_set, self.labels)

        return self.best_model

    def get_plot(self):
        """
        
        Function for ploting the comparision of accuracy.

        """

        if self.best_model is None:
            raise ValueError('Cannot get plots!')


        data = self.values

        number_of_items = len(data[list(data.keys())[0]])
        x_labels = list(range(1, number_of_items+1))

        plt.xticks(x_labels)
        for key, data in data.items():
            plt.plot(x_labels, data, label=key)
        plt.grid()
        plt.legend().set_title('Weight type')
        
        plt.show()


class Output(object):
    """

    Class for making output.

    """

    @staticmethod
    def atoi(text):
        return int(text) if text.isdigit() else text

    @staticmethod
    def natural_keys(text):
        text = text[0]
        return [Output.atoi(c) for c in re.split('(\d+)', text)]

    @staticmethod
    def normalize_single(data, values):

        for i, (mean, std_dev) in enumerate(values):
            
            data[i] -= mean
            data[i] /= std_dev

        return data

    @staticmethod
    def create(classifier, data_set, output_file, classes, norm_data):
        """
        Saving the file.

        Args:
            classifier (KNeighborsClassifier): trained classifier
            data_set (np.array): array of images
            output_file (str): directory of outpur file
            classes (dict): dictionary with class names
            normdata (list of dicts): mean and std_dev for each feature

        """

        #try:

        # Create output file
        with open(output_file, 'w') as out_file:

            # create zipped list of tuples and sort it later
            # to have the proper order to saving results
            # keeping the names - because there are few files
            # missing - use function to sort to keep right numeric
            # order - like 9 before the 10 
            output_list = zip(data_set.names, data_set.data)
            
            for name, image in sorted(output_list, key=Output.natural_keys):

                image = Output.normalize_single(image, norm_data)

                prediction_value = classifier.predict([image])[0]

                # find class name based on integer value in
                # classes dictionary
                prediction_class_name = classes[prediction_value]

                # Write output - pass number and actual predicted
                # class name in proper order
                out_file.write("{} {}\n".format(name, prediction_class_name))

        #except:
        #    print('Something goes wrong! The result file has not been saved.')


def get_args():
    """

    Get initialization arguments.

    """

    parser = argparse.ArgumentParser(description="K Neighbors \
                                     Classifier Script - #3 \
                                     Computer Vision Coursework")

    parser.add_argument("-s", "--source", default="training/", 
                        type=str,
                        help="Source training set directory \
                        (with classes as a directory names inside)\
                        #Leave blank for use default set")
 
    parser.add_argument("-t", "--testing", default="testing/", 
                        help="Testing set directory\
                        #Leave blank for use default set")

    parser.add_argument("-o", "--output", default="run_3.txt", 
                        help="Output file\
                        #Leave blank for use default set")

    return parser.parse_args()


if __name__ == "__main__":

    arguments = get_args()

    ### This part is mostly self commented ###

    if not os.path.isdir(arguments.source):
        print('Wrong source dir!')
        sys.exit(2)

    if not os.path.isdir(arguments.testing):
        print('Wrong testing dir!')
        sys.exit(2)

    training_data = DataSet()

    training_data.create_training_set(arguments.source)

    classifier = ClassifierTester(training_data.images, training_data.labels)

    best_classifier = classifier.get_best()

    testing_data = DataSet.get_files_from_dir(arguments.testing)

    normalization_data = zip(classifier.means, classifier.std_devs)

    Output.create(best_classifier, testing_data, arguments.output, training_data.classes, normalization_data)

    classifier.get_plot()

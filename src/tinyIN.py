# adapted from https://github.com/seshuad/IMagenet/blob/master/TinyImagenet.ipynb
"""
load tiny imagenet
"""
import os
import matplotlib
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import zipfile
from sklearn import preprocessing

ROOT = r'D:/tmp/tiny-imagenet'
BATCH_SIZE = 20
NUM_CLASSES = 200
NUM_IMAGES_PER_CLASS = 500
NUM_IMAGES = NUM_CLASSES * NUM_IMAGES_PER_CLASS
TRAINING_IMAGES_DIR = f'{ROOT}/tiny-imagenet-200/train/'
TRAIN_SIZE = NUM_IMAGES

NUM_VAL_IMAGES = 10000
VAL_IMAGES_DIR = f'{ROOT}/tiny-imagenet-200/val/'

IMAGE_SIZE = 64
NUM_CHANNELS = 3
IMAGE_ARR_SIZE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS
IMAGES_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'


def load_training_images(image_dir, batch_size=500):

    image_index = 0

    images = np.ndarray(shape=(NUM_IMAGES, IMAGE_ARR_SIZE))
    names = []
    labels = []

    # Loop through all the types directories
    for type in os.listdir(image_dir):
        if os.path.isdir(image_dir + type + '/images/'):
            type_images = os.listdir(image_dir + type + '/images/')
            # Loop through all the images of a type directory
            batch_index = 0
            # print ("Loading Class ", type)
            for image in type_images:
                image_file = os.path.join(image_dir, type + '/images/', image)

                # reading the images as they are; no normalization, no color editing
                image_data = mpimg.imread(image_file)
                # print ('Loaded Image', image_file, image_data.shape)
                if (image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)):
                    images[image_index, :] = image_data.flatten()

                    labels.append(type)
                    names.append(image)

                    image_index += 1
                    batch_index += 1
                if (batch_index >= batch_size):
                    break

    return (images, np.asarray(labels), np.asarray(names))

def get_label_from_name(data, name):
    for idx, row in data.iterrows():
        if (row['File'] == name):
            return row['Class']

    return None


def load_validation_images(testdir, validation_data, batch_size=NUM_VAL_IMAGES):
    labels = []
    names = []
    image_index = 0

    images = np.ndarray(shape=(batch_size, IMAGE_ARR_SIZE))
    val_images = os.listdir(testdir + '/images/')

    # Loop through all the images of a val directory
    batch_index = 0


    for image in val_images:
        image_file = os.path.join(testdir, 'images/', image)
        # print (testdir, image_file)

        # reading the images as they are; no normalization, no color editing
        image_data = mpimg.imread(image_file)
        if (image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)):
            images[image_index, :] = image_data.flatten()
            image_index += 1
            labels.append(get_label_from_name(validation_data, image))
            names.append(image)
            batch_index += 1

        if (batch_index >= batch_size):
            break

    print ("Loaded Validation images ", image_index)
    return (images, np.asarray(labels), np.asarray(names))



def plot_object(data):
    plt.figure(figsize=(1 ,1))
    image = data.reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    plt.imshow(image, cmap = matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    plt.show()

def plot_objects(instances, images_per_row=10, **options):
    size = IMAGE_SIZE
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size ,size ,NUM_CHANNELS) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        if (row == len(instances ) /images_per_row):
            break
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)/255
    plt.imshow(image, **options)
    plt.axis("off")
    plt.show()


training_images, training_labels, training_files = load_training_images(TRAINING_IMAGES_DIR, batch_size=BATCH_SIZE)
shuffle_index = np.random.permutation(len(training_labels))
training_images = training_images[shuffle_index]
training_labels = training_labels[shuffle_index]
training_files  = training_files[shuffle_index]

le = preprocessing.LabelEncoder()
training_le = le.fit(training_labels)
training_labels_encoded = training_le.transform(training_labels)

val_data = pd.read_csv(VAL_IMAGES_DIR + 'val_annotations.txt', sep='\t', header=None,
                       names=['File', 'Class', 'X', 'Y', 'H', 'W'])
val_images, val_labels, val_files = load_validation_images(VAL_IMAGES_DIR, val_data, batch_size=BATCH_SIZE)
val_labels_encoded = training_le.transform(val_labels)


def get_next_batch(batchsize=50):
    for cursor in range(0, len(training_images), batchsize):
        batch = []
        batch.append(training_images[cursor:cursor +batchsize])
        batch.append(training_labels_encoded[cursor:cursor +batchsize])
        yield batch

def get_next_labels(batchsize=50):
    for cursor in range(0, len(training_images), batchsize):
        yield training_labels_encoded[cursor:cursor +batchsize]

if __name__ == '__main__':
    print("First 30 Training Labels", training_labels_encoded[0:30])
    plot_objects(training_images[0:30])

    plot_objects(val_images[0:30])
    print(val_labels_encoded[0:30])

    print(training_images.shape)
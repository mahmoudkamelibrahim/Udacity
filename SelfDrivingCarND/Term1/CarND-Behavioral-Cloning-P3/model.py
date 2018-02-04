import cv2
import csv
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
import sklearn

def getLinesFromDrivingLogs(dataPath, skipHeader=False):
    """
    Returns the lines from a driving log with base directory `dataPath`.
    If the file include headers, pass `skipHeader=True`.
    """
    lines = []
    with open(dataPath + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        if skipHeader:
            next(reader, None)
        for line in reader:
            lines.append(line)
    return lines


def findImages():
    """
    Finds all the images needed for training on current directory
    Returns `([centerPaths], [leftPath], [rightPath], [measurement])`
    """
    directory = "./"
    centerTotal = []
    leftTotal = []
    rightTotal = []
    measurementTotal = []
    lines = getLinesFromDrivingLogs(directory)
    center = []
    left = []
    right = []
    measurements = []
    for line in lines:
        measurements.append(float(line[3]))
        center.append( line[0].strip())
        left.append(line[1].strip())
        right.append(line[2].strip())
    centerTotal.extend(center)
    leftTotal.extend(left)
    rightTotal.extend(right)
    measurementTotal.extend(measurements)
    return (centerTotal, leftTotal, rightTotal, measurementTotal)

def combineImages(center, left, right, measurement, correction):
    """
    Combine the image paths from `center`, `left` and `right` using the correction factor `correction`
    Returns ([imagePaths], [measurements])
    """
    imagePaths = []
    imagePaths.extend(center)
    imagePaths.extend(left)
    imagePaths.extend(right)
    measurements = []
    measurements.extend(measurement)
    measurements.extend([x + correction for x in measurement])
    measurements.extend([x - correction for x in measurement])
    return (imagePaths, measurements)
import os
from IPython import display
import pylab
pylab.rcParams['figure.figsize'] = (20, 15)
def generator(samples, batch_size=32):
    """
    Generate the required images and measurments for training/
    `samples` is a list of pairs (`imagePath`, `measurement`).
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        # shuffle data to ensure that data hasn't a specific pattern 
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            path_write='path'
            for imagePath, measurement in batch_samples:
                originalImage = cv2.imread(imagePath)
                # Next step is important to change the color space from BGR to RGB 
                # Without this step the trainer get confused about the colors and
                # run on the colored surroundings    
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)

                # Flipping images to generalize the model
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            inputs = np.array(images)
            outputs = np.array(angles)

            yield sklearn.utils.shuffle(inputs, outputs)

def createPreProcessingLayers():
    """
    Creates a model with the initial pre-processing layers.
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    return model

def nVidiaModel():
    """
    Creates nVidea Autonomous Car Group model
    """
    model = createPreProcessingLayers()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


# Reading images from their locations , dividing them to 3 arrays of Left camera Images , Right Camera Images and Center Camera Images and one array of angle measurements
centerPaths, leftPaths, rightPaths, measurements = findImages()


#Combine images from center,left and right using the correction factor=0.2 so we have one array of images and one array of their angle measurements
imagePaths, measurements = combineImages(centerPaths, leftPaths, rightPaths, measurements, 0.2)


# Splitting Images using train_test_split to :
# Training Images"80%" , Validation Images "20%" 

samples = list(zip(imagePaths, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('All Images: {}'.format( len(imagePaths)))
print('Training Images: {}'.format(len(train_samples)))
print('Validation Images: {}'.format(len(validation_samples)))

#Creating generators to process each part of data only when we need them, which is much more memory-efficient.
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Model creation
model = nVidiaModel()

# Compiling and training the model using adam optimizer
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
                 len(train_samples), validation_data=validation_generator, \
                 nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

model.save('model4Feb2nd.h5')
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

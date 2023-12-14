#!/usr/bin/env python3

"""
 * https://www.tensorflow.org/tutorials/keras/classification
 * ~/.keras/datasets/fashion-mnist/
"""

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy
from matplotlib import pyplot

from helpers import plot_image, plot_value_array

print("TensorFlow version:", tf.__version__)

####
# Import the Fashion MNIST dataset
####


# This guide uses the Fashion MNIST dataset which contains 70,000 grayscale images in 10 categories.
# The images show individual articles of clothing at low resolution (28 by 28 pixels)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# print('train_images, train_label', train_images, train_labels)

# Loading the dataset returns four NumPy arrays:
#
#     * The train_images and train_labels arrays are the training set—the data the model uses to learn.
#     * The model is tested against the test set, the test_images, and test_labels arrays.
#
# The images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255. The labels are an array of integers, ranging from 0 to 9. These correspond to the class of clothing the image represents:
# Label     Class
# 0     T-shirt/top
# 1     Trouser
# 2     Pullover
# 3     Dress
# 4     Coat
# 5     Sandal
# 6     Shirt
# 7     Sneaker
# 8     Bag
# 9     Ankle boot

# Each image is mapped to a single label. Since the class names are not included with the dataset, store them here to use later when plotting the images:

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

####
# Explore the data
####

# Let's explore the format of the dataset before training the model.
# The following shows there are 60,000 images in the training set, with each image represented as 28 x 28 pixels:

shape = train_images.shape
print('train_images', shape)

# Likewise, there are 60,000 labels in the training set, Each label is an integer between 0 and 9:

print('train_labels', len(train_labels), train_labels)

# There are 10,000 images in the test set. Again, each image is represented as 28 x 28 pixels:

shape = test_images.shape
print('test_images', shape)

# And the test set contains 10,000 images labels:

print('test_labels', len(test_labels), test_labels)

####
# Explore the data
####

# The data must be preprocessed before training the network.
# If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255:

# disabled:
# pyplot.figure()
# pyplot.imshow(train_images[0])
# pyplot.colorbar()
# pyplot.grid(False)
# pyplot.show()

# Scale these values to a range of 0 to 1 before feeding them to the neural network model.
# To do so, divide the values by 255.
# It's important that the training set and the testing set be preprocessed in the same way:

train_images = train_images / 255.0 # numpy array division
test_images = test_images / 255.0 # numpy array division

# To verify that the data is in the correct format and that you're ready to build and train the network,
# let's display the first 25 images from the training set and display the class name below each image.

# disabled:
# pyplot.figure(figsize=(10,10))
# for i in range(25):
#     pyplot.subplot(5,5,i+1)
#     pyplot.xticks([])
#     pyplot.yticks([])
#     pyplot.grid(False)
#     pyplot.imshow(train_images[i], cmap=pyplot.cm.binary)
#     pyplot.xlabel(class_names[train_labels[i]])
# pyplot.show()

####
# Build the model
####

## Set up the layers

# The basic building block of a neural network is the layer. Layers extract representations from the data fed into them.
# Hopefully, these representations are meaningful for the problem at hand.

# Most of deep learning consists of chaining together simple layers.
#Most layers, such as tf.keras.layers.Dense, have parameters that are learned during training.

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# The first layer in this network, tf.keras.layers.Flatten,
# transforms the format of the images from a two-dimensional array (of 28 by 28 pixels)
# to a one-dimensional array (of 28 * 28 = 784 pixels).
# Think of this layer as unstacking rows of pixels in the image and lining them up
# This layer has no parameters to learn; it only reformats the data.

# After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers.
# These are densely connected, or fully connected, neural layers.
# The first Dense layer has 128 nodes (or neurons).
# The second (and last) layer returns a logits array with length of 10.
# Each node contains a score that indicates the current image belongs to one of the 10 classes.

####
# Compile the model
####

# Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:
#
# * Optimizer — This is how the model is updated based on the data it sees and its loss function.
# * Loss function — This measures how accurate the model is during training.
#       You want to minimize this function to "steer" the model in the right direction.
# * Metrics — Used to monitor the training and testing steps.
#       The following example uses accuracy, the fraction of the images that are correctly classified.
#
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

####
# Train the model
# Training the neural network model requires the following steps:
#
# 1. Feed the training data to the model. In this example, the training data is in the train_images and train_labels arrays.
# 2. The model learns to associate images and labels.
# 3. You ask the model to make predictions about a test set—in this example, the test_images array.
# 4. Verify that the predictions match the labels from the test_labels array.
####

## 1. Feed the model

# To start training, call the model.fit method—so called because it "fits" the model to the training data:

model.fit(train_images, train_labels, epochs=10)

# As the model trains, the loss and accuracy metrics are displayed. This model reaches an accuracy of about 0.91 (or 91%) on the training data.

# Evaluate accuracy
# Next, compare how the model performs on the test dataset:

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print()
print(' - Test loss:', test_loss)
print(' - Test accuracy:', test_acc)

## https://www.baeldung.com/cs/ml-loss-accuracy
# Loss:
#   Loss is a value that represents the summation of errors in our model.
#   It measures how well (or bad) our model is doing.
#   If the errors are high, the loss will be high, which means that the model does not do a good job.
#   Otherwise, the lower it is, the better our model works.
# Accuracy:
#   Having a low accuracy but a high loss would mean that the model makes big errors in most of the data.
#   But, if both loss and accuracy are low, it means the model makes small errors in most of the data.
#   However, if they’re both high, it makes big errors in some of the data.
#   Finally, if the accuracy is high and the loss is low, then the model makes small errors on just some of the data,
#   which would be the ideal case.
##

## https://www.obviously.ai/post/the-difference-between-training-data-vs-test-data-in-machine-learning
# What is Testing Data?
# Once your machine learning model is built (with your training data),
# you need unseen data to test your model. This data is called testing data,
# and you can use it to evaluate the performance and progress of your algorithms’ training and adjust or optimize it for improved results.
##

## https://elitedatascience.com/overfitting-in-machine-learning
# Overfitting ... Our model doesn’t generalize well from our training data to unseen data.
# Underfitting occurs when a model is too simple – informed by too few features or regularized too much – which makes it inflexible in learning from the dataset.
#
# The Bias-Variance Tradeoff
# - high bias => low variance (too simple)
# - low bias => high variance (too complex)
#
# This trade-off between too simple (high bias) vs. too complex (high variance)
# is a key concept in statistics and machine learning,
# and one that affects all supervised learning algorithms.
##

# It turns out that the accuracy on the test dataset is a little less than the accuracy on the training dataset.
# This gap between training accuracy and test accuracy represents overfitting.
# Overfitting happens when a machine learning model performs worse on new, previously unseen inputs than it does on the training data.

####
# Make predictions
####

# With the model trained, you can use it to make predictions about some images.
# Attach a softmax layer to convert the model's linear outputs—logits—to probabilities, which should be easier to interpret.

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

print('predictions[0]', predictions[0])

# A prediction is an array of 10 numbers.
# They represent the model's "confidence" that the image corresponds to each of the 10 different articles of clothing.
# You can see which label has the highest confidence value:

res = numpy.argmax(predictions[0])
print('max predictions[0]', res) # '9'

# So, the model is most confident that this image is an ankle boot, or class_names[9].
# Examining the test label shows that this classification is correct:

assert res == test_labels[0]


####
# Verify predictions
####

# With the model trained, you can use it to make predictions about some images.
# Let's look at the 0th image, predictions, and prediction array.
# Correct prediction labels are blue and incorrect prediction labels are red.
# The number gives the percentage (out of 100) for the predicted label.

# disabled:
# i = 0
# pyplot.figure(figsize=(6,3))
# pyplot.subplot(1,2,1)
# plot_image(i, predictions[i], test_labels, test_images, class_names)
# pyplot.subplot(1,2,2)
# plot_value_array(i, predictions[i],  test_labels)
# pyplot.show()
#
# i = 12
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions[i],  test_labels)
# plt.show()

# Let's plot several images with their predictions. Note that the model can be wrong even when very confident.

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.

# disabled:
# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# pyplot.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#   pyplot.subplot(num_rows, 2*num_cols, 2*i+1)
#   plot_image(i, predictions[i], test_labels, test_images, class_names)
#   pyplot.subplot(num_rows, 2*num_cols, 2*i+2)
#   plot_value_array(i, predictions[i], test_labels)
# pyplot.tight_layout()
# pyplot.show()

####
# Use the trained model
####

# Finally, use the trained model to make a prediction about a single image.
# Grab an image from the test dataset.
img = test_images[1]

print('img.shape', img.shape)

# tf.keras models are optimized to make predictions on a batch, or collection, of examples at once. Accordingly, even though you're using a single image, you need to add it to a list:
# Add the image to a batch where it's the only member.
img = (numpy.expand_dims(img, 0))

print('img.shape', img.shape)

# Now predict the correct label for this image:

predictions_single = probability_model.predict(img)
print('predictions_single', predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = pyplot.xticks(range(10), class_names, rotation=45)
pyplot.show()

res = numpy.argmax(predictions_single[0])
print('numpy.argmax(predictions_single[0])', res)

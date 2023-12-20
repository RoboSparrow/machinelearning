#!/usr/bin/env python3

"""
 * https://www.tensorflow.org/tutorials/keras/regression

 jb this tut seems incomplete and throws some errors (noted below)
"""

from  matplotlib import pyplot
import numpy
import pandas
import seaborn

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from helpers import plot_loss, plot_horsepower


# Make NumPy printouts easier to read.
numpy.set_printoptions(precision=3, suppress=True)

print(tf.__version__)

"""
This tutorial uses the classic Auto MPG dataset and demonstrates how to build models
to predict the fuel efficiency of the late-1970s and early 1980s automobiles.
"""

####
# Get the data
####

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pandas.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()
print('dataset.tail', dataset.tail())

####
# Clean the data
####

# The dataset contains a few unknown values
# Drop those rows to keep this initial tutorial simple:

unknown = dataset.isna().sum()
# print('unknown', unknown)
dataset = dataset.dropna()

# The "Origin" column is categorical, not numeric.
# So the next step is to one-hot encode the values in the column with pd.get_dummies.
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pandas.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
# print('dataset.tail', dataset.tail())

####
# Split the data into training and test sets
####

# Now, split the dataset into a training set and a test set.
# You will use the test set in the final evaluation of your models.

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# The top row suggests that the fuel efficiency (MPG) is a function of all the other parameters.
# The other rows indicate they are functions of each other.
# seaborn.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
# pyplot.show()

# Let's also check the overall statistics. Note how each feature covers a very different range:
res = train_dataset.describe().transpose()
# print(res)

####
# Split features from labels
####

# Separate the target value—the "label"—from the features.
# This label is the value that you will train the model to predict.
# JB: 'mpg' is the value we try to predict, so we do remove it from test and train

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

####
# Normalization
####

# In the table of statistics it's easy to see how different the ranges of each feature are:

res = train_dataset.describe().transpose()[['mean', 'std']]
print(res)

# It is good practice to normalize features that use different scales and ranges.
# One reason this is important is because the features are multiplied by the model weights.
# So, the scale of the outputs and the scale of the gradients are affected by the scale of the inputs.
# Although a model might converge without feature normalization, normalization makes training much more stable.

####
# The Normalization layer
####

# The tf.keras.layers.Normalization is a clean and simple way to add feature normalization into your model.
# The first step is to create the layer:

normalizer = tf.keras.layers.Normalization(axis=-1)

# Then, fit the state of the preprocessing layer to the data by calling Normalization.adapt:
# print(numpy.array(train_features))

## JB: error tutorial ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type int).
## https://stackoverflow.com/a/60750937 (array contains ints and Booleans)
# original: normalizer.adapt(numpy.array(train_features))
x = numpy.asarray(train_features).astype('float32')
normalizer.adapt(x)

# Calculate the mean and variance, and store them in the layer:
res = normalizer.mean.numpy()
print('normalizer.mean', res)

# When the layer is called, it returns the input data, with each feature independently normalized:

first = numpy.array(train_features[:1])
with numpy.printoptions(precision=2, suppress=True):
    print('First example:', first)
    print()
    ## JB: error tutorial ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type int).
    ## https://stackoverflow.com/a/60750937 (array contains ints and Booleans)
    # print('Normalized:', normalizer(first).numpy())
    x = first.astype('float32')
    print('Normalized:', normalizer(x))

####
# Linear regression
####

## TRAINING: Linear regression with one variable

# Begin with a single-variable linear regression to predict 'MPG' from 'Horsepower'.

# There are two steps in your single-variable linear regression model:
#   1. Normalize the 'Horsepower' input features using the tf.keras.layers.Normalization preprocessing layer.
#   2. Apply a linear transformation (y = mx + b) to produce 1 output using a linear layer (tf.keras.layers.Dense).

# - First, create a NumPy array made of the 'Horsepower' features.
# - Then, instantiate the tf.keras.layers.Normalization and fit its state to the horsepower data:

horsepower = numpy.array(train_features['Horsepower'])
horsepower_normalizer = layers.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)

# Build the Keras Sequential model:
# This model will predict 'MPG' from 'Horsepower'.
horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])
horsepower_model.summary()

# Run the untrained model on the first 10 'Horsepower' values.
# The output won't be good, but notice that it has the expected shape of (10, 1):

horsepower_model.predict(horsepower[:10])

# Once the model is built, configure the training procedure using the Keras Model.compile method.
# The most important arguments to compile are the loss and the optimizer,
# since these define what will be optimized (mean_absolute_error) and how (using the tf.keras.optimizers.Adam).

horsepower_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

# Use Keras Model.fit to execute the training for 100 epochs:

history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

# Visualize the model's training progress using the stats stored in the history object:

hist = pandas.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

plot_loss(history)
# pyplot.show()

# Collect the results on the test set for later:

test_results = {}

test_results['horsepower_model'] = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=0)

# Since this is a single variable regression, it's easy to view the model's predictions as a function of the input:

x = tf.linspace(0.0, 250, 251) # Generates evenly-spaced values in an interval along a given axis
y = horsepower_model.predict(x)

plot_horsepower(train_features, train_labels, x, y)
# pyplot.show()

####
# Linear regression with multiple inputs
####

# You can use an almost identical setup to make predictions based on multiple inputs.
# This model still does the same 'y = mx + b' except that 'm' is a matrix and is a 'x' vector.

linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

# When you call Model.predict on a batch of inputs, it produces units=1 outputs for each example:

# jb: original thowing ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type int).
#     - ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type int).

train_features_fix = numpy.asarray(train_features[:10]).astype(numpy.float32)
linear_model.predict(train_features_fix)

# Configure the model with Keras Model.compile and train with Model.fit for 100 epochs:

linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


history = linear_model.fit(
    train_features_fix,
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

# Using all the inputs in this regression model achieves a much lower training
# and validation error than the horsepower_model, which had one input:

# jb graph shows bs
plot_loss(history)
pyplot.show()



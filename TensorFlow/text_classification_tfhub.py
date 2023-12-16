#!/usr/bin/env python3

"""
 * https://www.tensorflow.org/tutorials/keras/text_classification_with_hub

 https://www.pinecone.io/learn/vector-embeddings/
"""

import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

####
# Download the IMDB dataset
####

# Split the training set into 60% and 40% to end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

# disabled:
# print('train_examples_batch', train_examples_batch)
# print('train_labels_batch', train_labels_batch)

####
# Build the model
####

# For this example you use a pre-trained text embedding model from TensorFlow Hub called google/nnlm-en-dim50/2.

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
print ("hub_layer()", hub_layer(train_examples_batch[:3]))

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

print('model.summary()', model.summary())

# The layers are stacked sequentially to build the classifier:
#
#    1. The first layer is a TensorFlow Hub layer.
#       This layer uses a pre-trained Saved Model to map a sentence into its embedding vector.
#       The pre-trained text embedding model that you are using (google/nnlm-en-dim50/2) splits the sentence into tokens,
#       embeds each token and then combines the embedding.
#       The resulting dimensions are: (num_examples, embedding_dimension).
#       For this NNLM model, the embedding_dimension is 50.
#    2. This fixed-length output vector is piped through a fully-connected (Dense) layer with 16 hidden units.
#    3. The last layer is densely connected with a single output node.

####
# compile the model.
####

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

####
# Train the model
####

# Train the model for 10 epochs in mini-batches of 512 samples.
# This is 10 iterations over all samples in the x_train and y_train tensors.
# While training, monitor the model's loss and accuracy on the 10,000 samples from the validation set:

history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=10,
                    validation_data=validation_data.batch(512),
                    verbose=1)

####
# Evaluate the model
####

# And let's see how the model performs. Two values will be returned. Loss (a number which represents our error, lower values are better), and accuracy.

results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))

# This fairly naive approach achieves an accuracy of about 87%. With more advanced approaches, the model should get closer to 95%.

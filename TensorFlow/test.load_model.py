#!/usr/bin/env python3

"""
 * https://www.tensorflow.org/tutorials/

 Load am\nd ude saved model from text_classification.py tutorial
"""
from os import path

import tensorflow as tf

from helpers import custom_standardization


mpath = './exports/text_classification_exported.keras'
if not path.exists(mpath):
    print(' * [Error] model \'{}\' missing: run ./text_classification.py tutporial first'.format(mpath))
    exit(1)

# model.save('my_model.keras')
model = tf.keras.models.load_model(mpath)

# Show the model architecture
print("model.summary", model.summary())

examples = [
    "excellent",
    "The movie was great!",
    "The movie was okay.",
    "The movie was terrible...",
    "what a bad movie",
    "fdsgsdg",
]

predictions = model.predict(examples)
print("predictions:", predictions)

# @see text_classification.py:358
# predictions: [[0.67221284]
#  [0.61775494]
#  [0.43876526]
#  [0.35616687]
#  [0.38900152]
#  [0.5116102 ]]

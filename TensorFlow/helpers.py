"""
 * https://www.tensorflow.org/tutorials/
"""
import re
import string

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy
from matplotlib import pyplot


####
# image classification helpers
####

def plot_value_array(i, predictions_array, true_label):
    """
     - https://www.tensorflow.org/tutorials/keras/classification#make_predictions
    """
    true_label = true_label[i]
    pyplot.grid(False)
    pyplot.xticks(range(10))
    pyplot.yticks([])
    thisplot = pyplot.bar(range(10), predictions_array, color="#777777")
    pyplot.ylim([0, 1])
    predicted_label = numpy.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def plot_image(i, predictions_array, true_label, img, class_names):
    """
     - https://www.tensorflow.org/tutorials/keras/classification#make_predictions
    """
    true_label, img = true_label[i], img[i]
    pyplot.grid(False)
    pyplot.xticks([])
    pyplot.yticks([])

    pyplot.imshow(img, cmap=pyplot.cm.binary)

    predicted_label = numpy.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    pyplot.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100 * numpy.max(predictions_array),
                                    class_names[true_label]),
                                    color=color)

####
# text classification helpers
####

@keras.saving.register_keras_serializable()
def custom_standardization(input_data):
    """
     - https://www.tensorflow.org/tutorials/keras/text_classification

    As you saw above, the reviews contain various HTML tags like <br />. These tags will not be removed by the default standardizer in the TextVectorization layer (which converts text to lowercase and strips punctuation by default, but doesn't strip HTML).
    You will write a custom standardization function to remove the HTML.
    """
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                  '')


####
# text regression helpers
####

def plot_loss(history):
    """
     - https://www.tensorflow.org/tutorials/keras/regression
    """
    pyplot.plot(history.history['loss'], label='loss')
    pyplot.plot(history.history['val_loss'], label='val_loss')
    pyplot.ylim([0, 10])
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Error [MPG]')
    pyplot.legend()
    pyplot.grid(True)
    # pyplot.show()


def plot_horsepower(train_features, train_labels, x, y):
    """
     - https://www.tensorflow.org/tutorials/keras/regression
    """
    pyplot.scatter(train_features['Horsepower'], train_labels, label='Data')
    pyplot.plot(x, y, color='k', label='Predictions')
    pyplot.xlabel('Horsepower')
    pyplot.ylabel('MPG')
    pyplot.legend()
    # pyplot.show()


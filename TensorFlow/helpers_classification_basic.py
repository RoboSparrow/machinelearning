

"""
 * https://www.tensorflow.org/tutorials/keras/classification
"""

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy
from matplotlib import pyplot


def plot_value_array(i, predictions_array, true_label):
    """
    https://www.tensorflow.org/tutorials/keras/classification#make_predictions
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
    https://www.tensorflow.org/tutorials/keras/classification#make_predictions
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

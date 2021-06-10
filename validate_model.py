from __future__ import print_function

import os

import tensorflow as tf
import numpy as np

import data_helpers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

my_path = "C:/Users/Teodor/Documents/NBU/NETB380 Programming Practice/trainind_data_creation/Generated/TimesNewRoman/"

images, labels = data_helpers.load(file_path=my_path, number=-1)

images = np.array(images)
labels = np.array(labels)

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(images)))
images = images[shuffle_indices]
labels = labels[shuffle_indices]

print(len(labels))

# Network Parameters
n_input = 32 * 32  # MNIST data input (img shape: 28*28)
n_classes = 26 + 26 + 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def avgpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 32, 32, 1])
    # x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    out = tf.subtract(out, tf.reduce_min(out))
    out = tf.divide(out, tf.reduce_mean(out))
    return out


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 64])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 64, 128])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([8 * 8 * 128, 2048])),
    # 'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([2048, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bd1': tf.Variable(tf.random_normal([2048])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Initializing the variables
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:

    saver.restore(sess, os.curdir + "/checkpoints/model-maxpool.ckpt")

    # Construct model
    pred = conv_net(x, weights, biases, keep_prob)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={x: images,
                                        y: labels,
                                        keep_prob: 1.}))
    for image in images:
        predictions = sess.run(pred, feed_dict={x: [image], keep_prob: 1.})
        data_helpers.printBMP(image)
        print(data_helpers.getTop5Pred(predictions))

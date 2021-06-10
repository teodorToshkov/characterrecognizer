from __future__ import print_function

import os

import tensorflow as tf
import datetime
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np

import data_helpers

my_path = "C:/Users/Teodor/Documents/NBU/NETB380 Programming Practice/trainind_data_creation/Generated/test2/"
tensorboard_run_name = "/super_small/"

# Parameters
learning_rate = 0.001
epochs = 200
batch_size = 128
display_step = 10

train_data_percentage = 0.1

# Network Parameters
n_input = 32 * 32
n_classes = 26 + 26 + 10
dropout = 0.75

conv1_strides = 1
maxpool1_strides = 2

conv2_strides = 1
maxpool2_strides = 2

tensorboard_run_name += "conv_strides=" + str(conv1_strides) + ";"\
                        + "max-pool_strides=" + str(maxpool1_strides)





os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

images, labels = data_helpers.load(file_path=my_path)

images = np.array(images)
labels = np.array(labels)

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(images)))
images = images[shuffle_indices]
labels = labels[shuffle_indices]

# Splitting the data into training and validation sets
train_data_len = min(int(len(images) * train_data_percentage), 1500)
images_test = images[:train_data_len]
labels_test = labels[:train_data_len]
images = images[train_data_len:]
labels = labels[train_data_len:]

print(len(labels))
print(len(labels_test))

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input], name='input')
y = tf.placeholder(tf.float32, [None, n_classes], name='output')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    with tf.name_scope('convolution'):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)


def maxpool2d(x, k=2):
    with tf.name_scope('max-pool'):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')


def avgpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def conv_net(x, dropout):
    with tf.name_scope('reshape_the_picture'):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, 32, 32, 1])

    with tf.name_scope('conv1'):
        # biases
        bc1 = tf.Variable(tf.random_normal([20]), name="bias_conv1")
        # weights
        wc1 = tf.Variable(tf.random_normal([5, 5, 1, 20]), name="weights_conv1")
        tf.summary.histogram('conv1_weights', wc1)
        tf.summary.histogram('conv1_biases', bc1)
        # Convolution Layer
        conv1 = conv2d(x, wc1, bc1, conv1_strides)
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=maxpool1_strides)
        tf.summary.histogram('conv1_activations', conv1)

    with tf.name_scope('conv2'):
        # biases
        bc2 = tf.Variable(tf.random_normal([15]), name="bias_conv2")
        # weights
        wc2 = tf.Variable(tf.random_normal([5, 5, 20, 15]), name="weights_conv2")
        tf.summary.histogram('conv2_weights', wc2)
        tf.summary.histogram('conv2_biases', bc2)
        # Convolution Layer
        conv2 = conv2d(conv1, wc2, bc2, conv1_strides)
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=maxpool2_strides)
        tf.summary.histogram('conv2_activations', conv2)

    with tf.name_scope('fc'):
        # biases
        bd1 = tf.Variable(tf.random_normal([100]), name="bias_fc")
        # weights
        shape = conv2.get_shape().as_list()
        input_size = shape[1] * shape[2] * shape[3]
        wd1 = tf.Variable(tf.random_normal([input_size, 100]), name="weights_fc")
        tf.summary.histogram('fc_weights', wd1)
        tf.summary.histogram('fc_biases', bd1)
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, wd1), bd1)
        fc1 = tf.nn.relu(fc1)
        tf.summary.histogram('fc_activations', fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

    with tf.name_scope('out'):
        # biases
        bout = tf.Variable(tf.random_normal([n_classes]), name="bias_out")
        # weights
        wout = tf.Variable(tf.random_normal([200, n_classes]), name="weights_out")
        tf.summary.histogram('out_weights', wout)
        tf.summary.histogram('out_biases', bout)
        # Output, class prediction
        out = tf.add(tf.matmul(fc1, wout), bout)
        out = tf.subtract(out, tf.reduce_min(out))
        out = tf.divide(out, tf.reduce_mean(out))
        tf.summary.histogram('out_activations', out)
    return out

# Construct model
pred = conv_net(x, keep_prob)

with tf.name_scope('optimizer'):
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
tf.summary.scalar('cost', cost)

with tf.name_scope('accuracy'):
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# tf.summary.image('input', images_test, )



# Initializing the variables
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

merged_summary_op = tf.summary.merge_all()

# Launch the graph
with tf.Session(config=config) as sess:
    sess.run(init)
    # saver.restore(sess, os.curdir + "/checkpoints/new/model-maxpool.ckpt")




    # Create randomly initialized embedding weights which will be trained.
    N = 62  # Number of items (vocab size).
    D = 200  # Dimensionality of the embedding.
    embedding_var = tf.Variable(tf.random_normal([N, D]), name='word_embedding')

    # Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
    config = projector.ProjectorConfig()

    # You can add multiple embeddings. Here we add only one.
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(os.curdir + '/tensorboard' + tensorboard_run_name + ";test", 'metadata.tsv')

    train_writer = tf.summary.FileWriter(os.curdir + '/tensorboard' + tensorboard_run_name + ";train", sess.graph)
    test_writer = tf.summary.FileWriter(os.curdir + '/tensorboard' + tensorboard_run_name + ";test", sess.graph)

    # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
    # read this file during startup.
    projector.visualize_embeddings(test_writer, config)



    # Evaluate model
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={x: images[:1000],
                                        y: labels[:1000],
                                        keep_prob: 1.}))

    step = 1
    # Keep training until reach max iterations
    batches = data_helpers.batch_iter(list(zip(images, labels)), batch_size, epochs)
    batches = list(batches)
    for batch in batches:
        batch_x, batch_y = zip(*batch)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate validation set loss and accuracy
            loss, acc, summary = sess.run([cost, accuracy, merged_summary_op], feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: dropout})
            loss_test, acc_test, summary_test = sess.run([cost, accuracy, merged_summary_op],
                                                         feed_dict={x: images_test,
                                                         y: labels_test,
                                                         keep_prob: 1.})
            time_str = datetime.datetime.now().isoformat()
            print(time_str + ": Iter " + str(step * batch_size) + "/" + str(epochs * len(labels)) +
                  " Process: {:.5f}".format(step * batch_size / epochs / len(labels) * 100) +
                  "% Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " +
                  "{:.5f}".format(acc) + ", Validation Loss= " +
                  "{:.6f}".format(loss_test) + ", Validation Accuracy= " +
                  "{:.5f}".format(acc_test))

            train_writer.add_summary(summary, step)
            test_writer.add_summary(summary_test, step)
        # else:
        #     train_writer.add_summary(acc, step)
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 1000 test images
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={x: images[:1000],
                                        y: labels[:1000],
                                        keep_prob: 1.}))
    # Save the variables to disk.
    save_path = saver.save(sess, os.curdir + '/tensorboard/super_small/model.ckpt')
    print("Model saved in file: %s" % save_path)

train_writer.close()
test_writer.close()

import tensorflow as tf
from svhn import SVHN
import matplotlib.pyplot as plt
from random import randint

# Parameters
learning_rate = 1e-3
iterations = 4000
batch_size = 300
display_step = 100

# Network Parameters
channels = 3
image_size = 32
n_classes = 10
dropout = 1.0
hidden_1 = 256
hidden_2 = 128
depth_1 = 16
depth_2 = 32
depth_3 = 64
filter_size = 5
normalization_offset = 0.0  # beta
normalization_scale = 1.0  # gamma
normalization_epsilon = 0.001  # epsilon


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(1.0, shape=shape))


def convolution(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# Load data
svhn = SVHN("../res", n_classes, use_extra=False, gray=False)

# Create the model
X = tf.placeholder(tf.float32, [None, image_size, image_size, channels])
Y = tf.placeholder(tf.float32, [None, n_classes])

# Weights & Biases
weights = {
    "layer1": weight_variable([filter_size, filter_size, channels, depth_1]),
    "layer2": weight_variable([filter_size, filter_size, depth_1, depth_2]),
    "layer3": weight_variable([filter_size, filter_size, depth_2, depth_3]),
    "layer4": weight_variable([image_size // 8 * image_size // 8 * depth_3, hidden_1]),
    "layer5": weight_variable([hidden_1, hidden_2]),
    "layer6": weight_variable([hidden_2, n_classes])
}

biases = {
    "layer1": bias_variable([depth_1]),
    "layer2": bias_variable([depth_2]),
    "layer3": bias_variable([depth_3]),
    "layer4": bias_variable([hidden_1]),
    "layer5": bias_variable([hidden_2]),
    "layer6": bias_variable([n_classes])
}


def normalize(x):
    """ Applies batch normalization """
    mean, variance = tf.nn.moments(x, [1, 2, 3], keep_dims=True)
    return tf.nn.batch_normalization(x, mean, variance, normalization_offset, normalization_scale,
                                     normalization_epsilon)


def cnn(x):
    # Batch normalization
    x = normalize(x)

    # Convolution 1 -> RELU -> Max Pool
    convolution1 = convolution(x, weights["layer1"])
    relu1 = tf.nn.relu(convolution1 + biases["layer1"])
    maxpool1 = max_pool(relu1)

    # Convolution 2 -> RELU -> Max Pool
    convolution2 = convolution(maxpool1, weights["layer2"])
    relu2 = tf.nn.relu(convolution2 + biases["layer2"])
    maxpool2 = max_pool(relu2)

    # Convolution 3 -> RELU -> Max Pool
    convolution3 = convolution(maxpool2, weights["layer3"])
    relu3 = tf.nn.relu(convolution3 + biases["layer3"])
    maxpool3 = max_pool(relu3)

    # Fully Connected Layer
    shape = maxpool3.get_shape().as_list()
    reshape = tf.reshape(maxpool3, [-1, shape[1] * shape[2] * shape[3]])
    fc4 = tf.nn.relu(tf.matmul(reshape, weights["layer4"]) + biases["layer4"])

    # Dropout Layer
    keep_prob_constant = tf.placeholder(tf.float32)
    fc4_dropout = tf.nn.dropout(fc4, keep_prob_constant)

    fc5 = tf.nn.relu(tf.matmul(fc4_dropout, weights["layer5"]) + biases["layer5"])

    return tf.matmul(fc5, weights["layer6"]) + biases["layer6"], keep_prob_constant


# Build the graph for the deep net
y_conv, keep_prob = cnn(X)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_conv))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    # Initialize Tensorflow variables
    sess.run(tf.global_variables_initializer())

    # Variables useful for batch creation
    start = 0
    end = 0

    # The accuracy and loss for every iteration in train and test set
    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []
    over_fit_vals = []

    for i in range(iterations):
        ind_train = [randint(0, svhn.train_examples - 1) for _ in range(batch_size)]
        ind_test = [randint(0, svhn.test_examples - 1) for _ in range(batch_size)]

        batch_x = svhn.train_data[ind_train]
        batch_y = svhn.train_labels[ind_train]

        batch_x_test = svhn.test_data[ind_test]
        batch_y_test = svhn.test_labels[ind_test]

        sess.run(optimizer, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 0.8})

        if (i + 1) % display_step == 0 or i == 0:
            _accuracy_train, _cost_train = sess.run([accuracy, cost],
                                                    feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
            _accuracy_test, _cost_test = sess.run([accuracy, cost],
                                                  feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.0})
            over_fit_vals.append(_cost_train / _cost_test)
            print("Step: {0:6d}, Training Accuracy: {1:5f}, Batch Loss: {2:5f}".format(i + 1, _accuracy_train,
                                                                                       _cost_train))
            print("Step: {0:6d}, Test Accuracy: {1:5f}, Batch Loss: {2:5f}".format(i + 1, _accuracy_test,
                                                                                   _cost_test))
            train_accuracies.append(_accuracy_train)
            train_losses.append(_cost_train)
            test_accuracies.append(_accuracy_test)
            test_losses.append(_cost_test)

    plt.subplot(311)
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    test_accuracies_legend, = plt.plot(test_accuracies, label='test')
    train_accuracies_legend, = plt.plot(train_accuracies, label='train')
    plt.legend([test_accuracies_legend, train_accuracies_legend], ['Train', 'Test'])

    plt.subplot(312)
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    train_losses_legend, = plt.plot(train_losses, label='train_mse')
    test_losses_legend, = plt.plot(test_losses, label='test_mse')
    plt.legend([train_losses_legend, test_losses_legend], ['Test', 'Train'])

    plt.subplot(313)
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel(r'$v_{ofc}$')
    plt.plot(over_fit_vals)

    plt.show()
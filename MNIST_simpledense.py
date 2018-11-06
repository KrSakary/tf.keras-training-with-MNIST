import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

# Mnist data load, x_train: 60,000, x_test: 10,000, labels: 10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

x_train = np.reshape(x_train, newshape=[-1, 28, 28, 1])
x_test = np.reshape(x_test, newshape=[-1, 28, 28, 1])
data_num_train = 60000
data_num_test = 10000
batch_size = 1000


# Placeholders
X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='X')
X_flat = tf.reshape(X, shape=[-1, 28*28])
Labels = tf.placeholder(dtype=tf.int32, shape=[None])
Labels_onehot = tf.one_hot(Labels, depth=10, axis=-1)
is_train = tf.placeholder(dtype=tf.bool, name='is_train')

def model_simpledense(input_shape):
    inputs = tf.keras.layers.Input(input_shape)

    dense_0 = tf.keras.layers.Dense(units=784)(inputs)
    bn_0 = tf.keras.layers.BatchNormalization()(dense_0)
    relu_0 = tf.keras.layers.ReLU()(bn_0)
    
    dense_1 = tf.keras.layers.Dense(units=392)(relu_0)
    bn_1 = tf.keras.layers.BatchNormalization()(dense_1)
    relu_1 = tf.keras.layers.ReLU()(bn_1)

    dense_1 = tf.keras.layers.Dense(units=10)(relu_1)

    model = tf.keras.Model(inputs=inputs, outputs=dense_1)

    return model

simpledense = model_simpledense(X_flat.shape[1:])
logits = simpledense(X_flat)
softmax = tf.nn.softmax(logits, axis=-1)

loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=Labels_onehot)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(softmax, axis=-1), tf.argmax(Labels_onehot, axis=-1)), tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss, var_list=simpledense.trainable_variables)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    list_loss = []

    for epoch in range(100):

        t_start = time.time()

        for batch in range(0, data_num_train, batch_size):


            l, acc, _ = sess.run((loss, accuracy, optimizer), feed_dict={X: x_train[batch:batch + batch_size], Labels: y_train[batch:batch + batch_size]})
            list_loss.append(l)

        t_end = time.time()

        if epoch % 10 == 0:
            print('Epoch:', epoch, ' / loss:', l, ' / accuracy(train):', acc, ' / running time:', t_end-t_start)


    acc_test = sess.run(accuracy, feed_dict={X: x_test, Labels: y_test})
    print('Test accuracy:', acc_test)

    plt.figure('Loss')
    plt.plot(list_loss)
    plt.show()





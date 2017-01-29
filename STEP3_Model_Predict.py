from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import PIL
from PIL import Image
from scipy import misc
import random
import pandas as pd
import tensorflow as tf
import h5py
import math

def predict_with_model(y, model_name):

    test_dataset = y
    #parameters
    batch_size = 16
    image_size = 32
    num_labels = 11
    patch_size = 3
    d1 = 4
    d2 = 8
    d3 = 16
    d4 = 32
    d5 = 64
    d6 = 128
    d7 = 256
    d8 = 512
    sdev = math.sqrt(2.0 / (32**2*1))


    #tensorflow computational graph
    graph = tf.Graph()
    with graph.as_default():

        # Input data.
        tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size, image_size, 1))
        tf_train_labels = tf.placeholder(tf.int64, shape=(batch_size,6))
        tf_eval = tf.placeholder(tf.float32,shape=(batch_size, image_size, image_size, 1))
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.

        conv1_w = tf.Variable(tf.truncated_normal([patch_size, patch_size, 1, d1], stddev=sdev))
        conv1_b = tf.Variable(tf.zeros([d1]))

        conv2_w = tf.Variable(tf.truncated_normal([patch_size, patch_size, d1, d2], stddev=sdev))
        conv2_b = tf.Variable(tf.zeros([d2]))

        conv3_w = tf.Variable(tf.truncated_normal([patch_size, patch_size, d2, d3], stddev=sdev))
        conv3_b = tf.Variable(tf.zeros([d3]))

        conv4_w = tf.Variable(tf.truncated_normal([patch_size, patch_size, d3, d4], stddev=sdev))
        conv4_b = tf.Variable(tf.zeros([d4]))

        conv5_w = tf.Variable(tf.truncated_normal([patch_size, patch_size, d4, d5], stddev=sdev))
        conv5_b = tf.Variable(tf.zeros([d5]))

        conv6_w = tf.Variable(tf.truncated_normal([patch_size, patch_size, d5, d6], stddev=sdev))
        conv6_b = tf.Variable(tf.zeros([d6]))

        conv7_w = tf.Variable(tf.truncated_normal([patch_size, patch_size, d6, d7], stddev=sdev))
        conv7_b = tf.Variable(tf.zeros([d7]))

        conv8_w = tf.Variable(tf.truncated_normal([patch_size, patch_size, d7, d8], stddev=sdev))
        conv8_b = tf.Variable(tf.zeros([d8]))


        fc1_w = tf.Variable(tf.truncated_normal([128,512], stddev=0.1))
        fc1_b = tf.Variable(tf.constant(1.0, shape=[64]))


        #for final logits
        hidden = 512
        s1_w = tf.get_variable('s1_w', shape=[hidden, num_labels],initializer=tf.contrib.layers.xavier_initializer())
        s1_b = tf.Variable(tf.constant(1.0, shape=[num_labels]))
        s2_w = tf.get_variable('s2_w',shape=[hidden, num_labels],initializer=tf.contrib.layers.xavier_initializer())
        s2_b = tf.Variable(tf.constant(1.0, shape=[num_labels]))
        s3_w = tf.get_variable('s3_w',shape=[hidden, num_labels],initializer=tf.contrib.layers.xavier_initializer())
        s3_b = tf.Variable(tf.constant(1.0, shape=[num_labels]))
        s4_w = tf.get_variable('s4_w',shape=[hidden, num_labels],initializer=tf.contrib.layers.xavier_initializer())
        s4_b = tf.Variable(tf.constant(1.0, shape=[num_labels]))
        s5_w = tf.get_variable('s5_w',shape=[hidden, num_labels],initializer=tf.contrib.layers.xavier_initializer())
        s5_b = tf.Variable(tf.constant(1.0, shape=[num_labels]))


        # Model.
        def model(data):
            '''
            Our best architecture consists of eight convolutional hidden layers
            one locally connected hidden layer, and two densely connected hidden layers.
            All connections are feedforward and go from one layer to the next (no skip connections).
            The first hidden layer contains maxout units (Goodfellow et al., 2013) (with three filters per unit)
            while the others contain rectifier units (Jarrett et al., 2009; Glorot et al., 2011). 
            The number of units at each spatial location in each layer is [48, 64, 128, 160] 
            for the first four layers and 192 for all other locally connected layers. 
            The fully connected layers contain 3,072 units each. 
            Each convolutional layer includes max pooling and subtractive normalization. 
            The max pooling window size is 2 × 2. 
            The stride alternates between 2 and 1 at each layer, 
            so that half of the layers don’t reduce the spatial size of the representation. 
            All convolutions use zero padding on the input to preserve representation size. 
            The subtractive normalization operates on 3x3 windows and preserves representation size. 
            All convolution kernels were of size 5 × 5. 
            We trained with dropout applied to all hidden layers but not the input.
            '''
            pad  = 'SAME'
            conv1 = tf.nn.conv2d(data, conv1_w, [1, 1, 1, 1], padding=pad)
            conv1 = tf.nn.relu(conv1 + conv1_b)
            conv1 = tf.nn.max_pool(conv1, [1,2,2,1], [1,2,2,1], padding=pad)
    #         print(conv1.get_shape())
            conv2 = tf.nn.conv2d(conv1, conv2_w, [1, 1, 1, 1], padding=pad)
            conv2 = tf.nn.relu(conv2 + conv2_b)
            conv2 = tf.nn.max_pool(conv2, [1,2,2,1], [1,2,2,1], padding=pad)
    #         print(conv2.get_shape())
            conv3 = tf.nn.conv2d(conv2, conv3_w, [1, 1, 1, 1], padding=pad)
            conv3 = tf.nn.relu(conv3 + conv3_b)
            conv3 = tf.nn.max_pool(conv3, [1,2,2,1], [1,2,2,1], padding=pad)
    #         print(conv3.get_shape())
            conv4 = tf.nn.conv2d(conv3, conv4_w, [1, 1, 1, 1], padding=pad)
            conv4 = tf.nn.relu(conv4 + conv4_b)
            conv4 = tf.nn.max_pool(conv4, [1,2,2,1], [1,2,2,1], padding=pad)
    #         print(conv4.get_shape())
            conv5 = tf.nn.conv2d(conv4, conv5_w, [1, 1, 1, 1], padding=pad)
            conv5 = tf.nn.relu(conv5 + conv5_b)
            conv5 = tf.nn.max_pool(conv5, [1,2,2,1], [1,2,2,1], padding=pad)
    #         print(conv5.get_shape())
            conv6 = tf.nn.conv2d(conv5, conv6_w, [1, 1, 1, 1], padding=pad)
            conv6 = tf.nn.relu(conv6 + conv6_b)
            conv6 = tf.nn.max_pool(conv6, [1,2,2,1], [1,2,2,1], padding=pad)
    #         print(conv6.get_shape())
            conv7 = tf.nn.conv2d(conv6, conv7_w, [1, 1, 1, 1], padding=pad)
            conv7 = tf.nn.relu(conv7 + conv7_b)
            conv7 = tf.nn.max_pool(conv7, [1,2,2,1], [1,2,2,1], padding=pad)
    #         print(conv7.get_shape())
            conv8 = tf.nn.conv2d(conv7, conv8_w, [1, 1, 1, 1], padding=pad)
            conv8 = tf.nn.relu(conv8 + conv8_b)
            conv8 = tf.nn.max_pool(conv8, [1,2,2,1], [1,2,2,1], padding=pad)
    #         print(conv8.get_shape())

            shape = conv8.get_shape().as_list()
            reshape = tf.reshape(conv8, [shape[0], shape[1] * shape[2] * shape[3]])
    #         print(reshape.get_shape())
    #         fc1 = tf.nn.relu(tf.matmul(reshape, fc1_w) + fc1_b)
    #         
            fc1 = reshape
            fc1 = tf.nn.dropout(fc1, keep_prob = 0.8)

            #five classifiers for each digit
            s1 = tf.matmul(fc1, s1_w) + s1_b
            s2 = tf.matmul(fc1, s2_w) + s2_b
            s3 = tf.matmul(fc1, s3_w) + s3_b
            s4 = tf.matmul(fc1, s4_w) + s4_b
            s5 = tf.matmul(fc1, s5_w) + s5_b




            return [s1, s2, s3, s4, s5]


        # Training computation. #no length logit
        [s1, s2, s3, s4, s5] = model(tf_train_dataset)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(s1, tf_train_labels[:,1])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(s2, tf_train_labels[:,2])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(s3, tf_train_labels[:,3])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(s4, tf_train_labels[:,4])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(s5, tf_train_labels[:,5]))

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Optimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss) 

        # Predictions for the training and evaluation data
        train_prediction = tf.pack([tf.nn.softmax(s1),
                          tf.nn.softmax(s2),
                          tf.nn.softmax(s3),
                          tf.nn.softmax(s4),
                          tf.nn.softmax(s5)])

        eval_prediction = tf.pack([tf.nn.softmax(model(tf_eval)[0]),
                                 tf.nn.softmax(model(tf_eval)[1]),
                                 tf.nn.softmax(model(tf_eval)[2]),
                                 tf.nn.softmax(model(tf_eval)[3]),
                                 tf.nn.softmax(model(tf_eval)[4])])

        test_prediction = tf.pack([tf.nn.softmax(model(tf_test_dataset)[0]),
                         tf.nn.softmax(model(tf_test_dataset)[1]),
                         tf.nn.softmax(model(tf_test_dataset)[2]),
                         tf.nn.softmax(model(tf_test_dataset)[3]),
                         tf.nn.softmax(model(tf_test_dataset)[4])])
    with tf.Session(graph=graph) as sess:

        sess.run(tf.initialize_all_variables())
        saver.restore(sess, os.getcwd() + model_name)
        pred = tf.argmax(test_prediction[:,0,:],1).eval()

        return pred

def format_image(path, image_name):
    image_size = 32
    filename = path + '/' + image_name
    img = Image.open(filename)
    img = misc.fromimage(img, flatten = True)
    img = np.array(img)
    img = misc.imresize(img, (image_size,image_size))
    #normalize
    img = img/float(img.max())*255. - (255./2.) #centering image around 0
    img = img.reshape((-1, image_size, image_size, 1)).astype(np.float32)
    return img

def image(path, image_name):
    filename = path + '/' + image_name
    img = Image.open(filename)
    img = misc.fromimage(img, flatten = False)
    img = np.array(img)
    return img

def predict_show(file_name, model_name, show = True):

    path = os.getcwd()
    img = format_image(path, file_name)
    img_big = image(path, file_name)
    label = predict_with_model(img, model_name)
    print(label)

    # Create figure and axes
    fig,ax = plt.subplots(1)
    ax.imshow(img_big)
    if show == True:
        plt.show()    



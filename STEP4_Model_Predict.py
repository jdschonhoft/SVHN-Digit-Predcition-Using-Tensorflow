from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os as os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import PIL
from PIL import Image, ImageFilter
from scipy import misc
import random
import pandas as pd
import tensorflow as tf
import h5py
import math
import matplotlib.patches as patches
from PIL import Image, ImageFilter


def predict_with_model(y, model_name):

    test_dataset = y
    batch_size = 16
    image_size = 64
    num_labels = 11
    patch_size = 3
    d1 = 16
    d2 = 32
    d3 = 64
    d4 = 128
    sdev = math.sqrt(2.0 / (32**2*1))


    #tensorflow computational graph
    graph = tf.Graph()
    with graph.as_default():

        # Input data.
        tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size, image_size, 1))
        tf_train_labels = tf.placeholder(tf.int64, shape=(batch_size,6))
        tf_train_bbox = tf.placeholder(tf.int64, shape=(batch_size,4))
        tf_eval = tf.placeholder(tf.float32,shape=(batch_size, image_size, image_size, 1))

        # Variables.

        conv1_w = tf.Variable(tf.truncated_normal([patch_size, patch_size, 1, d1], stddev=sdev))
        conv1_b = tf.Variable(tf.zeros([d1]))

        conv2_w = tf.Variable(tf.truncated_normal([patch_size, patch_size, d1, d2], stddev=sdev))
        conv2_b = tf.Variable(tf.zeros([d2]))

        conv3_w = tf.Variable(tf.truncated_normal([patch_size, patch_size, d2, d3], stddev=sdev))
        conv3_b = tf.Variable(tf.zeros([d3]))

        conv4_w = tf.Variable(tf.truncated_normal([patch_size, patch_size, d3, d4], stddev=sdev))
        conv4_b = tf.Variable(tf.zeros([d4]))

        fc1_w = tf.Variable(tf.truncated_normal([2048,64], stddev=0.1))
        fc1_b = tf.Variable(tf.constant(1.0, shape=[64]))


        #for final logits
        hidden = 2048
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

    #     #boundboxes
    #     d1_bb = 1
    #     d2_bb = 2
    #     conv1_w_bb = tf.Variable(tf.truncated_normal([patch_size, patch_size, 1, d1_bb], stddev=sdev))
    #     conv1_b_bb = tf.Variable(tf.zeros([d1_bb]))

    #     conv2_w_bb = tf.Variable(tf.truncated_normal([patch_size, patch_size, d1_bb, d2_bb], stddev=sdev))
    #     conv2_b_bb = tf.Variable(tf.zeros([d2_bb]))

    #     fc1_w_bb = tf.Variable(tf.truncated_normal([512,128], stddev=0.1))
    #     fc1_b_bb = tf.Variable(tf.constant(1.0, shape=[64]))


        #for final logits
        hidden2 = 2048
        b1_w = tf.get_variable('b1_w', shape=[hidden2, num_labels],initializer=tf.contrib.layers.xavier_initializer())
        b1_b = tf.Variable(tf.constant(1.0, shape=[num_labels]))
        b2_w = tf.get_variable('b2_w',shape=[hidden2, num_labels],initializer=tf.contrib.layers.xavier_initializer())
        b2_b = tf.Variable(tf.constant(1.0, shape=[num_labels]))
        b3_w = tf.get_variable('b3_w',shape=[hidden2, num_labels],initializer=tf.contrib.layers.xavier_initializer())
        b3_b = tf.Variable(tf.constant(1.0, shape=[num_labels]))
        b4_w = tf.get_variable('b4_w',shape=[hidden2, num_labels],initializer=tf.contrib.layers.xavier_initializer())
        b4_b = tf.Variable(tf.constant(1.0, shape=[num_labels]))



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

            conv1 = tf.nn.conv2d(data, conv1_w, [1, 1, 1, 1], padding='SAME')
            conv1 = tf.nn.relu(conv1 + conv1_b)
            conv1 = tf.nn.max_pool(conv1, [1,2,2,1], [1,2,2,1], padding='SAME')

            conv2 = tf.nn.conv2d(conv1, conv2_w, [1, 1, 1, 1], padding='SAME')
            conv2 = tf.nn.relu(conv2 + conv2_b)
            conv2 = tf.nn.max_pool(conv2, [1,2,2,1], [1,2,2,1], padding='SAME')

            conv3 = tf.nn.conv2d(conv2, conv3_w, [1, 1, 1, 1], padding='SAME')
            conv3 = tf.nn.relu(conv3 + conv3_b)
            conv3 = tf.nn.max_pool(conv3, [1,2,2,1], [1,2,2,1], padding='SAME')

            conv4 = tf.nn.conv2d(conv3, conv4_w, [1, 1, 1, 1], padding='SAME')
            conv4 = tf.nn.relu(conv4 + conv4_b)
            conv4 = tf.nn.max_pool(conv4, [1,2,2,1], [1,2,2,1], padding='SAME')
            shape = conv4.get_shape().as_list()
            reshape = tf.reshape(conv4, [shape[0], shape[1] * shape[2] * shape[3]])  
            fc1 = tf.nn.dropout(reshape, keep_prob = 0.8)


            #bounding box
            shape_bb = conv4.get_shape().as_list()
            reshape_bb = tf.reshape(conv4, [shape_bb[0], shape_bb[1] * shape_bb[2] * shape_bb[3]])  
            fc1_bb = tf.nn.dropout(reshape_bb, keep_prob = 0.8)


            #five classifiers for each digit
            s1 = tf.matmul(fc1, s1_w) + s1_b
            s2 = tf.matmul(fc1, s2_w) + s2_b
            s3 = tf.matmul(fc1, s3_w) + s3_b
            s4 = tf.matmul(fc1, s4_w) + s4_b
            s5 = tf.matmul(fc1, s5_w) + s5_b

            #four classifiers for the bounding box, locating the number in the image
            b1 = tf.matmul(fc1_bb, b1_w) + b1_b
            b2 = tf.matmul(fc1_bb, b2_w) + b2_b
            b3 = tf.matmul(fc1_bb, b3_w) + b3_b
            b4 = tf.matmul(fc1_bb, b4_w) + b4_b

            return [s1, s2, s3, s4, s5, b1, b2, b3, b4]


        # Training computation. #no length logit
        [s1, s2, s3, s4, s5, b1, b2, b3, b4] = model(tf_train_dataset)


        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(s1, tf_train_labels[:,1])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(s2, tf_train_labels[:,2])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(s3, tf_train_labels[:,3])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(s4, tf_train_labels[:,4])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(s5, tf_train_labels[:,5])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(b1, tf_train_bbox[:,0])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(b2, tf_train_bbox[:,1])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(b3, tf_train_bbox[:,2])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(b4, tf_train_bbox[:,3]))



        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Optimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.0005).minimize(loss) 

        # Predictions for the training and evaluation data
        train_prediction = tf.pack([tf.nn.softmax(s1),
                          tf.nn.softmax(s2),
                          tf.nn.softmax(s3),
                          tf.nn.softmax(s4),
                          tf.nn.softmax(s5),
                          tf.nn.softmax(b1),
                          tf.nn.softmax(b2),
                          tf.nn.softmax(b3),
                          tf.nn.softmax(b4)])

        eval_prediction = tf.pack([tf.nn.softmax(model(tf_eval)[0]),
                                 tf.nn.softmax(model(tf_eval)[1]),
                                 tf.nn.softmax(model(tf_eval)[2]),
                                 tf.nn.softmax(model(tf_eval)[3]),
                                 tf.nn.softmax(model(tf_eval)[4]),
                                 tf.nn.softmax(model(tf_eval)[5]),
                                 tf.nn.softmax(model(tf_eval)[6]),
                                 tf.nn.softmax(model(tf_eval)[7]),
                                 tf.nn.softmax(model(tf_eval)[8]),])

        test_prediction = tf.pack([tf.nn.softmax(model(test_dataset)[0]),
                         tf.nn.softmax(model(test_dataset)[1]),
                         tf.nn.softmax(model(test_dataset)[2]),
                         tf.nn.softmax(model(test_dataset)[3]),
                         tf.nn.softmax(model(test_dataset)[4]),
                         tf.nn.softmax(model(test_dataset)[5]),
                         tf.nn.softmax(model(test_dataset)[6]),
                         tf.nn.softmax(model(test_dataset)[7]),
                         tf.nn.softmax(model(test_dataset)[8]),])
    with tf.Session(graph=graph) as sess:

        sess.run(tf.initialize_all_variables())
        saver.restore(sess, os.getcwd() + model_name)
        pred = tf.argmax(test_prediction[:,0,:],1).eval()

        return pred

def format_image(path, image_name):
    image_size = 64
    filename = path + '/' + image_name
    img = Image.open(filename)
    img = Image.filter(ImageFilter.GaussianBlur(radius=7))
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

# path = os.getcwd()
# file_names = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg', 'image5.jpg','image7.png', 'image8.jpg']
# # file_names = ['image7.png']

def predict_show(file_name, model_name, show = True):

    path = os.getcwd()
    img = format_image(path, file_name)
    img_big = image(path, file_name)
    label = predict_with_model(img, model_name)
    print(label[0:5])

    # Create figure and axes
    fig,ax = plt.subplots(1)
    ax.imshow(img_big)
    # Create a Rectangle patch
    y_top = float(label[6])*10./64. * float(img_big.shape[0])
    x_top = float(label[7])*10./64. * float(img_big.shape[1])
    height = float(label[6]-label[5]) *10./64. * img_big.shape[0]
    width = float(label[8]-label[7]) *10./64. * img_big.shape[1]
    rect = patches.Rectangle((x_top,y_top), width, -height, linewidth=3,edgecolor='r',facecolor='none')
    ax.add_patch(rect)

    if show == True:
        plt.show()    


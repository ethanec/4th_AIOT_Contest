# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:06:00 2019

@author: ethan
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random

img_h=img_w=64
img_size_flat=img_h*img_w
n_classes=1
n_channels=3

data = np.load('cat_dog_dataset.npy',allow_pickle=True)
X_train=data.item().get('X_train')
y_train=data.item().get('y_train')
X_test= data.item().get('X_test')
y_test= data.item().get('y_test')
print('X_train',X_train.shape)
print('y_train',y_train.shape)
print('X_test',X_test.shape)
print('X_test',X_test.shape)

logs_path="./logs"
epochs=70
batch_size=100
display_freq=10
lr=0.0002

def conv2d(x, filter_shape, name):
    W = tf.Variable(tf.random_normal(filter_shape))
    b = tf.Variable(tf.random_normal([filter_shape[-1]]))
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def max_pool(x, ksize, stride, name):
    return tf.nn.max_pool(x,ksize=[1, ksize, ksize, 1],strides=[1, stride, stride, 1],padding="SAME",name=name)

def flatten_layer(layer):
    with tf.variable_scope('Flatten_layer'):
        num_features = layer.shape[1:].num_elements()
        layer_flat = tf.reshape(layer,[-1, num_features])
    return layer_flat

with tf.name_scope('Input'):
    X = tf.placeholder(tf.float32, shape=[None, img_h, img_w, n_channels], name='X')
    y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')
    
conv1 = conv2d(X,[3,3,3,32], name="conv1")
conv2 = conv2d(conv1,[3,3,32,64], name="conv2")
pool1 = max_pool(conv2, ksize=2, stride=2, name="pool1")
conv3 = conv2d(pool1,[3,3,64,128], name="conv3")
conv4 = conv2d(conv3,[3,3,128,256], name="conv4")
pool2 = max_pool(conv4, ksize=2, stride=2, name="pool2")


layer_flat = flatten_layer(pool2)
fc1 = tf.layers.dense(layer_flat, units=128, activation=tf.nn.relu)
output_logits = tf.layers.dense(fc1, n_classes)

with tf.variable_scope('Train'):
    with tf.variable_scope('Loss'):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
    tf.summary.scalar('loss', loss)
    with tf.variable_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='Adam-op').minimize(loss)
    with tf.variable_scope('Accuracy'):
        pred = tf.nn.sigmoid(output_logits)
        correct_prediction = tf.equal(tf.round(pred), y, name='correct_pred')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accracy')
    tf.summary.scalar('accuracy',accuracy)
    with tf.variable_scope('Prediction'):
        pred = tf.nn.sigmoid(output_logits)
        cls_prediction = tf.round(pred, name='predictions')

init = tf.global_variables_initializer()
merged = tf.summary.merge_all()

def randomize(x, y):
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :, :, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y
def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch




sess = tf.InteractiveSession()
sess.run(init)
global_step = 0
summary_writer = tf.summary.FileWriter(logs_path, sess.graph)
num_tr_iter = int(len(y_train) / batch_size)
for epoch in range(epochs):
    print('Training epoch:{}'.format(epoch + 1))
    X_train, y_train = randomize(X_train, y_train)
    for iteration in range(num_tr_iter):
        global_step += 1
        start = iteration*batch_size
        end = (iteration + 1)*batch_size
        X_batch, y_batch = get_next_batch(X_train, y_train, start, end)
        
        feed_dict_batch = {X: X_batch, y:y_batch}
        sess.run(optimizer, feed_dict=feed_dict_batch)
        
        if iteration % display_freq ==0:
            loss_batch, acc_batch, summary_tr = sess.run([loss, accuracy, merged],feed_dict=feed_dict_batch)
            summary_writer.add_summary(summary_tr, global_step)
            
            print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".format(iteration, loss_batch, acc_batch))
            
    feed_dict_test = {X: X_test, y: y_test}
    loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
    print('-------------------------------------------------------------')
    print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".format(epoch + 1, loss_test, acc_test ))
    print('-------------------------------------------------------------')
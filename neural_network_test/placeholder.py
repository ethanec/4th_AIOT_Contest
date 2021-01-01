# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:39:00 2019

@author: ethan
"""


import tensorflow as tf
a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)
c= tf.multiply(a,b)
with tf.Session() as sess:
    print(sess.run(c,feed_dict = {a:100,b:200}))
 
    
x1 = tf.placeholder(tf.float32,[2,3])
x2 = tf.placeholder(tf.float32,[3,2])
x3 = tf.matmul(x1,x2)
with tf.Session() as sess:
    print(sess.run(x3,feed_dict = {x1:[[1,2,3],[4,5,6]],x2:[[1,2],[3,4],[5,6]]}))

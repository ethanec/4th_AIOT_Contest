# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 21:38:10 2019

@author: ethan
"""

import cv2
import tensorflow as tf

I = cv2.imread('cathon.jpg')
I1 = cv2.imread('new')

# 池化核 5*5
ksize=[1, 5, 5, 1]
# 步長 = 2
strides=[1, 1, 1, 1]

# 輸入影像格式設定
img_shape = [1, I.shape[0], I.shape[1], I.shape[2]]
#img_shape1 = [1, I1.shape[0], I1.shape[1], I1.shape[2]]

# 池化運算
x = tf.placeholder(tf.float32, shape = img_shape)
pool = tf.nn.max_pool(x, ksize, strides,'SAME')

with tf.Session() as sess:
    I1 = sess.run(pool, feed_dict={x:I.reshape(img_shape)})
    print(I1.shape)
    #I2 = sess.run(pool, feed_dict={x:I1.reshape(img_shape1)})

# 轉回正常影像格式，並存檔
I1 = I1.reshape(I1.shape[1], I1.shape[2], I1.shape[3]) 
print(I1.shape)
cv2.imwrite('new.jpg', I1)
#I2 = I2.reshape(I2.shape[1], I2.shape[2], I2.shape[3])  
#cv2.imwrite('new1.jpg', I2)
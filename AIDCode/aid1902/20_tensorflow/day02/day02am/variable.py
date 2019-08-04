# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf

# Create two variables.
weights = tf.Variable(tf.random_normal(
    [784, 200], stddev=0.35), name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")
# 把全局变量初始化, 返回的init_op将会在sess.run()时执行
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)

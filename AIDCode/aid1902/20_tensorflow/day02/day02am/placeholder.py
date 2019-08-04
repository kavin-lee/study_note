# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import tensorflow as tf

with tf.Session() as sess:
    # 喂一组数据：
    x1 = tf.placeholder(tf.float32, shape=(1, 2))
    x2 = tf.placeholder(tf.float32, shape=(1, 2))
    y = x1 + x2
    r = sess.run(y, feed_dict={x1: [[0.5, 0.6]], x2: [[0.5, 0.6]]})
    print(r)

    x = tf.placeholder(tf.float32, shape=(None, 2))
    y = tf.reduce_sum(x, 0)
    r = sess.run(
        y, feed_dict={x: [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]})
    print(r)

import os

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import mnist_infernece

BATCH_SIZE = 100    #一个训练batch的数据个数
                    #数字越大，越接近梯度下降，数字越小，越接近随机梯度下降

LEARNING_RATE_BASE = 0.8    #基础学习率
LEARNING_RATE_DECAY = 0.99  #学习率衰减率
REGULARIZATION_RATE = 0.0001    #描述模型复杂度的正则化的系数
TRAINING_STEPS = 30000      #训练轮数
MOVING_AVRRAGE_DECAY = 0.99     #滑动平均衰减率

MODEL_SAVE_PATH = "/Users/xjohn/tensorData/model/"
MODEL_NAME = "model.ckpt"

def train(mnist):
    x = tf.placeholder(tf.float32,
                       [BATCH_SIZE,
                        mnist_infernece.IMAGE_SIZE,
                        mnist_infernece.IMAGE_SIZE,
                        mnist_infernece.NUM_CHANNELS],
                       name='x-input')
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, mnist_infernece.OUTPUT_NODE], name = 'y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    y = mnist_infernece.inference(x, True, regularizer)

    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVRRAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    #计算交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_, 1))
    #计算所有batch的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 基础学习率
        global_step,  # 当前的训练轮数
        mnist.train.num_examples / BATCH_SIZE,  # 训练完所有数据需要的轮数
        LEARNING_RATE_DECAY  # 学习率的衰减速度
    )
    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate) \
        .minimize(loss=loss, global_step=global_step)
    # 1.反向传播，2.更新每个参数的滑动平均值
    train_op = tf.group(train_step, variable_averages_op)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                             mnist_infernece.IMAGE_SIZE,
                                             mnist_infernece.IMAGE_SIZE,
                                             mnist_infernece.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:reshaped_xs, y_:ys})

            if i % 1000 == 0:
                print("After %d training step , loss on traing batch is %g" %(step, loss_value))

                saver.save(sess, os.path.join(MODEL_SAVE_PATH,MODEL_NAME), global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("/Users/xjohn/data", one_hot=True)
    train(mnist)
if __name__=='__main__':
    tf.app.run()
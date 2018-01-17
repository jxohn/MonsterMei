import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


import mnist_infernece
import mnist_train

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_infernece.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_infernece.OUTPUT_NODE], name='y-input')

        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        y =mnist_infernece.inference(x, False, None)

        #检测向前传播的结果是否正确
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        #表示一组数据的正确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVRRAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)

                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)

                    print("After %s training step, Validation accuracy = %g" %(global_step, accuracy_score))

                else:
                    print("no file found")
                    return
                time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets("/Users/xjohn/data", one_hot=True)
    evaluate(mnist)
if __name__=='__main__':
    tf.app.run()
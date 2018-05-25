import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import control_flow_ops
import flower_inference
import flower_eval
import flower_read
import os
import numpy as np
# 日志相关参数
SUMMARY_DIR = '/home/qwe/PycharmProjects/Flower_Inception-V3/LOG/'

# training相关参数
batch_size = 40
image_height = 299
image_width = 299
n_classes = 5
learning_rate = 0.001
steps = 10000
num_val_example = 73 * 5

model_save_path = '/home/qwe/PycharmProjects/Flower_Inception-V3/flower_Inception-V3_model'
model_name = 'model.cpkt'

model_deep_save_path = '/home/qwe/PycharmProjects/Flower_Inception-V3/flower_Inception-V3_model_deep'
model_deep_name = 'model_deep.cpkt'

root_file_path = '/media/qwe/085ae2e6-a267-472c-8d94-395de15fb30a/flower_photos_string'
name_list = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
os.chdir(root_file_path)

filename_val_list = os.listdir('./val')
filename_train_list = os.listdir('./train')
for x in range(5):
    filename_train_list[x] = './train/' + filename_train_list[x]
    filename_val_list[x] = './val/' + filename_val_list[x]


def train(input_batch, label_batch, input_batch_val, label_batch_val, load_model,
          input_batch_test, label_batch_test):
    # 定义输入层的输入inputs holder [None X 2048]
    inputs_holder = tf.placeholder(tf.float32,
                                   [None, image_height, image_width, 3],
                                   name='images_input')
    # 定义最终输出层的labels holder [None X 5]
    labels_holder = tf.placeholder(tf.int32,
                                   [None, n_classes],
                                   name='GroundTruthInput')

    logits, end_points = flower_inference.inference(input_holder=inputs_holder, is_training=True)
    prediction = end_points['Predictions']

    with tf.variable_scope('loss'):
        g_step = tf.get_variable('g_step', [],
                                 initializer=tf.constant_initializer(0.0),
                                 trainable=False)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels_holder
        )
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([cross_entropy_mean] + l2_loss)
        # opt = tf.train.AdamOptimizer()
        opt = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
        train_step = slim.learning.create_train_op(loss, opt,
                                                   global_step=g_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            loss = control_flow_ops.with_dependencies([updates], loss)
        # with tf.control_dependencies([tf.group(*update_ops)]):
        #     train_step = tf.train.AdamOptimizer().minimize(cross_entropy_mean)

        # 记录交叉熵
        tf.summary.scalar('loss', loss)

    with tf.variable_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(prediction, 1),
                                      tf.argmax(labels_holder, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 记录准确率
        tf.summary.scalar('accuracy', evaluation_step)

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    with tf.Session() as sess:

        # 可视化信息写入对象_用于训练
        summary_writer_train = tf.summary.FileWriter(SUMMARY_DIR+'train', sess.graph)

        # 可视化信息写入对象_用于验证
        summary_writer_val = tf.summary.FileWriter(SUMMARY_DIR+'val', sess.graph)

        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # # step轮的训练过程
        # ckpt = tf.train.get_checkpoint_state(model_save_path)
        # # 如果模型存在且需要载入
        # if ckpt and load_model:
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        #     print('successfully load model')
        # else:
        #     print('trains the model from zero')

        ckpt_deep = tf.train.get_checkpoint_state(model_deep_save_path)
        # 如果模型存在且需要载入
        if ckpt_deep and load_model:
            saver.restore(sess, ckpt_deep.model_checkpoint_path)
            print('successfully load model: %s' % ckpt_deep.model_checkpoint_path)
        else:
            print('trains the model from zero')
        max_accuracy = 0
        max_test_acc = 0
        for i in range(12250, 2*steps):

            # 获取一个batch的原始图片和标签
            # ndarray: (batch_size, 299*299*3) (batch_size, 5)
            images_train, labels_train = sess.run([input_batch, label_batch])

            # 对images_train进行reshape
            # ndarray: (batch_size, 299, 299, 3) (batch_size, 5)
            images_train = np.reshape(images_train,
                                      (batch_size, image_width, image_height, 3))

            sess.run(train_step, feed_dict={inputs_holder: images_train,
                                            labels_holder: labels_train})

            if i % 50 == 0:
                print('step %d 训练完毕' % i)
                summary = sess.run(merged, feed_dict={inputs_holder: images_train,
                                                      labels_holder: labels_train})
                summary_writer_train.add_summary(summary, i)
                if i % 250 == 0:

                    images_val, labels_val = sess.run([input_batch_val, label_batch_val])
                    images_val = np.reshape(images_val,
                                            (num_val_example, image_width, image_height, 3))

                    # val_accuracy = sess.run(evaluation_step,
                    #                                   feed_dict={inputs_holder: images_val,
                    #                                              labels_holder: labels_val})

                    summary = sess.run(merged, feed_dict={inputs_holder: images_val,
                                                          labels_holder: labels_val})
                    summary_writer_val.add_summary(summary, i)

                    flower_eval.train(input_batch_test, label_batch_test)

                    # 保存模型
                    saver.save(sess, os.path.join(model_deep_save_path, model_deep_name))
                    # if test_acc > max_test_acc:
                    #     max_test_acc = test_acc
                    #     saver.save(sess, os.path.join(model_deep_save_path, model_deep_name))

        coord.request_stop()
        coord.join(threads)
    summary_writer_train.close()
    summary_writer_val.close()


def main(load_model=True):
    input_batch, label_batch = flower_read.read_training(filename_train_list,
                                                         batch_size, n_classes,
                                                         image_width * image_height * 3)

    input_batch_val, label_batch_val = flower_read.read_testing(filename_val_list,
                                                                num_val_example, n_classes,
                                                                image_width * image_height * 3)

    input_batch_test, label_batch_test = flower_read.read_testing(flower_eval.filename_test_list,
                                                                  flower_eval.num_test_example,
                                                                  n_classes,
                                                                  image_width * image_height * 3)

    train(input_batch, label_batch, input_batch_val, label_batch_val, load_model,
          input_batch_test, label_batch_test)


if __name__ == '__main__':
    main()

import tensorflow as tf
import os


def read_training(filename, batch_size, num_classes, bottleneck_tensor_size):

    filename_queue = tf.train.string_input_producer(filename)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
                    'label': tf.FixedLenFeature([num_classes], tf.int64),
                    'img_raw': tf.FixedLenFeature([bottleneck_tensor_size], tf.float32),
                                       })

    # features['img_raw'] 就是一个图片的1-D ndarray (299*299*3,)
    img = features['img_raw']
    label = features['label']

    img_batch, label_batch = tf.train.shuffle_batch(
        [img, label],
        batch_size=batch_size,
        num_threads=2,
        capacity=4000,
        min_after_dequeue=3000
    )
    # img_batch:    tensor: shape=(batch_size, 299*299*3) dtype=float32
    # label_batch:  tensor: shape=(batch_size, 5) dtype=int64
    return img_batch, label_batch


def read_testing(filename, batch_size, num_classes, bottleneck_tensor_size):
    filename_queue = tf.train.string_input_producer(filename)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([num_classes], tf.int64),
            'img_raw': tf.FixedLenFeature([bottleneck_tensor_size], tf.float32),
        })

    # features['img_raw'] 就是一个图片的1-D ndarray (299*299*3,)
    img = features['img_raw']
    label = features['label']

    img_batch, label_batch = tf.train.batch(
        [img, label],
        batch_size=batch_size,
        num_threads=2,
        capacity=2000
    )
    # img_batch:    tensor: shape=(batch_size, 299*299*3) dtype=float32
    # label_batch:  tensor: shape=(batch_size, 5) dtype=int64
    return img_batch, label_batch


# def TEST():
#     root_file_path = '/media/qwe/085ae2e6-a267-472c-8d94-395de15fb30a/flower_photos_string'
#     name_list = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
#     os.chdir(root_file_path)
#     filename_train_list = []
#     for name in name_list:
#         filename_train_list.append('./train_' + name +'.tfrecords')
#     filename_test = ['./test.tfrecords']
#     num_classes = 5
#     SIZE = 299 * 299 * 3
#     BATCH_SIZE = 200
#     image_batch, label_batch = read_training(filename_test, BATCH_SIZE, num_classes, SIZE)
#
#     with tf.Session() as sess:
#
#         sess.run(tf.global_variables_initializer())
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#         x, y = sess.run([image_batch, label_batch])
#         print(x.shape, type(x))
#         print(y.shape, type(y))
#         coord.request_stop()
#         coord.join(threads)
#
#
# TEST()

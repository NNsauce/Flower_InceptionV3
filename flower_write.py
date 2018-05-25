import os
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import multiprocessing
import random


"""daisy: 雏菊--dandelion：蒲公英--rose：玫瑰--sunflower：向日葵--tulip：郁金香"""

root_file_path = '/media/qwe/085ae2e6-a267-472c-8d94-395de15fb30a/flower_photos'
sub_file_path_list = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
os.chdir(root_file_path)

num_classes = 5


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def img_process(img_data):

    img_data = tf.image.random_flip_up_down(img_data)
    img_data = tf.image.random_flip_left_right(img_data)

    img_data = tf.image.central_crop(img_data, 0.5)

    img_data = tf.image.random_brightness(img_data, max_delta=0.5)
    img_data = tf.image.random_contrast(img_data, lower=0, upper=5)
    img_data = tf.image.random_saturation(img_data, 0, 5)

    img_data = tf.image.resize_images(img_data, [299, 299], method=1)

    return img_data


def img_to_ndarray(img_data):
    # 至此,每张图片的img_data都是一个(299, 299, 3)的ndarray
    img_data = img_data.eval()
    # 至此,每张图片的img_data都是一个299x299x3的ndarray
    img_data = np.ravel(img_data, order='C')

    return img_data


def write_fun(index, sub_file_path):

    writer_train = tf.python_io.TFRecordWriter('../flower_photos_string/train/train_' + sub_file_path + '.tfrecords')
    writer_test = tf.python_io.TFRecordWriter('../flower_photos_string/test/test_' + sub_file_path + '.tfrecords')
    writer_val = tf.python_io.TFRecordWriter('../flower_photos_string/val/val_' + sub_file_path + '.tfrecords')
    # 一个花目录下的所有图片路径列表
    image_list = os.listdir(sub_file_path)
    # 将所有图片的顺序打乱
    random.shuffle(image_list)
    # 用于train的图片路径列表 (后146张图片用于test 和 val)
    image_train_list = image_list[:len(image_list) - 146]
    # 用于test的图片路径列表
    image_test_list = image_list[len(image_list) - 146: len(image_list) - 73]
    image_val_list = image_list[len(image_list) - 73:]

    label = np.zeros(num_classes, dtype=np.int64)
    label[index] = 1
    label = label.tolist()

    with tf.Session() as sess:

        for img_name in image_train_list:
            img_path = sub_file_path + '/' + img_name
            img_raw = gfile.FastGFile(img_path, 'rb').read()

            img_data = tf.image.decode_jpeg(img_raw, channels=3)
            img_data = tf.image.resize_images(img_data, [299, 299], method=1)
            img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)

            # 随机决定是否进行图像处理
            pre_flag = random.randint(0, 5)
            if not pre_flag:
                img_data_aux = img_process(img_data)
                img_data_aux = tf.image.per_image_standardization(img_data_aux)
                img_data_aux = img_to_ndarray(img_data_aux)

                example_aux = tf.train.Example(
                    features=tf.train.Features(feature={'label': _int64_feature(label),
                                                        'img_raw': _floats_feature(img_data_aux),
                                                        })
                )
                writer_train.write(example_aux.SerializeToString())

            img_data = tf.image.per_image_standardization(img_data)
            img_data = img_to_ndarray(img_data)

            example = tf.train.Example(
                features=tf.train.Features(feature={'label': _int64_feature(label),
                                                    'img_raw': _floats_feature(img_data),
                                                    })
            )
            writer_train.write(example.SerializeToString())

        writer_train.close()
        print(sub_file_path, 'train 写入完成')

        for img_name in image_test_list:
            img_path = sub_file_path + '/' + img_name
            img_raw = gfile.FastGFile(img_path, 'rb').read()

            img_data = tf.image.decode_jpeg(img_raw, channels=3)
            img_data = tf.image.resize_images(img_data, [299, 299], method=1)
            img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)

            img_data = tf.image.per_image_standardization(img_data)
            img_data = img_to_ndarray(img_data)

            example = tf.train.Example(
                features=tf.train.Features(feature={'label': _int64_feature(label),
                                                    'img_raw': _floats_feature(img_data),
                                                    })
            )
            writer_test.write(example.SerializeToString())
        writer_test.close()
        print(sub_file_path, 'test 写入完成')

        for img_name in image_val_list:
            img_path = sub_file_path + '/' + img_name
            img_raw = gfile.FastGFile(img_path, 'rb').read()

            img_data = tf.image.decode_jpeg(img_raw, channels=3)
            img_data = tf.image.resize_images(img_data, [299, 299], method=1)
            img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)

            # 随机决定是否进行图像处理
            # pre_flag = random.randint(0, 5)
            # if not pre_flag:
            #     img_data_aux = img_process(img_data)
            #     img_data_aux = tf.image.per_image_standardization(img_data_aux)
            #     img_data_aux = img_to_ndarray(img_data_aux)
            #
            #     example_aux = tf.train.Example(
            #         features=tf.train.Features(feature={'label': _int64_feature(label),
            #                                             'img_raw': _floats_feature(img_data_aux),
            #                                             })
            #     )
            #     writer_val.write(example_aux.SerializeToString())

            img_data = tf.image.per_image_standardization(img_data)
            img_data = img_to_ndarray(img_data)

            example = tf.train.Example(
                features=tf.train.Features(feature={'label': _int64_feature(label),
                                                    'img_raw': _floats_feature(img_data),
                                                    })
            )
            writer_val.write(example.SerializeToString())
        writer_val.close()

        print(sub_file_path, 'val 写入完成')


def main():

    for index, sub_file_path in enumerate(sub_file_path_list):
        if index == 4:
            t = multiprocessing.Process(target=write_fun, args=(index, sub_file_path))
            t.start()


if __name__ == '__main__':
    main()

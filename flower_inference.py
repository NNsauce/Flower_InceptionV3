
import tensorflow.contrib.slim as slim
import tensorflow as tf

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def inception_v3_arg_scope(weight_decay=0.00001, stddev=0.1,
                           batch_norm_var_collection='moving_var'):
    """用来生成NN中经常用到的函数默认参数
       weight_decay :L2正则损失的lambda系数
       stddev :NN参数初始化标准正太分布的标准差"""

    batch_norm_params = {
        'decay': 0.9,
        'epsilon': 0.001,
        
    }

    # 利用slim.arg_scope对卷积层和全连接层赋予缺省参数'weights_regularizer'
    # 默认值为'slim.l2_regularizer(weight_decay)'   (正则损失)
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        # 嵌套一个scope单独对卷积层的缺省参数进行设置
        # 权重初始化值, 激活函数类型, 标准化器, 标准化器参数
        #
        with slim.arg_scope(
            [slim.conv2d],
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params
        ) as sc:
            return sc   # 返回定义好相关缺省参数的scope


def inception_v3_base(inputs, scope=None):
    """定义网络的卷积部分, 接受图片tensor的输入和带有缺省参数的scope"""

    end_points = {}    # 用来保存关键节点的字典

    # 把整个网络结构的变量scope命名为'InceptionV3',
    # 且inputs输入单独设置为非'InceptionV3'下的变量
    with tf.variable_scope(scope, 'InceptionV3', [inputs]):

        # 对卷积层 最大池化层 平均池化层设置默认参数： 步长为1 不填充
        # 即在当前的slim.arg_scope肚子里, 上述三个层均默认步长为1且不填充
        # 不会干扰到其他slim.arg_scope肚子里的函数
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='VALID'):
            net = slim.conv2d(inputs, 32, [3, 3], stride=2, scope='Conv2d_1a_3x3')
            net = slim.conv2d(net, 32, [3, 3], scope='Conv2d_2a_3x3')
            net = slim.conv2d(net, 64, [3, 3], padding='SAME', scope='Conv2d_2b_3x3')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_3a_3x3')
            net = slim.conv2d(net, 80, [1, 1], scope='Conv2d_3b_1x1')
            net = slim.conv2d(net, 192, [3, 3], scope='Conv2d_4a_3x3')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_5a_3x3')
        # 图片tensor(299,299,3)经历5个卷积层 2个最大池化层后变成net tensor(35, 35, 192)

        # 设置三个模块组(5系6系7系)的卷积层 最大池化层 平均池化层默认步长为1且全0填充
        # 因此如果以下的层没有修改步长和填充,无论卷积尺寸和池化尺寸怎么改都不会影响
        # 输出tensor的尺寸
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):

            # 第一个5系模块：Mixed_5b
            with tf.variable_scope('Mixed_5b'):
                # 分支0: 输出tensor(35, 35, 64)
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                # 分支1: 输出tensor(35, 35, 64)
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
                # 分支2: 输出tensor(35, 35, 96)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                # 分支3: 输出tensor(35, 35, 32)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
                # sum(64, 64, 96, 32)
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            # 4个分支输出合并：(35, 35, 256)

            # 第二个5系模块：Mixed_5c
            with tf.variable_scope('Mixed_5c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv_1_0c_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')

                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                # sum(64, 64, 96, 64)
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            # 4个分支输出合并：(35, 35, 288)


            # 第一个6系模块：Mixed_6a
            with tf.variable_scope('Mixed_6a'):
                # 分支0: 输出tensor(17, 17, 384)
                # 修改的步长为2且不填充,所以尺寸缩小
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 384, [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_1x1')
                # 分支1: 输出tensor(17, 17, 96)
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2, padding='VALID',
                                           scope='Conv2d_1a_1x1')
                # 分支3: 输出tensor(17, 17, 288)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                               scope='Maxpool_1a_3x3')
                # sum(384, 96, 288)
                net = tf.concat([branch_0, branch_1, branch_2], 3)
            # 3个分支输出合并：(17, 17, 768)

            # 第二个6系模块：Mixed_6b
            with tf.variable_scope('Mixed_6b'):
                # 分支0: 输出tensor(17, 17, 192)
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')

                # 分支1: 输出tensor(17, 17, 192)
                # 后尺寸分别为1x7和7x1的两个卷积层串联相当于一个7x7的卷积层,
                # (参数减少为2/7,多了一个激活函数)
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                # 分支2: 输出tensor(17, 17, 192)
                # same like 分支1：有两对1x7和7x1的两个卷积层串联
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                # 分支3: 输出tensor(17, 17, 192)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')

                # sum(192, 192, 192, 192)
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            # 4个分支输出合并：(17, 17, 768)

            
            # 把经过2个6系模块后的输出存在字典中, size:(17, 17, 768)
            end_points['Mixed_6e'] = net

            # 第一个7系模块：Mixed_7a
            with tf.variable_scope('Mixed_7a'):

                # 分支0: 输出tensor(8, 8, 320)
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                # 分支1: 输出tensor(8, 8, 192)
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2, padding='VALID',
                                           scope='Conv2d_1a_3x3')
                # 分支2: 输出tensor(8, 8, 768)
                # 池化层不改变通道数
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                # sum(320, 192, 768)
                net = tf.concat([branch_0, branch_1, branch_2], 3)
            # 3个分支输出合并：(8, 8, 1280)


            return net, end_points
        # 返回结果： (8, 8, 2048)tensor, 存放6系模块组输出(17, 17, 768)tensor的字典


def inception_v3(inputs,
                 num_classes=5,
                 is_training=True,
                 dropout_keep_prob=0.5,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=tf.AUTO_REUSE,
                 scope='InceptionV3'):
    """
        定义网络的最后一部分

        inputs:             输入tensor

        num_classes:        千分类

        is_training:        是否处于训练过程的标志,只有为True时,
                            批量归一化(Batch Normalization)和
                            随机失活(Dropout)才会启用

        dropout_keep_prob:  训练过程中随机失活保留节点比例,
                            默认0.8

        prediction_fn：     分类函数选择,默认使用slim.softmax

        spatial_squeeze:    降维标志,去除维数为1的维度
                            eg:(5, 3, 1)变成(5, 3)

        reuse：             网络结构和变量重复使用标志

        scope：             包含函数缺省参数的环境
    """

    # 定义网络 创建命名空间 名称为'InceptionV3', 把inputs, num_classes
    # 设置为非'InceptionV3'下的变量, reuse属性为None(会继承父级scope的reuse值)
    with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes],
                           reuse=reuse) as scope:       # scope值为'InceptionV3'

        # 定义批量归一化和随机失活的缺省参数is_training,若为False
        # 即在这个arg_scope下默认slim.batch_norm和slim.dropout函数
        # 不起作用,输入什么返回什么
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            # 传入inputs和scope参数,调用inception_v3_base,
            # 获得输出： (8, 8, 2048)tensor, 存放6系模块输出(17, 17, 768)tensor的字典
            net, end_points = inception_v3_base(inputs, scope=scope)

            # 把卷积层 最大池化层 平均池化层的步长和padding默认设为1和SAME
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
                # 6系模块组输出(17, 17, 768)tensor 作为辅助分类结果
                aux_logits = end_points['Mixed_6e']
                # 在6系模块组后开个叉,再经过一系列的卷积和池化
                with tf.variable_scope('Auxlogits'):

                    # 平均池化,卷积后： (5, 5, 768)
                    aux_logits = slim.avg_pool2d(
                        aux_logits, [5, 5], stride=3, padding='VALID',
                        scope='AvgPool_1a_5x5'
                    )
                    aux_logits = slim.conv2d(aux_logits, 128, [1, 1],
                                             scope='Conv2d_1b_1x1')
                    # 权重初始化的标准差不使用默认值变为0.01, padding变为VALID
                    # 5x5卷积后： (1, 1, 768)
                    aux_logits = slim.conv2d(aux_logits, 768, [5, 5],
                                             weights_initializer=trunc_normal(0.01),
                                             padding='VALID', scope='Conv2d_2a_5x5')
                    # 不使用激活函数和批量归一化
                    # 权重初始化的标准差不使用默认值变为0.001
                    # 1x1卷积后： (1, 1, 1000)
                    aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1],
                                             activation_fn=None,
                                             normalizer_fn=None,
                                             weights_initializer=trunc_normal(0.001),
                                             scope='Conv2d_2b_1x1')
                    # 降维： (1, 1, 1000)变成(1000,)
                    if spatial_squeeze:
                        aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')

                # 把(1000,)的aux_logits tensor存在字典中
                end_points['AuxLogits'] = aux_logits

            # 对7系模块组后的输出net (8, 8, 2048)tensor继续处理, 来获得最终分类结果
            with tf.variable_scope('Logits'):
                # net首先经过平均池化: (1, 1, 2048)
                net = slim.avg_pool2d(net, [8, 8], padding='VALID',
                                      scope='AvgPool_1a_8x8')
                # 对紧接着上一层的2048个节点进行随机失活(以一个概率让某些节点的值置零)
                # 默认0.8的概率
                # 随机失活后尺寸不会发生任何变化
                net = slim.dropout(net, keep_prob=dropout_keep_prob,
                                   scope='Dropout_1b')

                # 把随机失活后的tensor作为预分类结果 (1, 1, 2048)储存在字典中
                end_points['Prelogits'] = net

                # 最后这一层不使用激活函数和批量归一化
                # 输出最终的分类结果： logits (1, 1, 1000)
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                     normalizer_fn=None, scope='Conv2d_1c_1x1')

                # 同理对logits降维, (1, 1, 1000)变成(1000,)
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

            # 把(1000,)的logits存在字典中
            end_points['Logits'] = logits
            # 同时把logits通过softmax分类器压缩成一个符合要求预测结果,
            # 一个(1000,)的概率tensor, 和为1, 并把这个预测结果存在字典中
            end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

            # 至此, end_points这个字典应该是这个鬼样子：
            # {
            #  'Mixed_6e':    (17, 17, 768)tensor
            #  'AuxLogits':   (1000,)tensor
            #  'PreLogits':   (1, 1, 2048)tensor
            #  'Logits':      (1000,)tensor
            #  'Predictions': (1000,)的概率tensor
            # }

    # 全部网络构建完毕, 返回分类结果 和 储存各个关键节点结果的字典
    return logits, end_points


def inference(input_holder, is_training):
    with slim.arg_scope(inception_v3_arg_scope()):
        logits, end_points = inception_v3(input_holder, is_training=is_training)
        return logits, end_points




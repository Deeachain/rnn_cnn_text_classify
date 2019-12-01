# coding: utf-8

import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64  # 词向量维度,相当于图片的3通道数
    seq_length = 600  # 序列长度，句子的长度
    num_classes = 2  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 6  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 256  # 全连接层神经元

    dropout_keep_prob = 0.45  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 256  # 每批训练大小
    num_epochs = 50  # 总迭代轮次

    print_per_batch = 50  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])  # embedding shape is [5000,64]，词汇表中有5000个词汇
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)  # 函数的用法主要是选取embedding里面索引(self.input_x)对应的元素
                # self.input的shape(?, 600),self.input是长度为600的索引值，返回shape(batch, 600, 64)：batch+句子长度+词向量维度
                # 首先随机生成词汇表大小的张量，将输入的词汇(onehot编码过的)用索引映射到随机生成的词汇表张量每行数据，输入一一对应变成了词汇表的映射
        with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')  # 600-5+1=596,(?, 596, 256)
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')  # (?, 256),对行的维度增长的方向(向下)axis=1整个列求最大值

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')  # shape=(?, 10)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 归一化到[0,1],预测类别 shape=(?,)

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)  # 比较pred和label是否相等，计算准确率
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
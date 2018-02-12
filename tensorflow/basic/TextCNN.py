import tensorflow as tf
import numpy as np
import re
import sys

class TextCNN():
    def __init__(self, debug=False):
        self.is_debug=debug
        self.filter_span = 5
        self.emb_size = 10
        self.seq_length = 10
        self.number_of_channels=1
        self.number_of_classes = 1
        self.number_of_filters = 2
        self.vocab_size = 10
        self.dropout = 0.2

        return

    def create_model(self, filter_sizes):
        self.x = tf.placeholder(tf.int32, shape=[None, self.seq_length, self.number_of_channels])
        self.y = tf.placeholder(tf.int32, shape=[None, self.number_of_classes])

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        # read embeddings
        embedding_matrix = tf.get_variable("W_emb",
                                           shape=[self.vocab_size, self.emb_size],
                                           initializer=tf.initializers.constant(value=1.0) if self.is_debug
                                           else tf.initializers.random_normal(stddev=0.1),
                                           dtype=tf.float32)

        emb_x = tf.nn.embedding_lookup(embedding_matrix, self.x)
        emb_x = tf.transpose(emb_x, perm=[0, 1, 3, 2])
        self.emb_x = emb_x

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-{}".format(filter_size)), tf.name_scope("conv-maxpool-{}".format(filter_size)):
                # Create convolution layer
                filter_shape = [filter_size, self.emb_size, self.number_of_channels, self.number_of_filters]

                W = tf.get_variable("W",
                                    shape=filter_shape,
                                    initializer=tf.initializers.constant(value=1.0) if self.is_debug
                                    else tf.initializers.random_normal(stddev=0.1))

                conv_result = tf.nn.conv2d(emb_x,
                                           W,
                                           strides=[1, 1, 1, 1],
                                           padding="VALID",
                                           name="conv")
                # Apply activation
                b = tf.get_variable("b",
                                    shape=[self.number_of_filters],
                                    initializer=tf.initializers.constant(value=1.0) if self.is_debug
                                    else tf.initializers.random_normal(stddev=0.1))

                h = tf.nn.relu(tf.nn.bias_add(conv_result, b))

                # Max pooling
                with tf.name_scope("maxpool-{}".format(filter_size)):
                    pooled_result = tf.nn.max_pool(h,
                                                   ksize=[1, self.seq_length - filter_size + 1, 1, 1],
                                                   strides=[1, 1, 1, 1],
                                                   padding="VALID")
                    pooled_outputs.append(pooled_result)


        # Create a fully connected node
        self.pooled_output = tf.concat(pooled_outputs, axis=-1)
        pooled_flat = tf.reshape(self.pooled_output, shape=[-1,  self.pooled_output.shape[-1]])

        # Add Dropout
        with tf.name_scope("dropout"):
            drop = tf.nn.dropout(pooled_flat, self.dropout)

        # Add fully connected layer
        with tf.name_scope("score_computation"):
            wo = tf.get_variable(
                "W_output",
                shape=[self.number_of_filters * len(filter_sizes), self.number_of_classes],
                initializer=tf.initializers.constant(value=1.0) if self.is_debug
                else tf.contrib.layers.xavier_initializer())
            bo = tf.get_variable(
                "B_output",
                shape=[self.number_of_classes],
                initializer=tf.initializers.constant(value=1.0) if self.is_debug
                else tf.initializers.constant(value=0.1))

            score = tf.nn.xw_plus_b(drop, wo, bo)

            self.predict = tf.argmax(score, axis=-1)

        with tf.name_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predict, tf.argmax(self.y, -1)), dtype=tf.float32))

        # Compute softmax loss
        with tf.name_scope("loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=score)
            self.loss = tf.reduce_mean(loss)

        with tf.name_scope("gradient"):
            optimiser = tf.train.AdamOptimizer()
            grad = optimiser.compute_gradients(self.loss)
            self.train_op = optimiser.apply_gradients(grad, global_step=self.global_step)

        self.init = tf.global_variables_initializer()

    def run_test(self):
        input_data = np.ones([1, self.seq_length, self.number_of_channels], dtype=np.float32)
        input_label = np.ones([1, self.number_of_classes], dtype=np.float32)
        self.run(input_data, input_label)

    def run(self, input_data, input_label):
        with tf.Session() as session:
            session.run(self.init)

            _,accuracy = session.run([self.train_op, self.accuracy], feed_dict={self.x: input_data, self.y: input_label})
            print(accuracy)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def prepare_data(in_file):
    from tensorflow.contrib import  learn

    max_len = -1
    messages, labels = [], []
    with open(in_file, "r", encoding="ascii", errors="surrogateescape") as f_in:
        for line in f_in:
            fields = re.split("\t", line.strip())
            message = fields[1]
            field_len = len(re.split("\s+", message))
            if  max_len <= field_len:
                max_len = field_len

            messages.append(message)
            #labels.append(int.parse(fields[1]))
            labels.append(0)

        vocab = learn.preprocessing.VocabularyProcessor(max_len, min_frequency=5)
        x = np.array(list(vocab.fit_transform(messages)))
        y = labels

        return (x, y)


if __name__ == "__main__":
#    cnn = TextCNN(debug=False)
#    cnn.create_model([5, 3])
#    cnn.run_test()
    (x,y) = prepare_data(r"C:\Code\Git\ML\tensorflow\basic\test.tsv")
    for item in batch_iter(list(zip(x,y)), 10, 1):
        print (item[0])
        print (item[1])
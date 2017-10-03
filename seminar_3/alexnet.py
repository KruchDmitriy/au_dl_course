import tensorflow as tf
import os
import numpy as np


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

# the 'two eyes' idea is implemented here via 'groups' in convolution
# if the number of groups is greater than one
# the input feature maps and current filters divides into groups
# that computes independently

# e.g. 2 groups -> X split on X[0] and X[1], filters -- filter[0] and filter[1]
# result = concat(conv(X[0], filter[0]), conv(X[1], filter[1]))

def conv(x, kernel_height, kernel_width, num_filters,
         stride_x, stride_y, name, padding='SAME', groups=1):

    input_channels = x.get_shape().as_list()[-1]

    convolve = lambda input_, filter_: \
        tf.nn.conv2d(input_, filter_, [1, stride_x, stride_y, 1], padding)

    with tf.name_scope(name):
        W = weight_variable(shape=[kernel_height, kernel_width,
                               input_channels // groups, num_filters], name='W')
        b = weight_variable(shape=[num_filters], name='b')

        if groups == 1:
            conv = convolve(x, W)
        else:
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            w_groups = tf.split(axis=3, num_or_size_splits=groups, value=W)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, w_groups)]

            conv = tf.concat(axis=3, values=output_groups)

        bias = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape().as_list())

        relu = tf.nn.relu(bias, 'relu')

        return relu

def fc(x, n_out, name, activ=tf.nn.relu):
    with tf.name_scope(name):
        return tf.layers.dense(x, n_out, activation=activ)

def max_pool(x, kernel_height, kernel_width, stride_x, stride_y, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, kernel_height, kernel_width, 1],
                         strides=[1, stride_x, stride_y, 1],
                         padding=padding, name=name)

def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha,
                                             beta=beta, bias=bias, name=name)

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


class AlexNet:
    def __init__(self, preproc):
        self.n_classes = 10
        self.keep_prob = 0.5
        self.preproc = preproc

        self.img_rows = 32
        self.img_cols = 32

        self.batch_size = 32
        self.X = tf.placeholder(shape=[self.batch_size, self.img_rows, self.img_cols, 3],
            dtype=tf.float32, name='X')

        self.X_preproc = self.preproc(self.X, training=True)

        self.y_true = tf.placeholder(shape=[self.batch_size, self.n_classes],
            dtype=tf.int64, name='y_true')
        self.y_true_cls = tf.argmax(self.y_true, dimension=1)

        self.create_model()

        y_pred = tf.nn.softmax(self.logits)
        self.y_pred_cls = tf.argmax(y_pred, dimension=1)

        self.create_loss()
        self.create_optimizer()
        self.create_saver()

    def create_model(self):
        # 1st Layer: Conv (with ReLu) -> Pool -> Lrn
        conv1 = conv(self.X_preproc, 7, 7, 64, 1, 1, name = 'conv1')
        # pool1 = max_pool(conv1, 3, 3, 2, 2, padding = 'VALID', name = 'pool1')
        norm1 = lrn(conv1, 2, 2e-05, 0.75, name = 'norm1')

        # reminder: if number of groups > 1, then computations run independently
        # 2nd Layer: Conv (with ReLu) -> Pool -> Lrn with 2 groups
        conv2 = conv(norm1, 5, 5, 96, 1, 1, groups = 2, name = 'conv2')
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding = 'VALID', name ='pool2')
        norm2 = lrn(pool2, 2, 2e-05, 0.75, name = 'norm2')

        # 3rd Layer: Conv (with ReLu)
        conv3 = conv(norm2, 3, 3, 192, 1, 1, name = 'conv3')

        # 4th Layer: Conv (with ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 192, 1, 1, groups = 2, name = 'conv4')

        # 5th Layer: Conv (with ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 128, 1, 1, groups = 2, name = 'conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding = 'VALID', name = 'pool5')

        # 6th Layer: Flatten -> FC (with ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 5*5*128])
        fc6 = fc(flattened, 256, name='fc6')
        dropout6 = dropout(fc6, self.keep_prob)

        # 7th Layer: FC (with ReLu) -> Dropout
        fc7 = fc(dropout6, 256, name = 'fc7')
        dropout7 = dropout(fc7, self.keep_prob)

        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        fc8 = fc(dropout7, self.n_classes, activ = None, name='fc8')

        self.logits = fc8

    def create_loss(self):
        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y_true, logits = self.logits))

    def create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(1e-4) \
                .minimize(self.cross_entropy, global_step=self.global_step)

    def create_saver(self):
        self.saver = tf.train.Saver()

        self.save_dir = 'checkpoints/'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.save_path = os.path.join(self.save_dir, 'cifar10_alexnet')

    def load_model(self, session):
        try:
            print("Trying to restore last checkpoint ...")
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=self.save_dir)
            self.saver.restore(session, save_path=last_chk_path)
            print("Restored checkpoint from:", last_chk_path)
        except:
            print("Failed to restore checkpoint. Initializing variablies.")
            session.run(tf.global_variables_initializer())

    def accuracy(self):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)
            return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, num_iterations, batch_generator,
                images_test, labels_test):
        self.X_preproc = self.preproc(self.X, training=True)

        with tf.Session() as session:
            self.load_model(session)
            accuracy = self.accuracy()

            tf.summary.FileWriter('graphs', session.graph)

            for i in range(num_iterations):
                x_batch, y_true_batch = batch_generator()

                feed_dict_train = {self.X: x_batch,
                                   self.y_true: y_true_batch}

                i_global, _ = session.run([self.global_step, self.optimizer],
                                          feed_dict=feed_dict_train)

                if (i_global % 100 == 0) or (i == num_iterations - 1):
                    batch_acc = session.run(accuracy,
                                            feed_dict=feed_dict_train)
                    msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
                    print(msg.format(i_global, batch_acc))

                if (i_global % 1000 == 0) or (i == num_iterations - 1):
                    self.saver.save(session,
                               save_path=self.save_path,
                               global_step=self.global_step)

                    print("Saved checkpoint.")

                    test_acc = self.test_in_session(images_test, labels_test, session)[1]
                    print("Test accuracy: ", test_acc)

    def test_in_session(self, images, labels, session):
        n_images = len(images)
        cls_pred = np.zeros(shape=n_images, dtype=np.int)

        for i in range(0, n_images, self.batch_size):
            if i + self.batch_size > len(images):
                i = len(images) - self.batch_size

            x_batch = images[i: i + self.batch_size]

            feed_dict = {self.X: x_batch}
            cls_pred[i: i + self.batch_size] = session.run(self.y_pred_cls, feed_dict)

        cls_true = np.argmax(labels, axis=1)
        correct = (cls_true == cls_pred)

        return cls_pred, np.mean(correct)

    def test(self, images, labels):
        self.X_preproc = self.preproc(self.X, training=False)

        with tf.Session() as session:
            self.load_model(session)

            return test_in_session(images, labels, session)

def random_batch():
    X = np.zeros((32, 32, 32, 3))
    y = np.zeros(shape=(32, 10))
    return X, y

def preproc(X, training):
    with tf.name_scope('preprocessing'):
        return tf.map_fn(lambda image: tf.random_crop(image, [24, 24, 3]), X)

if __name__ == "__main__":
    # simple test
    model = AlexNet(preproc)
    model.train(2, random_batch, None, None)

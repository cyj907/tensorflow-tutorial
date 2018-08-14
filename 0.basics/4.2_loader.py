import tensorflow as tf
import tensorflow.contrib.layers as Layers
import glob
import os

flags = tf.app.flags
flags.DEFINE_float('learning_rate', 1e-6, 'learning rate.')
flags.DEFINE_integer('epoch', 10, 'epoch to train model')
flags.DEFINE_integer('batch_size', 5, 'batch_size')
flags.DEFINE_integer('log_step', 50, 'logging step on screen')
flags.DEFINE_integer('save_step', 50, 'saving step for model')
flags.DEFINE_string('save_path', 'ckpt', 'location to store models')
flags.DEFINE_string('tfrecord_dir', 'tfrecords', 'location to store tfrecords')
FLAGS = flags.FLAGS

def _parse_function(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value="")}
    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.cast(tf.decode_raw(parsed_features['image'], tf.uint8), tf.float32)
    image = tf.reshape(image, [32,32,3])
    image = image / 255.0
    return image

def load_dataset():
    filenames = glob.glob(FLAGS.tfrecord_dir + '/*')
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)       # Parse the record into tensors.
    dataset = dataset.shuffle(buffer_size=10000) # random shuffle data
    dataset = dataset.repeat(FLAGS.epoch)  # Repeat the input for number of epochs
    dataset = dataset.batch(FLAGS.batch_size)
    iterator = dataset.make_one_shot_iterator() # can be used together with tf.train.MonitoredTrainingSession
    return iterator.get_next()



def encoder(inputs):
    # 32x32x3 => 16x16x8
    h1 = Layers.conv2d(inputs, 8, 5, 2)
    # 16x16x8 => 8x8x16
    h2 = Layers.conv2d(h1, 16, 5, 2)
    # 8x8x16 => 4x4x16
    h3 = Layers.conv2d(h2, 16, 5, 2)
    # 4x4x16 => 256
    h3_flattened = tf.reshape(h3, [-1,256])
    # 256 => 2000
    h4 = Layers.fully_connected(h3_flattened,2000)
    # 2000 => 64
    h5 = Layers.fully_connected(h4, 64,activation_fn=None)
    return h5

def decoder(inputs):
    # 64 => 2000
    h1 = Layers.fully_connected(inputs, 2000)
    # 2000 => 256
    h2 = Layers.fully_connected(h1, 256)
    # 256 => 4x4x16
    h2 = tf.reshape(h2, [-1,4,4,16])
    # 4x4x16 => 8x8x16
    h3 = Layers.conv2d_transpose(h2, 16, 5, 2)
    # 8x8x16 => 16x16x8
    h4 = Layers.conv2d_transpose(h3, 8, 5, 2)
    # 16x16x8 => 32x32x3
    h5 = Layers.conv2d_transpose(h4, 3, 5, 2, activation_fn=tf.sigmoid)
    return h5

def network(inputs):
    code = encoder(inputs)
    outputs = decoder(code)
    return outputs

def main(_):
    # load data
    inputs = load_dataset()

    # build the network
    outputs = network(inputs)
    # compute the loss function.
    loss = tf.reduce_mean(tf.square(inputs - outputs))

    # optimizer
    global_step = tf.Variable(0, name='global_step', trainable=False) # record the current training step
    optim = tf.train.AdamOptimizer(FLAGS.learning_rate)
    train_op = optim.minimize(loss, global_step=global_step) # auto increment global_step every time the minimizer is called

    # create a session
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # train model for a number of epochs
        try:
            while True:
                _, cur_step = sess.run([train_op, global_step])

                if cur_step % FLAGS.log_step == 0:
                    loss_value = sess.run(loss)
                    print(loss_value) # print loss
        
                if cur_step % FLAGS.save_step == 0:
                    # save model
                    saver.save(sess, os.path.join(FLAGS.save_path,'model'),global_step=global_step)

        except tf.errors.OutOfRangeError:
            print('Training Finish')


if __name__ == '__main__':
    tf.app.run()
import tensorflow as tf
import skimage
import skimage.io
import skimage.transform
import glob
import os
import numpy as np
import threading

flags = tf.app.flags
flags.DEFINE_string('image_dir', 'raw_images', 'directory to save images')
flags.DEFINE_string('tfrecord_dir', 'tfrecords', 'directory to save tfrecords')
flags.DEFINE_integer('chunks', 5, 'number of chunks to separate the images')
FLAGS = flags.FLAGS

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfrecords(coord, filename, image_files):
    with tf.python_io.TFRecordWriter(filename) as writer:
        i = 0
        max_i = len(image_files)
        while not coord.should_stop():
            img = skimage.io.imread(image_files[i]) # bytes?
            img = process_image(img)
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': _bytes_feature(img.tobytes()),
            }))
            writer.write(example.SerializeToString())
            i += 1
            if i >= max_i:
                coord.request_stop()

def process_image(img):
    # just a simple example
    img = skimage.transform.resize(img, [32, 32, 3])
    # after transformation, the image is ranged (0,1)
    img = img * 255
    # NOTE: remember to convert float data back to uint8
    return img.astype(np.uint8)

def main(_):
    # get filename queue
    image_files = glob.glob(FLAGS.image_dir + '/*.jpg')
    num_files = len(image_files)

    # instantiate images
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        if os.path.exists(FLAGS.tfrecord_dir) is False:
            os.makedirs(FLAGS.tfrecord_dir)

        coord = tf.train.Coordinator()
        threads = []
        for i in range(FLAGS.chunks):
            filename = os.path.join(FLAGS.tfrecord_dir, str(i) + '.tfrecords')
            st = int(float(num_files)/FLAGS.chunks*i)
            en = int(float(num_files)/FLAGS.chunks*(i+1))
            if i == FLAGS.chunks - 1:
                en = num_files
            thread = threading.Thread(target=write_tfrecords,
                        args=(coord,filename,image_files[st:en]))
            threads.append(thread)
        
        for t in threads:
            t.start()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
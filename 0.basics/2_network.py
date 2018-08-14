import tensorflow as tf
import tensorflow.contrib.layers as Layers # my favorite. simple and clean.

def simple_conv_net(inputs):
    h1 = Layers.conv2d(
            inputs=inputs,
            num_outputs=8,
            kernel_size=5,
            stride=2,
            padding='SAME',
            activation_fn=tf.nn.relu,
            weights_initializer=Layers.xavier_initializer(),
            weights_regularizer=None,
            biases_initializer=tf.zeros_initializer(),
            biases_regularizer=None)

    # a simpler way
    h2 = Layers.conv2d(h1, 16, 5, 2)
    return h2

def simple_fc(inputs):
    h1 = Layers.fully_connected(
        inputs=inputs,
        num_outputs=2000,
        activation_fn=tf.nn.relu,
        weights_initializer=Layers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=tf.zeros_initializer(),
        biases_regularizer=None)
    # a simpler way
    h2 = Layers.fully_connected(h1, 64,activation_fn=None)
    return h2


def network(inputs):
    batch_size = inputs.get_shape().as_list()[0]
    conv_out = simple_conv_net(inputs)
    conv_out_flatted = tf.reshape(conv_out, [batch_size, -1])
    fc_out = simple_fc(conv_out_flatted)
    return fc_out

def run():
    # a fake image for simple demonstration
    inputs = tf.ones([10, 32, 32, 3],dtype=tf.float32)
    # build the network
    network_out = network(inputs)

    # create a session
    with tf.Session() as sess:
        # remember to initialize all variables
        # all weights and biases in your network are variables
        # before we compute the values, we have to initialize them, right
        sess.run(tf.global_variables_initializer())
        # compute the network output values
        out_ = sess.run(network_out)
        print(out_)
        print(out_.shape) # (10, 64)

if __name__ == '__main__':
    run()
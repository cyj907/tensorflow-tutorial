""" this contains some snippets to use variable scope """
import tensorflow as tf

def apply(inputs, reuse):
    """ a simple operation to enlarge the second elements of a vector by x2 """
    with tf.variable_scope('my_var_scope') as scope:
        if reuse:
            scope.reuse_variables()
        weight = tf.Variable([[1.,0.],[0.,2.]], name='weights')
        outputs = tf.matmul(weight, inputs)
    
    return outputs

def main(_):
    a = tf.constant([[1.],[2.]], name='my_var1')
    b = tf.constant([[2.],[3.]], name='my_var2')

    a_out = apply(a,False) # the first time to use the variable, we define it
    b_out = apply(b,True) # the second time to use the variable, we reuse it

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a_val, b_val = sess.run([a_out, b_out])
        print(a_val)
        print(b_val)

if __name__ == '__main__':
    tf.app.run()

import tensorflow as tf

def arithmetic_op():
    # define two tensors.
    a = tf.constant([0.1, 0.2, 0.3], name='a')
    b = tf.constant([0.3, 0.2, 0.1], name='b')

    # define operations
    c = a + b
    d = a - b
    e = a * b
    f = a / b

    return a, b, c, d, e, f

if __name__ == '__main__':
    a, b, c, d, e, f = arithmetic_op()

    # uncomment the following code line to see what happen?
    # this might be the first tensorflow shock
    # print(a, b, c, d, e, f)

    sess = tf.Session() # create a session to run your tensorflow operations
    a_, b_, c_, d_, e_, f_ = sess.run([a,b,c,d,e,f])
    print(a_, b_, c_, d_, e_, f_)
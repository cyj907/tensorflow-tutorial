# Decay learning rate

> Sometimes, we might want the learning rate to decay over steps. Using _tf.train.exponential\_decay_ is a simple and useful option.

```python
start_learning_rate = 0.01
lr_decay_step = 100
lr_decay_rate = 0.99
# remember to increment the global_step using optimizer
global_step = tf.Variable(0, name='global_step', trainable=False)
learning_rate = tf.train.exponential_decay(start_learning_rate,
                        global_step,FLAGS.lr_decay_step,FLAGS.lr_decay_rate)
```
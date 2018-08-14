# Batch norm

> Using _batch norm_ in tensorflow can be tricky. Some people tend to write their own _batch norm_ layer. And using the _batch norm_ in _tf.contrib.layers_ can be an option for beginners.

As mentioned in the [document](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm), we have to add some extra codes to get the batch norm working correctly.
Another thing to notice is that, remember to save _global\_variables_ instead of only _trainable\_variables_, because the parameters hidden in the batch norm layer that cover the moving average and variance of the batch data are __NOT__ trainable.

```python
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)
```
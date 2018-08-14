# Session config

> We already know that _tf.Session_ is critical for use to run a tensorflow program. Actually, we can apply some configurations to it so that the program is more suitable for our use.

```python
config = tf.ConfigProto()
# the following line controls the fraction of GPU memory taken for training
# so that we can use one gpu for multiple programs (saving resources)
config.gpu_options.per_process_gpu_memory_fraction = 0.4
with tf.Session(config=config) as sess:
    # your training codes here
```

More can be found in this [document](https://www.tensorflow.org/guide/using_gpu)
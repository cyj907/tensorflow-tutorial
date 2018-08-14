# Selected variables

> Sometimes it is important to give names to different variables and assign them to different _variable scope_, so that we can find the corresponding variables easier by their names or scopes. This mechanism is useful for debugging your codes, and might also be helpful when the training is specific to some variables.

The basic usage of _variable\_scope_:
```python
with tf.variable_scope('my_var_scope'):
    a = tf.Variable(0, name='my_var_name')
```

Here, we will show scenarios where _variable scope_ is useful:
1. reuse some variables
2. save and load only part of the variables in the checkpoint
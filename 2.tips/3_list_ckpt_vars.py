""" codes are from https://stackoverflow.com/questions/38944238/how-do-i-list-certain-variables-in-the-checkpoint """

from tensorflow.contrib.framework.python.framework import checkpoint_utils

var_list = checkpoint_utils.list_variables('../0.basics/ckpt')
for v in var_list:
    print(v)
# hello world

> Firstly, tensorflow is wrapped by python.
>
> You can use python codes in your program without any trouble.

```python
import tensorflow as tf

def hello_world():
    print('Hello world! Welcome to tensorflow {}'.format(tf.__version__))

if __name__ == '__main__':
    hello_world()
```

Remember to import _tensorflow_ every time you use it.
This program simply print the version of tensorflow in your environment.
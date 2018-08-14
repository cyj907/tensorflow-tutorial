# Dataloader

> For now, we only use fake data for training.
> Let's start to use real data, and build a data loader to load raw data.
> To speed up data loading process in tensorflow, converting all your data into _tfrecords_ are recommended.

First, let's learn to convert image data into tfrecords.
In _4.1\_tfrecords.py_, we try to convert some images into tfrecords.
Then, we will write a data loader to load these files in _4.2\_loader.py_

Since the codes are becoming long, we will no longer show them here.
But some important codes and explanation will still be listed below.

1. multi-thread
   
   The codes in _4.1\_tfrecords.py_ use multi-thread mechanism (coordinator) to generate _tfrecords_.

2. multi-tfrecords

   All image data were divided into _chunks_ of data stored in _tfrecord\_dir_

3. images SHOULD be stored in _uint8_ if the RGB values are ranged between 0 and 255.

Let's have a look at _4.2\_loader.py_. It used _tf.dataset_ to implement the data loader, which is the recommended way for tensorflow.

1. a parser should be defined to extract the tfrecords data
2. _global\_step_ is defined to record the current training step
3. a _try_ _except_ mechanism is used to train the model for exact _epoch_
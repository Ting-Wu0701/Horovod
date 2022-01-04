import tensorflow as tf
import pandas as pd
from tensorflow.keras.datasets import mnist
import pandas as pd
import numpy as np
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

import horovod.tensorflow.keras as hvd

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

batch_size = 128
num_classes = 10

# Horovod: adjust number of epochs based on number of GPUs.
epochs = int(math.ceil(12.0 / hvd.size()))

# Input image dimensions
img_rows, img_cols = 28, 28

# The data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()


def getImagedata(X_table):
    """
    前提工作：将图片转为数组的形式存放在csv文件中，
    此函数的功能是将mnist中的图片转为数组形式后，将每一张图片存储到List中，List[0]即代表第一张图片
    原mnist中的图片转为数组形式后shape是（6000，28，28）
    5000个训练样本，1000个测试样本，现在取前训练样本中的前1000个作为训练样本，测试样本中的前10个作为测试样本
    :param X_table:由1000个训练样本和10个测试样本组成的csv文件（不含标签）
    :return:List，List中的每一项代表一张图片
    """

    List = []
    list = []
    for i in X:
        l = np.array(i).tolist()
        list.append(l)
        if len(list) != 28:
            continue
        else:
            List.append(list)
            list = []

    return List

X = pd.read_csv("/root/data/mnist_data.csv")
X = np.array(X)
label = X[0:1010, -1]
X = X[:, 0:X.shape[1]-1]

List = getImagedata(X)
LL = np.array(List)
x_train = LL[0:1000]
x_test = LL[1000:1010]


# x_train_label = y_train[0:1000]
# y_test_label = y_test[0:10]
# label = []
# for i in x_train_label:
#     label.append(i)
# for j in y_test_label:
#     label.append(j)

# label = [5, 0, 4, 1, 9, 2, 1, 3, 1, 4, 3, 5, 3, 6, 1, 7, 2, 8, 6, 9, 4, 0, 9, 1, 1, 2, 4, 3, 2, 7, 3, 8, 6, 9, 0, 5, 6, 0, 7, 6, 1, 8, 7, 9, 3, 9, 8, 5, 9, 3, 3, 0, 7, 4, 9, 8, 0, 9, 4, 1, 4, 4, 6, 0, 4, 5, 6, 1, 0, 0, 1, 7, 1, 6, 3, 0, 2, 1, 1, 7, 9, 0, 2, 6, 7, 8, 3, 9, 0, 4, 6, 7, 4, 6, 8, 0, 7, 8, 3, 1, 5, 7, 1, 7, 1, 1, 6, 3, 0, 2, 9, 3, 1, 1, 0, 4, 9, 2, 0, 0, 2, 0, 2, 7, 1, 8, 6, 4, 1, 6, 3, 4, 5, 9, 1, 3, 3, 8, 5, 4, 7, 7, 4, 2, 8, 5, 8, 6, 7, 3, 4, 6, 1, 9, 9, 6, 0, 3, 7, 2, 8, 2, 9, 4, 4, 6, 4, 9, 7, 0, 9, 2, 9, 5, 1, 5, 9, 1, 2, 3, 2, 3, 5, 9, 1, 7, 6, 2, 8, 2, 2, 5, 0, 7, 4, 9, 7, 8, 3, 2, 1, 1, 8, 3, 6, 1, 0, 3, 1, 0, 0, 1, 7, 2, 7, 3, 0, 4, 6, 5, 2, 6, 4, 7, 1, 8, 9, 9, 3, 0, 7, 1, 0, 2, 0, 3, 5, 4, 6, 5, 8, 6, 3, 7, 5, 8, 0, 9, 1, 0, 3, 1, 2, 2, 3, 3, 6, 4, 7, 5, 0, 6, 2, 7, 9, 8, 5, 9, 2, 1, 1, 4, 4, 5, 6, 4, 1, 2, 5, 3, 9, 3, 9, 0, 5, 9, 6, 5, 7, 4, 1, 3, 4, 0, 4, 8, 0, 4, 3, 6, 8, 7, 6, 0, 9, 7, 5, 7, 2, 1, 1, 6, 8, 9, 4, 1, 5, 2, 2, 9, 0, 3, 9, 6, 7, 2, 0, 3, 5, 4, 3, 6, 5, 8, 9, 5, 4, 7, 4, 2, 7, 3, 4, 8, 9, 1, 9, 2, 8, 7, 9, 1, 8, 7, 4, 1, 3, 1, 1, 0, 2, 3, 9, 4, 9, 2, 1, 6, 8, 4, 7, 7, 4, 4, 9, 2, 5, 7, 2, 4, 4, 2, 1, 9, 7, 2, 8, 7, 6, 9, 2, 2, 3, 8, 1, 6, 5, 1, 1, 0, 2, 6, 4, 5, 8, 3, 1, 5, 1, 9, 2, 7, 4, 4, 4, 8, 1, 5, 8, 9, 5, 6, 7, 9, 9, 3, 7, 0, 9, 0, 6, 6, 2, 3, 9, 0, 7, 5, 4, 8, 0, 9, 4, 1, 2, 8, 7, 1, 2, 6, 1, 0, 3, 0, 1, 1, 8, 2, 0, 3, 9, 4, 0, 5, 0, 6, 1, 7, 7, 8, 1, 9, 2, 0, 5, 1, 2, 2, 7, 3, 5, 4, 9, 7, 1, 8, 3, 9, 6, 0, 3, 1, 1, 2, 6, 3, 5, 7, 6, 8, 3, 9, 5, 8, 5, 7, 6, 1, 1, 3, 1, 7, 5, 5, 5, 2, 5, 8, 7, 0, 9, 7, 7, 5, 0, 9, 0, 0, 8, 9, 2, 4, 8, 1, 6, 1, 6, 5, 1, 8, 3, 4, 0, 5, 5, 8, 3, 6, 2, 3, 9, 2, 1, 1, 5, 2, 1, 3, 2, 8, 7, 3, 7, 2, 4, 6, 9, 7, 2, 4, 2, 8, 1, 1, 3, 8, 4, 0, 6, 5, 9, 3, 0, 9, 2, 4, 7, 1, 2, 9, 4, 2, 6, 1, 8, 9, 0, 6, 6, 7, 9, 9, 8, 0, 1, 4, 4, 6, 7, 1, 5, 7, 0, 3, 5, 8, 4, 7, 1, 2, 5, 9, 5, 6, 7, 5, 9, 8, 8, 3, 6, 9, 7, 0, 7, 5, 7, 1, 1, 0, 7, 9, 2, 3, 7, 3, 2, 4, 1, 6, 2, 7, 5, 5, 7, 4, 0, 2, 6, 3, 6, 4, 0, 4, 2, 6, 0, 0, 0, 0, 3, 1, 6, 2, 2, 3, 1, 4, 1, 5, 4, 6, 4, 7, 2, 8, 7, 9, 2, 0, 5, 1, 4, 2, 8, 3, 2, 4, 1, 5, 4, 6, 0, 7, 9, 8, 4, 9, 8, 0, 1, 1, 0, 2, 2, 3, 2, 4, 4, 5, 8, 6, 5, 7, 7, 8, 8, 9, 7, 4, 7, 3, 2, 0, 8, 6, 8, 6, 1, 6, 8, 9, 4, 0, 9, 0, 4, 1, 5, 4, 7, 5, 3, 7, 4, 9, 8, 5, 8, 6, 3, 8, 6, 9, 9, 1, 8, 3, 5, 8, 6, 5, 9, 7, 2, 5, 0, 8, 5, 1, 1, 0, 9, 1, 8, 6, 7, 0, 9, 3, 0, 8, 8, 9, 6, 7, 8, 4, 7, 5, 9, 2, 6, 7, 4, 5, 9, 2, 3, 1, 6, 3, 9, 2, 2, 5, 6, 8, 0, 7, 7, 1, 9, 8, 7, 0, 9, 9, 4, 6, 2, 8, 5, 1, 4, 1, 5, 5, 1, 7, 3, 6, 4, 3, 2, 5, 6, 4, 4, 0, 4, 4, 6, 7, 2, 4, 3, 3, 8, 0, 0, 3, 2, 2, 9, 8, 2, 3, 7, 0, 1, 1, 0, 2, 3, 3, 8, 4, 3, 5, 7, 6, 4, 7, 7, 8, 5, 9, 7, 0, 3, 1, 6, 2, 4, 3, 4, 4, 7, 5, 9, 6, 9, 0, 7, 1, 4, 2, 7, 3, 6, 7, 5, 8, 4, 5, 5, 2, 7, 1, 1, 5, 6, 8, 5, 8, 4, 0, 7, 9, 9, 2, 9, 7, 7, 8, 7, 4, 2, 6, 9, 1, 7, 0, 6, 4, 2, 5, 7, 0, 7, 1, 0, 3, 7, 6, 5, 0, 6, 1, 5, 1, 7, 8, 5, 0, 3, 4, 7, 7, 5, 7, 8, 6, 9, 3, 8, 6, 1, 0, 9, 7, 1, 3, 0, 5, 6, 4, 4, 2, 4, 4, 3, 1, 7, 7, 6, 0, 3, 6, 7, 2, 1, 0, 4, 1, 4, 9, 5, 9]
y_train = label[0:1000]
y_test = label[1000:1010]


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Horovod: adjust learning rate based on number of GPUs.
opt = keras.optimizers.Adadelta(1.0 * hvd.size())

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt, backward_passes_per_step=1)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

model.fit(x_train, y_train,
          batch_size=batch_size,
          callbacks=callbacks,
          epochs=epochs,
          verbose=1 if hvd.rank() == 0 else 0,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

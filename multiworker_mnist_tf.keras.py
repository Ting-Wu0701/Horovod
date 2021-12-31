import os

import tensorflow as tf
import numpy as np


strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()


NUM_WORKERS = len(os.getenv("TF_CONFIG")["cluster"]["worker"])
GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS

mnist_images = np.random.randint(low=0, high=255, size=[60000, 28, 28], dtype=np.uint8)
mnist_labels = np.random.randint(low=0, high=9, size=[60000,], dtype=np.uint8)

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
             tf.cast(mnist_labels, tf.int64))
)
dataset = dataset.repeat().shuffle(10000).batch(GLOBAL_BATCH_SIZE)

with strategy.scope():
    mnist_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    mnist_model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])

# Keras 的 `model.fit()` 以特定的时期数和每时期的步数训练模型。
# 注意此处的数量仅用于演示目的，并不足以产生高质量的模型。
mnist_model.fit(x=dataset, epochs=50, steps_per_epoch=5)

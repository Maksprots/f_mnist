import tensorflow as tf

perceptron = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(794 / 2, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

simple_cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,
                           (3, 3),
                           padding='same',
                           activation='relu',
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                 strides=2,
                                 padding='valid'),
    tf.keras.layers.Conv2D(64,
                           (3, 3),
                           padding='same',
                           activation='relu',
                           ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                 strides=2,
                                 padding='valid'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(300, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)

])

# Пытался написать resnet подобную, но что-то невышло
# inp = tf.keras.Input(shape=( 28, 28, 1), name='img')
# layer = tf.keras.layers.Conv2D(32,
#                                 (3, 3),
#                                 padding='same',
#                                 activation='relu',
#                                 input_shape=(28, 28, 1))(inp)
# layer = tf.keras.layers.Conv2D(64,
#                                (3, 3),
#                                padding='same',
#                                activation='relu')(layer)
# out_1 = tf.keras.layers.MaxPooling2D(3)(layer)
#
# layer = tf.keras.layers.Conv2D(64,
#                                (3, 3),
#                                padding='same',
#                                activation='relu')(out_1)
# layer = tf.keras.layers.Conv2D(64,
#                                (3, 3),
#                                padding='same',
#                                activation='relu')(layer)
# out_2 = tf.keras.layers.add([layer, out_1])
#
#
# layer = tf.keras.layers.Conv2D(64,
#                                (3, 3),
#                                padding='same',
#                                activation='relu')(out_2)
# layer = tf.keras.layers.Conv2D(64,
#                                (3, 3),
#                                padding='same',
#                                activation='relu')(layer)
# out_3 = tf.keras.layers.add([layer, out_2])
# layer = tf.keras.layers.Conv2D(64,
#                                (3, 3),
#                                padding='same',
#                                activation='relu')(out_3)
# layer = tf.keras.layers.GlobalAveragePooling2D()(layer)
# layer = tf.keras.layers.Dense(256, activation=tf.nn.relu)(layer)
# layer = tf.keras.layers.Dropout(0.5)(layer)
#
# main_out = tf.keras.layers.Dense(256, activation='softmax')(layer)
# rel = tf.keras.Model(inp, main_out, name='resnet_similar')

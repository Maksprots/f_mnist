import numpy as np
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import models, layers
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import img_to_array, array_to_img
from sklearn.model_selection import train_test_split


class Vgg16FMnist:
    def __init__(self):
        self.data_tr = None
        self.data_tst = None
        self.ans_tr = None
        self.ans_tst = None
        self.my_vgg = None

    def load_and_prepearing_data(self):
        (data_tr,
         ans_tr), \
            (data_tst, ans_tst) = \
            keras.datasets.fashion_mnist.load_data()
        data_tr = np.dstack([data_tr] * 3).reshape(-1, 28, 28, 3)
        data_tst = np.dstack([data_tst] * 3).reshape(-1, 28, 28, 3)
        data_tr = np.asarray([img_to_array(array_to_img(im, scale=False)
                                           .resize((48, 48))) for im in data_tr])
        data_tst = np.asarray([img_to_array(array_to_img(im, scale=False)
                                            .resize((48, 48))) for im in data_tst])
        x = np.array(data_tr)
        y = to_categorical(ans_tr)
        self.data_tr, self.data_tst, \
            self.ans_tr, self.ans_tst = \
            train_test_split(x, y, test_size=0.2, random_state=5)

    def make_model(self):
        input_layer = layers.Input(shape=(48, 48, 3))
        my_vgg16 = VGG16(weights='imagenet',
                         input_tensor=input_layer,
                         include_top=False)
        last_layer = my_vgg16.output
        flatten = layers.Flatten()(last_layer)
        layer = layers.Dense(100, activation='relu')(flatten)
        layer = layers.Dense(100, activation='relu')(flatten)
        layer = layers.Dense(100, activation='relu')(flatten)
        output_layer = layers.Dense(10, activation='softmax')(flatten)
        self.my_vgg = models.Model(inputs=input_layer, outputs=output_layer)

        for layer in self.my_vgg.layers[:-1]:
            layer.trainable = False
        self.my_vgg.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self):
        self.my_vgg.fit(self.data_tr, self.ans_tr,
                        epochs=5,
                        batch_size=256,
                        verbose=True,
                        validation_data=(self.data_tst,
                                         self.ans_tst))

    def testing(self):
        results = self.my_vgg.evaluate(self.data_tst,
                                       self.ans_tst, batch_size=128)
        return results[1]

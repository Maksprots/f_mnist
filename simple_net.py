from models import perceptron, simple_cnn
import tensorflow_datasets as tfd
import math
from utils import show_sample


class SimpleNet:
    def __init__(self,
                 optimazer='adam',
                 loss='sparse_categorical_crossentropy',
                 batch=32,
                 net_model=None):
        self.net_model = net_model
        self.net_model.compile(optimizer=optimazer,
                               loss=loss,
                               metrics=['accuracy'])
        self.batch = batch
        self._fit_set = []
        self._train_len = None
        self._testing_len = None
        self._test_set = None

    def load_dataset(self):
        dataset, meta = tfd.load('fashion_mnist',
                                 as_supervised=True,
                                 with_info=True)

        fit_set = dataset['train']
        test_set = dataset['test']
        self._train_len = meta.splits['train'].num_examples
        self._testing_len = meta.splits['test'].num_examples
        self._fit_set = fit_set.repeat(32) \
            .shuffle(self._train_len). \
            batch(self.batch)
        self._test_set = test_set.batch(self.batch)

    def learn(self, epochs):
        self.net_model.fit(self._fit_set,
                           epochs=epochs,
                           steps_per_epoch=
                           math.ceil(self._train_len
                                     / self.batch))

    def testing(self):
        test_loss, test_accuracy = self.net_model. \
            evaluate(self._test_set,
                     steps=math.ceil(self._testing_len / self.batch))
        return test_accuracy
        print(f'Точность {name}: {test_accuracy:.2f}',)

    def predict_element(self, number):
        images = []
        titles = []
        for im, l in self._fit_set.take(number):
            images.append(im.numpy())
            titles.append(l.numpy())
        for i in range(len(images)):
            predicted = self.net_model.predict(images[i])
            show_sample(predicted[0], titles[i][0], images[i][0])


if __name__ == '__main__':
    s = SimpleNet(net_model=v, optimazer='adam')
    s.load_dataset()
    s.learn(3)
    s.testing('percept')
    # s.predict_element(1)

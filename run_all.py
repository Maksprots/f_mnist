from f_mnist.models import perceptron, simple_cnn
from f_mnist.simple_net import SimpleNet
from f_mnist.vgg_16 import Vgg16FMnist


def main():
    perc = SimpleNet(net_model=perceptron)
    cnn = SimpleNet(net_model=simple_cnn)
    vgg16 = Vgg16FMnist()
    vgg16.make_model()

    perc.load_dataset()
    cnn.load_dataset()
    vgg16.load_and_prepearing_data()

    perc.learn(5)
    cnn.learn(5)
    vgg16.fit()

    p_a = perc.testing()
    c_a = cnn.testing()
    v_a = vgg16.testing()

    perc.predict_element(1)
    cnn.predict_element(1)
    print('Точности\n перцептрон: {}\n простая сверточная: {}\n vgg16: {}'
          .format(p_a, c_a, v_a))


if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
import numpy as np

categories = ['Футболка', 'Шорты', 'Свитер', 'Платье',
               'Плащ', 'Сандали', 'Рубашка', 'Кроссовка', 'Сумка',
               'Ботинок']


def show_sample(predictions_array, sets_label, image):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(image, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == sets_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel('Ответ: {} {:.1f}\nНа самом деле: {}'.format(
        categories[predicted_label],
        100 * np.max(predictions_array),
        categories[sets_label]),
        color='green' if predicted_label == sets_label else 'red')
    plt.show()

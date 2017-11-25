import numpy as np
import cPickle
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

"""
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

filename = '../datasets/cifar-10-batches-py/data_batch_1'
dic = unpickle(filename)
x_train_raw = dic['data']
x_train = np.ndarray([10000, 32, 32, 3])
for i in range(10000):
    x_train[i, :, :, :] = np.transpose(np.reshape(x_train_raw[i],(3, 32, 32)), (1, 2, 0))
y_train = np.array(dic['labels'])

filename = '../datasets/cifar-10-batches-py/test_batch'
dic = unpickle(filename)
x_test_raw = dic['data']
x_test = np.ndarray([10000, 32, 32, 3])
for i in range(10000):
    x_test[i, :, :, :] = np.transpose(np.reshape(x_test_raw[i],(3, 32, 32)), (1, 2, 0))

y_test = np.array(dic['labels'])

k = 56
filename = '../datasets/cifar-10-batches-py/batches.meta'
dic = unpickle(filename)
label_names = dic['label_names']

print label_names[y_test[k]]

plt.imshow(x_test[k])
plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import cPickle
import keras
from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

# import parameters
from mnist_params import *

"""
loading vae model back is not a straight-forward task because of custom loss layer.
we have to define some architecture back again to specify custom loss layer and hence to load model back again.
"""

# encoder architecture
x = Input(shape=(original_dim,))
encoder_h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(encoder_h)
z_log_var = Dense(latent_dim)(encoder_h)

# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x

# load saved models
vae = keras.models.load_model('../models/ld_%d_id_%d_e_%d_vae.h5' % (latent_dim, intermediate_dim, epochs),
    custom_objects={'latent_dim':latent_dim, 'epsilon_std':epsilon_std, 'CustomVariationalLayer':CustomVariationalLayer})
encoder = keras.models.load_model('../models/ld_%d_id_%d_e_%d_encoder.h5' % (latent_dim, intermediate_dim, epochs),
    custom_objects={'latent_dim':latent_dim, 'epsilon_std':epsilon_std, 'CustomVariationalLayer':CustomVariationalLayer})
generator = keras.models.load_model('../models/ld_%d_id_%d_e_%d_generator.h5' % (latent_dim, intermediate_dim, epochs),
    custom_objects={'latent_dim':latent_dim, 'epsilon_std':epsilon_std, 'CustomVariationalLayer':CustomVariationalLayer})

fname = '../models/ld_%d_id_%d_e_%d_history.pkl' % (latent_dim, intermediate_dim, epochs)

# load history if saved
try:
    with open(fname, 'rb') as fo:
        history = cPickle.load(fo)
    print history
except:
    print "training history not saved"

# load dataset to plot latent space
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()


# display a 2D manifold of the digits
n = 30  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian

grid_x = 1.5 * norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = 1.5 * norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        #xi = input()
        #yi = input()
        z_sample = np.array([[xi, yi]])
        #print z_sample
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        #plt.figure(figsize=(10, 10))
        #plt.imshow(digit, cmap='Greys_r')
        #plt.show()
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()

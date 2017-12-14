# VAE for Image Generation
Variational AutoEncoder - Keras implementation on mnist and cifar10 datasets

## Dependencies
- keras
- tensorflow / theano (current implementation is according to tensorflow. It can be used with theano with few changes in code)
- numpy, matplotlib, scipy

## implementation Details
code is highly inspired from keras examples of vae : ![vae](https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py),
![vae_deconv](https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder_deconv.py)  
(_source files contains some code duplication_)

### MNIST

- images are flatten out to treat them as 1D vectors
- encoder and decoder - both have normal neural network architecture
- network architecture is as follows  
![mnist_vae_architecture](images/mnist_architecture.png)

##### ![src/mnist_train.py](src/mnist_train.py)
- it trains vae model according to the hyperparameters defined in ![src/mnist_params.py](src/mnist_params.py)
- entire vae model, encoder and decoder is stored as keras models in ![models](models/) directory as `ld_<latent_dim>_id_<intermediate_dim>_e_<epochs>_<vae/encoder/decoder>.h5` where `<latent_dim>` is number of latent dimensions, `<intermediate_dim>` is number of neurons in hidden layer and `<epochs>` is number of training epochs
- after training, the saved model can be used to analyse the latent distribution and to generate new images
- it also stores the training history in `ld_<latent_dim>_id_<intermediate_dim>_e_<epochs>_history.pkl`

##### ![src/mnist_2d_latent_space_and_generate.py](src/mnist_2d_latent_space_and_generate.py)
- it is only for 2 dimensional latent space
- it loads trained model according to the hyperparameters defined in `mnist_params.py`
- it displays the latent space distribution and then generates the images according to user input of latent variables (see the code as it is almost self-explanatory)
- it can also generate images from latent vectors randomly sampled from 2D latent space (comment out the user input lines) and display them in a grid

##### ![src/mnist_3d_latent_space_and_generate.py](src/mnist_3d_latent_space_and_generate.py)
- it is same as `mnist_2d_latent_space_and_generate.py` but it is for 3d latent space

##### ![src/mnist_general_latent_space_and_generate.py](src/mnist_general_latent_space_and_generate.py)
- it loads trained model according to the hyperparameters defined in `mnist_params.py`
- if latent space is either 2D or 3D, it displays it
- it displays a grid of images generated from randomly sampled latent vectors

#### results
##### uniformly sampled 2D latent space
![2D](images/ld_2_e_60_uniform.png)
##### uniformly sampled 3D latent space
![2D](images/ld_3_e_60_uniform.png)


### CIFAR10

- images are treated as 2D input
- encoder has the architecture of convolutional neural network and decoder has the architecture of deconvolutional network
- network architecture for encoder and decoder are as follows  
##### encoder
![cifar10_vae_encoder](images/cifar10_encoder.png)
##### dencoder
![cifar10_vae_decoder](images/cifar10_decoder.png)

##### ![src/cifar10_train.py](src/cifar10_train.py) , ![src/cifar10_generate.py](src/cifar10_generate.py)
same as mnist files

### CALTECH101

- `caltech101_<sz>_train.py` and `caltech101_<sz>_generate.py` (where `sz` is the size of input image - here the training was done for two sizes - 92*92 and 128*128) are same as cifar10 dataset files
- as the image size is large, more computation power is needed to train the model
- in `dataset` directory, ![src/caltech101_preprocess.py](src/caltech101_preprocess.py) is provided to preprocess the dataset

# Intro

In this tutorial you will learn how to:

1. Create a Docker image to train a neural network that classifies images of clothing
2. Import the image and a dataset of labeled images into BatchX
3. Run the image in BatchX
4. Get results back

# Docker image creation





# Import to BatchX all you need: data & Docker image

Download data:

> wget 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

Copy to BatchX file system:

> bx cp mnist.npz bx://data/fashion_mnist/mnist.npz

Import image:

> bx import josemfer/batchx-demo-tensorflow:0.0.1 


See your imported image:

> bx images


# Run your BatchX image

> bx submit tutorial/batchx-demo-tensorflow:1.0.0 '{"input_file_path":"bx://data/fashion_mnist/mnist.npz","num_epochs": 10}'
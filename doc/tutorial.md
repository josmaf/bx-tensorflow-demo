# Intro

In this tutorial you will learn how to:

1. Create a Docker image to train a neural network that classifies images of clothing
2. Import the image and a dataset of labeled images into BatchX
3. Run the image in BatchX
4. Get results back

# Requisites

BatchX client & a BatchX working account
Docker client & a public Docker registry account. For instance, https://hub.docker.com/

# 1. Docker image creation

TODO: add steps to build the Docker image

Build image:

> docker build -f ./docker/Dockerfile -t josemfer/batchx-tensorflow-gpu-demo:latest .

Push to your Docker registry:

> docker push josemfer/batchx-tensorflow-gpu-demo:latest

Please note: we're using a public registry, but a private one could also be used instead.

# 2. Import to BatchX all you need: Docker image & data

Import image:

> bx import josemfer/batchx-tensorflow-gpu-demo:latest

See your imported image:

> bx images

There should be an image named: tutorial/tensorflow-gpu-demo:0.0.1

Download data:

> wget 'https://github.com/josmaf/bx-tensorflow-demo/blob/master/data/fashion_mnist.npz'

Copy data to BatchX file system:

> bx cp fashion_mnist.npz bx://data/fashion_mnist/mnist.npz

# Run your BatchX image

> bx run -v=4 -m=15000 -g=1 -f=T4 tutorial/tensorflow-gpu-demo:0.0.2 '{"data_file_path":"bx://data/fashion_mnist.npz","num_epochs": 10}'

Parameters:
- v=4     -> 4 vCPUs
- m=15000 -> 15 GB of RAM
- g=1     -> At least 1 GPU
- f=T4    -> GPU type

# Intro

In this tutorial you will learn how to:

1. Create a Docker image to train a neural network that classifies images of clothing
2. Import the image and a dataset of training data (clothing photos) into BatchX
3. Run the image in BatchX
4. Get results back

# Requisites

1. BatchX client and a BatchX working account
2. Docker client and a public Docker registry account. For instance, https://hub.docker.com/

# 1. Docker image creation

We'll create three files: 

1. Dockerfile : Docker image definition
2. entrypoint.py : script to act as a 'bridge' between BatchX and the 'trainer.py' script in charge of training the model
3. trainer.py : script to train the model

TODO: more



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

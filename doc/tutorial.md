# Intro

In this tutorial you'll learn how to:

1. Create a Docker image to train a neural network that classifies images of clothing
2. Import the image and a dataset of training data (labeled clothing photos) into BatchX
3. Run your BatchX image
4. Get results

# Prerequisites

- BatchX client and a BatchX working account
- Docker client and a public Docker registry account. For instance, https://hub.docker.com/

# 1. Docker image creation

We'll create three files: 

1. Dockerfile : Docker image definition

```
FROM tensorflow/tensorflow:latest-gpu-py3
WORKDIR /batchx
COPY . .
RUN chmod -R +x /batchx 
ENTRYPOINT python3 /batchx/entrypoint.py
LABEL 'io.batchx.manifest-03'='{ \
	"name":"tutorial/tensorflow-gpu-demo", \
	"title":"BatchX Tensorflow GPU Demo Image", \
	"schema":{  \
		"input":{ \
			"$schema":"http://json-schema.org/draft-07/schema#", \
			"type":"object", \
			"properties":{ \
				"data_file_path":{ \
					"type":"string", \
					"required":true, \
					"format":"file", \
					"description":"Input data file path" \
					}, \
				"num_epochs":{ \
					"type":"integer", \
					"required":true, \
					"description":"Number of epochs" \
					} \
				} \
			}, \
		"output":{ \
			"$schema":"http://json-schema.org/draft-07/schema#", \
			"type":"object", \
			"additionalProperties":false, \
			"properties":{ \
				"model_file_path":{ \
					"required":true, \
					"type":"string", \
					"format":"file", \
					"description":"Tensorflow HDF5 model file path" \
					}, \
				"meta_file_path":{ \
					"required":true, \
					"type":"string", \
					"format":"file", \
					"description":"Meta-data file path" \
					}, \
				"predictor_file_path" : { \
					"required":true, \
					"type":"string", \
					"format":"file", \
					"description":"Predictor script file path" \
					} \
				} \
			} \
		}, \
	"author":"BatchX", \
	"version":"1.0.0", \
	"runtime":{"minMem":8000, \
				"gpus":"required" \
			} \
	}'
```


2. entrypoint.py : script to act as a 'bridge' between BatchX and the 'trainer.py' script in charge of training the model

It will read input parameters and pass them to trainer module.

```
import json
import os
import trainer  # Module with our code to train a neural network
from shutil import copyfile


# BatchX saves into /batchx/input/input.json what we passed to bx client when running the job
# So now we have to read input.json file to get the input parameters
with open("/batchx/input/input.json", "r") as input_file:
    input_json_dict = json.loads(input_file.read()) 

# Get input data file local path
input_file_path = input_json_dict["data_file_path"]  

# Get number of epochs
num_epochs = input_json_dict["num_epochs"]

# Generated files must be located somewhere below '/batchx/output/' folder
# This folder has been automatically created by BatchX
output_folder = "/batchx/output/"

# The train method needs:
#     input_data_file: path of training data file
#     num_epochs: number of epochs (iterations)
#     output_folder: path of the folder where model and meta-data files will be saved
# It will return the paths of generated model and meta-data files
model_file_path, meta_file_path = trainer.train(input_file_path, num_epochs, output_folder)

# Additionally, we copy a script to use the trained model
copyfile('/batchx/predictor.py', os.path.join(output_folder, 'predictor.py'))

# Write model and meta-data file paths into 'output.json'. BatchX will copy them into its FS. 
with open('/batchx/output/output.json', 'w+') as output_file:
    json.dump({'model_file_path': model_file_path, 
    	'meta_file_path' : meta_file_path, 
    	'predictor_file_path' : os.path.join(output_folder, 'predictor.py')  }, output_file)
```

3. trainer.py : script to train the model

```
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os.path


def train(input_file_path, num_epochs, output_folder):

    # Read data
    data = np.load(input_file_path, allow_pickle=True)
    train_images = data['x_train']
    train_labels =data['y_train']
    test_images = data['x_test']
    test_labels = data['y_test']

    # Pre-process data
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Force using first GPU device
    with tf.device('/gpu:0'):

        # Build the model
        model = keras.Sequential([
            keras.layers.Dense(32, input_shape=(784,)),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(10)])

        # Compile the model
        model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

        # Train the model
        model.fit(train_images, train_labels, epochs=num_epochs)

        # Evaluate model
        test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

        # Save model
        model_file_path = os.path.join(output_folder, 'model.h5')
        model.save(model_file_path)

        # Save meta-data
        meta_file_path = os.path.join(output_folder, 'model.info')
        with open(meta_file_path, 'w+') as output_file:
            json.dump({'test loss': str(test_loss), 'test accuracy': str(test_acc)}, output_file)

    return model_file_path, meta_file_path
```

Build image:

> docker build -f ./Dockerfile -t <docker_registry_username>/batchx-tensorflow-gpu-demo:latest .

Please note: you must change <docker_registry_username> by your Docker registry user name. 

Push to your Docker registry:

> docker push <docker_registry_username>/batchx-tensorflow-gpu-demo:latest

Note: we're using a public registry, but a private one could be used instead.

# 2. Import to BatchX all you need: Docker image & Data

Import image:

> bx import <docker_registry_username>/batchx-tensorflow-gpu-demo:latest

See your imported image:

> bx images

There should be an image named: tutorial/tensorflow-gpu-demo:1.0.0

Download data:

> wget 'https://github.com/josmaf/bx-tensorflow-demo/blob/master/data/fashion_mnist.npz'

Copy data to BatchX file system:

> bx cp fashion_mnist.npz bx://data/fashion_mnist/mnist.npz

# 3. Run your BatchX image

> bx run -v=4 -m=15000 -g=1 -f=T4 tutorial/tensorflow-gpu-demo:0.0.2 '{"data_file_path":"bx://data/fashion_mnist.npz","num_epochs": 10}'

Parameters:
- v=4     -> 4 vCPUs
- m=15000 -> 15 GB of RAM
- g=1     -> At least 1 GPU
- f=T4    -> GPU type

If everything went ok, you should see something like:

> [batchx] [2020/06/10 15:24:59] Job status: SUCCEEDED
> {"model_file_path":"bx://jobs/127/output/model.h5","meta_file_path":"bx://jobs/127/output/model.info","predictor_file_path":"bx://jobs/127/output/predictor.py"}


# 4. Get results

Copy model binary file from BatchX to your local filesystem:

> bx cp bx://jobs/127/output/model.h5 .

Copy model meta-data file and predictor.py script:

> bx cp bx://jobs/127/output/model.info .
> bx cp bx://jobs/127/output/predictor.py .

You can test the model by downloading an input image and trying predictor.py script along with the generated model:

TODO


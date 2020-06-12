# Intro

In this tutorial you'll learn how to:

1. Create a Docker image to train a neural network that classifies photos of clothing
2. Import the Docker image into BatchX 
3. Import a dataset of labeled clothing photos into BatchX
4. Run your BatchX image (namely, the Docker image you've already imported in step 2)
5. Get the trained model from BatchX file system and use it to classify a photo

# Prerequisites

- BatchX client and a BatchX working account
- Docker client and a public Docker registry account. For instance, https://hub.docker.com/

# 1. Docker image creation

We'll have to write three files: 

1. Dockerfile : Docker image definition

```text
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
				"training_images_path":{ \
					"type":"string", \
					"required":true, \
					"format":"file", \
					"description":"Zip file with training images" \
					}, \
                "training_labels_path":{ \
					"type":"string", \
					"required":true, \
					"format":"file", \
					"description":"Csv file with training labels" \
					}, \
				"testing_images_path":{ \
					"type":"string", \
					"required":true, \
					"format":"file", \
					"description":"Zip file with testing images" \
					}, \
                "testing_labels_path":{ \
					"type":"string", \
					"required":true, \
					"format":"file", \
					"description":"Csv file with testing labels" \
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

It will read input parameters and pass them to the training module.

```python
import json
import os
import trainer  # Module with our code to train a neural network
from shutil import copyfile


# BatchX saves into /batchx/input/input.json what we passed to bx client when running the job
# So now we have to read input.json file to get a dictionary with the input parameters
with open("/batchx/input/input.json", "r") as input_file:
    input_json = json.loads(input_file.read())

# Generated files must be located below '/batchx/output/' folder. This folder has been automatically created by BatchX
output_folder = "/batchx/output/"

# The train method needs:
#     input_json_dict: a dictionary with the input parameters
#     output_folder: path of the folder where model and meta-data files will be saved
model_file_path, meta_file_path = trainer.train(input_json, output_folder)

# Additionally, we copy a script to use the trained model
copyfile('predictor.py', os.path.join(output_folder, 'predictor.py'))

# Write model and meta-data file paths into 'output.json'. BatchX will copy them into its FS. 
with open('/batchx/output/output.json', 'w+') as output_file:
    json.dump({'model_file_path': model_file_path, 'meta_file_path': meta_file_path,
               'predictor_file_path': os.path.join(output_folder, 'predictor.py')}, output_file)
```

3. trainer.py : python module to train the model

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os
import zipfile
import imageio
import pandas as pd


def read_zipped_images(zip_file_path):
    """Returns a numpy array representing all the images in a zip file."""
    zip_file = zipfile.ZipFile(zip_file_path)
    images_array = [imageio.imread(zip_file.read('{}.png'.format(i)))
                    for i in range(0, len(zip_file.namelist()))]
    return np.rollaxis(np.dstack(images_array), -1)


def train(input_json, output_folder):
    """Create a new model and returns h5 model and metadata files."""
    
    # Read data
    train_images = read_zipped_images(input_json['training_images_path'])
    train_labels = np.array(pd.read_csv(input_json['training_labels_path']).iloc[:, 0])
    test_images = read_zipped_images(input_json['testing_images_path'])
    test_labels = np.array(pd.read_csv(input_json['testing_labels_path']).iloc[:, 0])

    # Pre-process data
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Force using first GPU device
    with tf.device('/gpu:0'):

        # Build the model
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10)])

        # Compile the model
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

        # Train the model
        model.fit(train_images, train_labels, epochs=input_json['num_epochs'])

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

Now we can build the Docker image:

```bash
$ docker build -f ./docker/Dockerfile -t <docker_registry_username>/batchx-tensorflow-gpu-demo:latest .
```

Please note: you must change <docker_registry_username> by your Docker registry user name. 

Push to your Docker registry:

```bash
$ docker push <docker_registry_username>/batchx-tensorflow-gpu-demo:latest
```

Note: we're using a public registry, but a private one could be used instead.

# 2. Import the Docker image into BatchX

Import image:

```bash
$ bx import <docker_registry_username>/batchx-tensorflow-gpu-demo:latest
```

In order to see your imported image:

```bash
$ bx images
```

There should be an image named: tutorial/tensorflow-gpu-demo:1.0.0

# 3. Import a dataset of labeled clothing photos into BatchX

Our dataset consists of 4 files:

- training_images.zip: A file with 60000 png images. It will be used for training. Each image is named with a number: 0.png, 1.png, etc.
- testing_labels.csv: A file with 60000 labels (column "label"). Label in row 0 provides 0.png image type, etc. 
- testing_images.zip: A file with 10000 png images. It will be used for testing. Each image is named with a number: 0.png, 1.png, etc.
- testing_labels.zip: A file with 10000 labels (column "label"). Label in row 0 provides 0.png image type, etc. 

You can download them to your local folder:

```bash
$ wget 'https://github.com/josmaf/bx-tensorflow-demo/blob/master/data/training_images.zip'
$ wget 'https://github.com/josmaf/bx-tensorflow-demo/blob/master/data/training_labels.csv'
$ wget 'https://github.com/josmaf/bx-tensorflow-demo/blob/master/data/testing_images.zip'
$ wget 'https://github.com/josmaf/bx-tensorflow-demo/blob/master/data/testing_labels.csv'
```

And then copy them to BatchX file system:

```bash
$ bx cp training_* bx://data/
```

And

```bash
$ bx cp testing_* bx://data/
```

Please note that downloading these files is not required, as you could instead set URLs as parameters when running the BatchX image. 

That way BatchX would take care of downloading files and make them available for the image.

But downloading the data you'll be able to have a look at it and see what we're dealing with. 

As for instance, if you extract a png file from training_images.zip you'll see something like:

<img src="https://github.com/josmaf/bx-tensorflow-demo/blob/master/test/0.png"
     alt="Training image"
     style="width:72px; height:72px"/>

Or

<img src="https://github.com/josmaf/bx-tensorflow-demo/blob/master/test/1.png"
     alt="Training image"
     style="width:72px; height:72px"/>

# 4. Run your BatchX image

```bash
$ bx run -v=4 -m=15000 -g=1 -f=T4 tutorial/tensorflow-gpu-demo:1.0.0 '{ "training_images_path" : "bx://data/training_images.zip", "training_labels_path" : "bx://data/training_labels.csv", "testing_images_path" : "bx://data/testing_images.zip", "testing_labels_path" : "bx://data/testing_labels.csv", "num_epochs" : 10}'
```

Parameters:
- v=4     -> 4 vCPUs
- m=15000 -> At least 15 GB of RAM
- g=1     -> At least 1 GPU
- f=T4    -> GPU type (https://www.nvidia.com/en-gb/data-center/tesla-t4/)

If everything went ok, you should see something like:

```bash
[batchx] [2020/06/10 15:24:59] Job status: SUCCEEDED
{"model_file_path":"bx://jobs/127/output/model.h5","meta_file_path":"bx://jobs/127/output/model.info","predictor_file_path":"bx://jobs/127/output/predictor.py"}
```

# 5. Get results

Copy model binary file from BatchX to your local filesystem:

```bash
$ bx cp bx://jobs/<job_id>/output/model.h5 .
```

Please note you must set the correct value of <job_id>.

Copy predictor.py script:

```bash
$ bx cp bx://jobs/<job_id>/output/predictor.py .
```

You can test the model by downloading an input image:

```bash
$ wget 'https://raw.githubusercontent.com/josmaf/bx-tensorflow-demo/master/test/trousers.png'
```

Input image:

<img src="https://github.com/josmaf/bx-tensorflow-demo/blob/master/test/trousers.png"
     alt="Training image"/>

And then running the predictor.py script, supposing you have a Python environment with Tensorflow > 2.0.

The script will read the generated model (it must be located in the same folder) and return a prediction:

```bash
$ python predictor.py trousers.png
RESULT: Trouser
```
# Intro

What is BatchX for?

In the past few months I have heard this question several times.

The answer is simple: a platform to asynchronously run **software** processing **data** on the right **hardware** in a collaborative manner, on demand.

It sounds easy, but...

- **Software**: out software needs other software and somehow everything should be packaged so that portability and dependency management do not become a nightmare
- **Data**: usually, our software reads some data and writes more data as a result. And we want to be sure about a lot of stuff: Is out data kept safe? Is it available for all members of the team? Is it known what input data corresponds to what output?
- **Hardware**: sometimes hardware makes the difference. Do we need a machine with multiple GPUs to process different batches of data in parallel? And if so, do GPUs have enough memory to handle our workloads?

BatchX offers a single cloud solution to simplify dealing with these issues, so that we can focus on what really matters: our code.

Well, our code or other's code. BatchX allows us to run "code packages" (being specific, Docker images) previously created by other people. 

But in this 'toy example' we'll learn how to go from 'zero-to-hero': from creating our own Docker image to run it in BatchX and use the results.

So, let's do this. Everything starts with someone wanting to do something: suppose we want to train a neural network to identify a specific type of images.

"Training" is nothing but running a program that needs a lot of data. 

In this technical guide we'll get there through 4 + 1 steps:

1. Create a Docker image to train a model (a neural network) that classifies photos of clothing (Zalando's dataset, please see below)
2. Import the Docker image into BatchX 
3. Import a dataset of labeled clothing photos into BatchX
4. Train a model by running the imported Docker image in BatchX

Finally, we'll be able to get the trained model from BatchX file system and use it to classify a photo.

Please note: We'll use the fashion-MINST dataset provided by Zalando (https://github.com/zalandoresearch/fashion-mnist) as training data. 

# Prerequisites

- BatchX: we need to install the client and configure our account, as explained in https://docs.batchx.io/batchx-cli/installation
- Docker: we need to install Docker (https://docs.docker.com/get-docker/) and set up a Docker registry account (https://hub.docker.com/)
Why? Because as of today BatchX only allows to import images which are hosted in a cloud-based repository service, as hub.docker.com
- Python 3.6 & TensorFlow 2.x local installations: only necessary if we want to run the trained model in our local machine

# 1. Docker image creation

Ok, let's do this. We need a working directory to put the files we are about to create. Directory name is not important.

We'll have to write four files: 

1. **Dockerfile** : Docker image definition

In short, among other things it tells Docker:
 - Where to download a base image in top of which we're going to stack our code
 - First script (entrypoint) to execute when running the image 
 - BatchX manifest: it's in the LABEL field and it contains some info BatchX needs to process the Docker image, as the input and output data

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

2. **entrypoint.py** : script to act as a 'bridge' between BatchX and the 'trainer.py' script in charge of training the model

It will read input parameters and pass them to the training module.

```python
import json
import os
import trainer  # Module with our code to train a neural network
from shutil import copyfile


# BatchX saves what we passed to bx client into /batchx/input/input.json when running the job
# So now we have to read input.json file to get a dictionary with the input parameters
with open("/batchx/input/input.json", "r") as input_file:
    input_json = json.loads(input_file.read())

# Generated files must be located below '/batchx/output/' folder. This folder has been automatically created by BatchX
output_folder = "/batchx/output/"

# Train the model and get the path of the generated model file and a meta-data info file
model_file_path, meta_file_path = trainer.train(train_images_path=input_json['training_images_path'],
                                                train_labels_path=input_json['training_labels_path'],
                                                test_images_path=input_json['testing_images_path'],
                                                test_labels_path=input_json['testing_labels_path'],
                                                num_epochs=input_json['num_epochs'],
                                                output_folder=output_folder)

# Additionally, we copy a script to use the trained model
copyfile('predictor.py', os.path.join(output_folder, 'predictor.py'))

# Write model, meta-data and predictor script file paths into 'output.json'. BatchX will copy them into its file system 
with open('/batchx/output/output.json', 'w+') as output_file:
    json.dump({'model_file_path': model_file_path, 'meta_file_path': meta_file_path,
               'predictor_file_path': os.path.join(output_folder, 'predictor.py')}, output_file)
```

3. **trainer.py** : python module to train the model

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


def train(train_images_path, train_labels_path, test_images_path, test_labels_path, num_epochs, output_folder):
    """Create a new model and returns h5 model and metadata files."""
    
    # Read data
    train_images = read_zipped_images(train_images_path)
    train_labels = np.array(pd.read_csv(train_labels_path).iloc[:, 0])
    test_images = read_zipped_images(test_images_path)
    test_labels = np.array(pd.read_csv(test_labels_path).iloc[:, 0])

    # Pre-process data
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Use all GPU devices
    with strategy.scope():
        # Build the model
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
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

4. **predictor.py** : Python script to use the trained model and make predictions

```python
import os
import sys
import numpy as np
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set before importing tf
import tensorflow as tf
from tensorflow import keras

labels = {0: 'T-shirt/top',
          1: 'Trouser',
          2: 'Pullover',
          3: 'Dress',
          4: 'Coat',
          5: 'Sandal',
          6: 'Shirt',
          7: 'Sneaker',
          8: 'Bag',
          9: 'Ankle boot'}


def convert_image_to_array(image_path):
    img = Image.open(image_path).convert('L').resize((28, 28), Image.ANTIALIAS)
    return np.array([np.array(img) / 255.0])


def predict(image_path):
    # Load model
    model = keras.models.load_model("model.h5")
    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    # Predict
    img_array = convert_image_to_array(image_path)
    predicted_class = model.predict_classes(img_array)

    return labels[predicted_class[0]]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please enter an image file path")
        exit()
    label = predict(sys.argv[1])
    print("RESULT: {}".format(label))
```

Now we can build the Docker image:

```bash
$ docker build -f ./docker/Dockerfile -t <docker_registry_username>/batchx-tensorflow-gpu-demo:latest .
```

Please note that we must change <docker_registry_username> by our Docker registry user name. 

Push to our Docker registry:

```bash
$ docker push <docker_registry_username>/batchx-tensorflow-gpu-demo:latest
```

We're using a public repository for simplicity, but a private one could be used instead as long as it's reachable from the Internet.

# 2. Import the Docker image into BatchX

Import image:

```bash
$ bx import <docker_registry_username>/batchx-tensorflow-gpu-demo:latest
```

In order to see our imported image:

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

We can download them to our local folder:

```bash
$ wget 'https://github.com/josmaf/bx-tensorflow-demo/raw/master/data/training_images.zip'; \
wget 'https://raw.githubusercontent.com/josmaf/bx-tensorflow-demo/master/data/training_labels.csv'; \
wget 'https://github.com/josmaf/bx-tensorflow-demo/raw/master/data/testing_images.zip'; \
wget 'https://raw.githubusercontent.com/josmaf/bx-tensorflow-demo/master/data/testing_labels.csv'
```

And then copy them to BatchX file system:

```bash
$ bx cp training_* bx://data
```

And...

```bash
$ bx cp testing_* bx://data
```

Please note that downloading these files is not required, as we could instead set URLs as parameters when running the BatchX image. 

That way BatchX would take care of downloading files and make them available for the image.

But downloading the data we'll be able to have a look at it and see what we're dealing with. 

As for instance, if we extract a png file from training_images.zip we'll see something like:

<img src="https://github.com/josmaf/bx-tensorflow-demo/blob/master/test/0.png"
     alt="Training image"
     style="width:72px; height:72px"/>

Or

<img src="https://github.com/josmaf/bx-tensorflow-demo/blob/master/test/1.png"
     alt="Training image"
     style="width:72px; height:72px"/>

# 4. Train a model by running the imported Docker image in BatchX

Given that our image has already been imported into BatchX, we just have to run the following code:

```bash
$ bx run -v=4 -m=15000 -g=1 -f=T4 tutorial/tensorflow-gpu-demo:1.0.0 '{ "training_images_path" : "bx://data/training_images.zip", "training_labels_path" : "bx://data/training_labels.csv", "testing_images_path" : "bx://data/testing_images.zip", "testing_labels_path" : "bx://data/testing_labels.csv", "num_epochs" : 10}'
```

Parameters:
- v=4     -> 4 vCPUs
- m=15000 -> At least 15 GB of RAM
- g=1     -> At least 1 GPU
- f=T4    -> GPU type (https://www.nvidia.com/en-gb/data-center/tesla-t4/)

If everything went ok, we should see something like:

```bash
[batchx] [2020/06/10 15:24:59] Job status: SUCCEEDED
{"model_file_path":"bx://jobs/127/output/model.h5","meta_file_path":"bx://jobs/127/output/model.info","predictor_file_path":"bx://jobs/127/output/predictor.py"}
```

Ok, what do we have here? Exactly what we told BatchX in the manifest file to be returned as output. That is, a json file with three fields:

- model_file_path: path of trained model in BatchX file system
- meta_file_path: path of meta-info text file in BatchX file system
- predictor_file_path: path in BatchX file system of Python script file able to run the trained model

Every new job will have its own folder: bx://jobs/<job_id>, being <job_id> an auto-generated unique number for that job.

At this point, anyone in our team (in case we use the same environment) can easily see past jobs as well as the input data and results of each execution.

# Get the trained model from BatchX file system and use it to classify a photo

A trained model by itself is just a binary file. 

If we want to see it in action, we need a script, and a Python working environment (in case you to install it: https://wiki.python.org/moin/BeginnersGuide/Download)

The job we just executed provide us with both files: the trained model and the script.

We first copy the model binary file from BatchX to our local filesystem:

```bash
$ bx cp bx://jobs/<job_id>/output/model.h5 .
```

Please note you must set the correct value of <job_id>.

Copy predictor.py script:

```bash
$ bx cp bx://jobs/<job_id>/output/predictor.py .
```

You can test the model by downloading an input image to a folder:

```bash
$ wget 'https://raw.githubusercontent.com/josmaf/bx-tensorflow-demo/master/test/trousers.png'
```

<img src="https://github.com/josmaf/bx-tensorflow-demo/blob/master/test/trousers.png"
     alt="Training image"/>

And then running the predictor.py script in the same folder, for which we need a Python environment with TensorFlow greater or equal than 2.0.

The script will read the generated model file and return a prediction:

```bash
$ python predictor.py trousers.png
RESULT: Trouser
```

Of course, we could instead create a Docker image to run a BatchX job in charge of doing exactly this: download images from the internet, read the generated model and predict image types, so that we could create a pipeline composed by both jobs able to download and classify images on a massive scale.
 
But that would be material for other post... Hope this helps you to better understand what BatchX is building! :-)

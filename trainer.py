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

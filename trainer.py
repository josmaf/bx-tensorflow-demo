import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os.path


def train(input_file_path, num_epochs, output_folder):

    # Read data
    data = np.load(input_file_path)
    train_images = data['x_train']
    train_labels = data['y_train']
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


if __name__ == '__main__':
    train('/home/jose.fernandez/projects/batchx/demo-tensorflow-gpu/fashion_mnist.npz', 10, '.')

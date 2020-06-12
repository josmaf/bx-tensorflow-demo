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

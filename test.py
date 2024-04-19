import numpy as np
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# load in the model
model = tf.keras.models.load_model('handwritten model.keras')

image_number = 1

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Define the desired image size
new_height = 28
new_width = 28


# Iterate over the test images
while os.path.isfile(f"TestData/Test{image_number}.png"):
    try:
        # Read and resize the image
        img = cv2.imread(f"TestData/Test{image_number}.png", cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (new_width, new_height))

        # Preprocess the image
        img_resized = np.invert(np.array([img_resized]))

        # Make prediction
        prediction = model.predict(img_resized)
        predicted_letter = letters[np.argmax(prediction)]
        print(f"The predicted letter is {predicted_letter}")

        # Display the resized image
        plt.imshow(img_resized[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        image_number += 1
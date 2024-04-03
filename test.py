import numpy as np
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# load in the model
model = tf.keras.models.load_model('handwritten model.keras')

image_number = 1

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# runs the .png files in the test folder as desired input for model evaluation
while os.path.isfile(f"TestData/Test{image_number}.png"):
    try:
        img = cv2.imread(f"TestData/Test{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        predicted_letter = letters[np.argmax(prediction)]
        print(f"The predicted letter is {predicted_letter}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        image_number += 1
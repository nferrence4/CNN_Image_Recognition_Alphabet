import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from imblearn.under_sampling import NearMiss
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load and preprocess the dataset
df = pd.read_csv('A_Z Handwritten Data.csv')
y = df['0']
del df['0']
x = y.replace(list(range(26)), ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
nM = NearMiss()
X_data, y_data = nM.fit_resample(df, y)
y = to_categorical(y_data)
num_classes = y.shape[1]
X_data = X_data / 255
X_data = np.array(X_data)
X_data = X_data.reshape(-1, 28, 28, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, random_state=102)

# Define the CNN architecture
model = models.Sequential([
    layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(26, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Change to categorical_crossentropy
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

model.save('handrwitten model.keras')
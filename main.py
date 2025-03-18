import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 # computer vision (load and process images)

mnist = tf.keras.datasets.mnist # 28x28 images of handwritten digits 0-9
(X, y), (_, _) = mnist.load_data()

# Define the split ratio
split_ratio = 1
split_index = int(len(X) * split_ratio) # find the index to split at

# manually split the data
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

#scale down the data from 255 to btw 0 and 1
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # convert the grid of 28*28 pixels into a 1D array off 28*28 ellements
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
# what softmax do is that its gives  us a probability distribution over the 10 possible output, the total sum of each probability is 1.
model.add(tf.keras.layers.Dense(10, activation='softmax')) # 10 neurons at the output layer cause there are 10 possible output from 0 to 9


# optimizer: (adaptive moment estimation) adam is a advance gradient descent algorithm, adjust the learning rate for each parameter automaticly
# loss: categorical_crossentropy is a loss function that measures how well the model is performing
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(X_train, y_train, epochs=20)

model.save("handwritten_digit_recognition.model")

model = tf.keras.models.load_model("handwritten_digit_recognition.model")

loss, accuracy = model.evaluate(X_test, y_test)
print(loss)
print(accuracy)

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1

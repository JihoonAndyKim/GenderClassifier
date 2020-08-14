import os
import time
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras import datasets, layers, models, initializers
from sklearn.metrics import confusion_matrix

root_dir = os.getcwd()

print("Loading the test data")
with open(root_dir + '/pickle/test.pickle', 'rb') as b:
    feature_test = pickle.load(b)
with open(root_dir + '/pickle/test_l.pickle', 'rb') as b:
    test_labels = pickle.load(b)

t = feature_test[0]
for i in range(1, len(feature_test)):
    t = np.vstack((t, feature_test[i]))
feature_test = t
test_labels = np.array([d for d in test_labels]).flatten()

print("Loading the model from checkpoint")
gender_model = models.Sequential()

gender_model.add(layers.Dense(16, activation = "relu"))
gender_model.add(layers.Dense(16, activation = "relu"))
gender_model.add(layers.Dropout(0.4))
gender_model.add(layers.Dense(1, activation = "sigmoid"))

gender_model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Get the latest checkpoint
checkpoint_dir = root_dir + "/checkpoints"
latest = tf.train.latest_checkpoint(checkpoint_dir)

# Loading in the weights
gender_model.load_weights(latest)

print("Inferencing...")
#Performing inference on the test set
pred = [1 if l > 0.5 else 0 for l in gender_model.predict(feature_test)]
test_loss, test_acc = gender_model.evaluate(feature_test,  test_labels, verbose=2)

tn, fp, fn, tp = confusion_matrix(pred, test_labels).ravel()
print("F1 Score: %.4f" % (tp / (tp + 0.5 * (fp + fn))))
print("Accuracy: %.2f%%" % ((tp + tn)/(tn + fp + fn + tp) * 100))

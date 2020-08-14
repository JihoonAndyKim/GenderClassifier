import os
import pathlib
import math
import time
import pickle
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import scipy.io

from tensorflow.keras import datasets, layers, models, initializers
from sklearn.metrics import confusion_matrix
from PIL import Image
from PIL import ImageOps
from vgg_face_descriptor import vgg_face_descriptor

# Loading in the current working directory
root_dir = os.getcwd()

#Load in the VGG face descriptor model
model, image_size, descriptions = vgg_face_descriptor(root_dir)

start = time.time()
print("Loading in the image data")

gender_dir = pathlib.Path(root_dir + "/combined")

# Size of the dataset to train on
num_of_images = len([item for item in tf.data.Dataset.list_files(str(gender_dir/"*/*/*"))])

# Batch size for training along with the minimum number of steps needed to cover whole dataset
batchSize = 256
steps = math.ceil(num_of_images/batchSize)

# Lambda function for loading in the image into tensors
def getImage(x, l):
    label = 0 if l == "M" else 1
    img = tf.io.read_file(x)
    img = tf.image.decode_jpeg(img, channels=3) #color images
    img = tf.image.resize(img, [image_size[0], image_size[0]], method=tf.image.ResizeMethod.BICUBIC)
    return img, label

# Male faces dataset
male_files = tf.data.Dataset.list_files(str(gender_dir/"*/*_M/*"))
male = male_files.map(lambda x: getImage(x, "M"))

# Female faces dataset
female_files = tf.data.Dataset.list_files(str(gender_dir/"*/*_F/*"))
female = female_files.map(lambda x: getImage(x, "F"))

# Concatenate the datasets and interleave them
dataset = tf.data.Dataset.zip((male, female)).flat_map(
    lambda x, y: male.from_tensors(x).concatenate(female.from_tensors(y)))


dataset = dataset.shuffle(1000)
dataset = dataset.batch(batchSize)
dataset = dataset.prefetch(1000)

# Load in the data
data = []
labels = []
for _ in range(steps):
    batch, l = next(iter(dataset))
    data.append(batch)
    labels.append(l)

# Slice into train, val, and test sets
data_n = len(data)
train = data[:round(0.75 * data_n)]
val = data[round(0.75 * data_n) : round(0.875 * data_n)]
test = data[round(0.875 * data_n):]

train_labels = labels[:round(0.75 * data_n)]
val_labels = labels[round(0.75 * data_n) : round(0.875 * data_n)]
test_labels = labels[round(0.875 * data_n):]

print("---- %f seconds ----" % (time.time() - start))
print("Inferencing on the VGG data")
start = time.time()

feature_train = []
for i, batch in enumerate(train):
    if i % 10 == 0:
        print("Loaded %d batches" % (i))
    feature_train.append(model.predict(batch.numpy()))

feature_val = []
for i, batch in enumerate(val):
    if i % 10 == 0:
        print("Loaded %d batches" % (i))
    feature_val.append(model.predict(batch.numpy()))

feature_test = []
for i, batch in enumerate(test):
    if i % 10 == 0:
        print("Loaded %d batches" % (i))
    feature_test.append(model.predict(batch.numpy()))

# OPTIONAL: Pickling for fast loading and unloading of inferenced
#           data to read.
#
# WRITING:
# with open('train.pickle', 'wb') as b:
#     pickle.dump(feature_train,b)
# with open('val.pickle', 'wb') as b:
#     pickle.dump(feature_val,b)
# with open('test.pickle', 'wb') as b:
#     pickle.dump(feature_test,b)
#
# READING
# with open('train.pickle', 'rb') as b:
#     feature_train = pickle.load(b)
# with open('val.pickle', 'rb') as b:
#     feature_val = pickle.load(b)
# with open('test.pickle', 'rb') as b:
#     feature_test = pickle.load(b)


# Collating validation set into one tensor of size
# (BATCHSIZE * STEPS * 0.125) x 2622
v = feature_val[0]
for i in range(1, len(feature_val)):
    v = np.vstack((v, feature_val[i]))
feature_val = v
val_labels = np.array([d for d in val_labels]).flatten()

# Collating test set into one tensor of size
# (BATCHSIZE * STEPS * 0.125) x 2622
t = feature_test[0]
for i in range(1, len(feature_test)):
    t = np.vstack((t, feature_test[i]))
feature_test = t
test_labels = np.array([d for d in test_labels]).flatten()

print("Number of training batches: ", len(feature_train))
print("Each training batch tensor shape: ", feature_train[0].shape)

print("Validation set shape: ", feature_val.shape)
print("Test set shape: ", feature_test.shape)

print("---- %f seconds ----" % (time.time() - start))
print("Creating the classifier")
start = time.time()

gender_model = models.Sequential()

gender_model.add(layers.Dense(16, activation = "relu"))
gender_model.add(layers.Dense(16, activation = "relu"))
gender_model.add(layers.Dropout(0.4))
gender_model.add(layers.Dense(1, activation = "sigmoid"))

gender_model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Initializing variables for saving training results
loss, accuracy, val_accuracy, val_loss = [], [], [], []

# Variables for early stopping.
# Tolerance is based on loss and the patience is for 2 epochs.
tolerance = 1e-2
patience = 2
stop_count = 0
prev_loss = None


print("---- %f seconds ----" % (time.time() - start))
print("Training the classifier")
start = time.time()

# Checkpoint directory
if not os.path.exists(root_dir + "/checkpoints"):
    os.makedirs(root_dir + "/checkpoints")

# TRAINING LOOPS
for epoch in range(30):
    print(("Epoch: ", epoch))
    val_loss_sum = 0

    # Train using the batches and store losses and accuracy
    for i, batch in enumerate(feature_train):
        if i % 40 == 0:
            print("Batch Number:, ", i)

        history = gender_model.fit(batch,
                                   train_labels[i],
                                   verbose = 0,
                                   validation_data = (feature_val, val_labels))
        loss.append(history.history['loss'])
        accuracy.append(history.history['accuracy'])
        val_accuracy.append(history.history['val_accuracy'])
        val_loss.append(history.history['val_loss'])

        val_loss_sum += history.history['val_loss'][0]


    avg_val_loss = val_loss_sum / len(feature_train)
    print(avg_val_loss)

    #Checkpointing
    gender_model.save_weights('./checkpoints/ck-epoch%d' % (epoch))

    # Early stopping if the validation loss does not improve after
    # patience amount of epochs
    if stop_count == patience:
        break

    if not prev_loss:
        prev_loss = avg_val_loss
    elif prev_loss >= avg_val_loss:
        prev_loss = avg_val_loss
        stop_count = 0
    else:
        stop_count += 1

# Model architecture
gender_model.summary()
#tf.keras.utils.plot_model(gender_model, to_file="architecture.png", show_shapes=True)

# Plotting
plt.figure()
plt.plot(accuracy, label='accuracy')
plt.plot(val_accuracy, label='val_accuracy')
plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
plt.xlabel('Batches')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
print("Accuracy and loss over time for train/val sets")

pred = [1 if l > 0.5 else 0 for l in gender_model.predict(feature_test)]
plt.figure(figsize=(20,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
    plt.subplot(6,5,n+1)
    plt.imshow(test[0].numpy().astype(np.uint8)[n])
    if pred[n] == 0:
        label = "male"
    else:
        label = "female"
    plt.title(label)
    plt.axis('off')
    _ = plt.suptitle("GenderNet predictions")
plt.savefig("examples.png")

test_loss, test_acc = gender_model.evaluate(feature_test,  test_labels, verbose=2)

print("Validation Error: %.2f%%" % ((1 - val_accuracy[-1][0]) * 100))
print("Train Error: %.2f%%" % ((1 - accuracy[-1][0]) * 100))
print("Test Error: %.2f%%" % ((1 - test_acc) * 100))

tn, fp, fn, tp = confusion_matrix(pred, test_labels).ravel()
print("F1 Score: %.4f" % (tp / (tp + 0.5 * (fp + fn))))
print("Accuracy: %.2f%%" % ((tp + tn)/(tn + fp + fn + tp) * 100))

import time
import math
import numpy as np
import tensorflow as tf
import scipy.io

from tensorflow.keras import datasets, layers, models, initializers

def vgg_face_descriptor(root_dir):
    start = time.time()
    print("Loading in the data")

    # Loading in weights from matlab
    matlab_weights = scipy.io.loadmat(root_dir + '/vgg_face.mat')

    # Open the dictionary containing the data
    net = matlab_weights['net']
    normalization = net['normalization']
    vgg_layers = net['layers']
    classes = net['classes']

    # Parameters from normalization
    image_size = normalization[0][0][0]['imageSize'][0][0]
    average_image = normalization[0][0][0]['averageImage'][0]

    # Parameters from classes
    descriptions = np.array([d[0] for d in classes[0][0][0]['description'][0].flatten()])

    print("---- %f seconds ----" % (time.time() - start))
    print("Building VGG Face Detector Model")
    start = time.time()
    # Initialize model
    model = models.Sequential()

    # Building the model layer by layer
    for ind, layer in enumerate(vgg_layers[0][0][0]):

        # Extract layer type and name
        layer_type = layer['type'][0][0][0]
        layer_name = layer['name'][0][0][0]

        # If it is a convolution Layer
        if layer_type == "conv":
            stride = layer['stride'][0][0][0][0]
            pad = layer['stride'][0][0][0][0]

            # Kernel weights and bias
            W_matrix = initializers.Constant(layer['weights'][0][0][0][0])
            b = initializers.Constant(layer['weights'][0][0][0][1])
            shape = layer['weights'][0][0][0][0].shape

            filter_size = shape[0:2]
            num_filt = shape[3]

            # As per original paper, fully connected layers
            # have valid padding
            if "fc" in layer_name:
                padding = "valid"
            else:
                padding = "same"

            # First layer needs input shape
            if not ind:
                model.add(layers.Conv2D(filters = num_filt,
                                    kernel_size = filter_size,
                                    input_shape = image_size,
                                    strides = (stride, stride),
                                    padding = padding,
                                    activation = "relu",
                                    kernel_initializer = W_matrix,
                                    bias_initializer = b))

            else:
                model.add(layers.Conv2D(filters = num_filt,
                                    kernel_size = filter_size,
                                    strides = (stride, stride),
                                    padding = padding,
                                    activation = "relu",
                                    kernel_initializer = W_matrix,
                                    bias_initializer = b))
        # Max pool layer with stride 2 and pool size 2
        elif layer_type == "pool":
            model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='same'))
        # Dropout layer with 50% rate
        elif layer_type == "dropout":
            model.add(layers.Dropout(0.5))
        # Final softmax output
        elif layer_type == "softmax":
            model.add(layers.Flatten())
            model.add(layers.Softmax())

    print("---- %f seconds ----" % (time.time() - start))
    print("Compiling and building VGG Face Detector Model")
    start = time.time()

    # Compile and build the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.build(input_shape = image_size)
    print("---- %f seconds ----" % (time.time() - start))

    return model, image_size, descriptions

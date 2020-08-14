# Gender Classifier using Transferred Learning from VGG Face Descriptor model
### Andy Kim - 8/13/20


### Requisites
Please take a look at the notebook at Gender_Classification_Challenge.ipynb for a step by step walkthrough of the classification code. Gender_Classification_Challenge.html is the HTML form of the notebook.

Please ensure tensorflow, sklearn and scipy.io are all installed in the python environment.





### Files
    
    - ak.jpg                                       Test image for model
    - architecture.png                             Tensorflow classifier model architecture trained on the gender dataset
    - checkpoints/                                 Contains all of the weights of the Tensorflow classifier model trained on the 
                                                 gender dataset. Use gender_classifier.py to load in the weights
    - combined/                                    Contains all of the image training data
    - gender_classifier.py                         Code for model evaluation. Contains results and metrics if run.
    - gender_classifier_train.py                   Code for model training and evaluation
    - Gender_Classification_Challenge.html         HTML form of the python notebook
    - Gender_Classification_Challenge.ipynb        Step-by-step walkthrough of model code. Also contains results and metrics
    - results.jpg                                  Training and validation accuracy and loss plot over training
    - predictions.jpg                              Sample of 30 faces from the dataset and the resulting predictions
    - pickle/                                      Contains pickled form of the test set for model evaluation
    - vgg_face.mat                                 VGG Face Descriptor model weights
    - vgg_face_descriptor.py                       Helper function for loading VGG Face Descriptor model
    
    
    
### Running the code:
(Ensure vgg_face.mat, combined/ are in the root directory)

        1. Run through all the cells in Gender_Classification_Challenge.ipynb
    
        OR 
    
        2.1. Run the following command to train (Also will perform evaluation)
            ```$ python gender_classifier_train.py```
        2.2. If only evaluation, then run the following command then to evaluate the model
            ```$ python gender_classifier.py``` 
            
    
    
### Results:
- Classification accuracy on hold out test set of ~94.3% with an F1 score of 0.944
- Please take a look at the results in Gender_Classification_Challenge.ipynb



### Steps Taken to solve the problem:

    1. Loaded in vgg_face.mat file into Tensorflow 2 and then pulled all architecture weights and transferred to Tensorflow
    2. Compiled and built the model for inferencing
    3. Loaded in image data (32k images) from https://s3.amazonaws.com/matroid-web/datasets/agegender_cleaned.tar.gz
    4. Inferenced all the images from the above data into pre-trained vgg_model to produce a (32k x 2622) dataset
    5. Created a second neural network classifier in tensorflow
        - Input is a 2622 element array
        - 2 hidden fully-connected layer NN with 16 activation units
        - sigmoid activation output
        - dropout layer in between with a 40% drop rate
    6. Inferenced test sets to evaluate in the second model for gender



### Citations:

VGG Face Descriptor - http://www.robots.ox.ac.uk/~vgg/software/vgg_face/
Face dataset - https://s3.amazonaws.com/matroid-web/datasets/agegender_cleaned.tar.gz
Loading in .mat file snippets - https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html
Tensorflow Docs for code snippets used for loading and training - https://www.tensorflow.org/api_docs
Saving and loading models code snippets - https://www.tensorflow.org/tutorials/keras/save_and_load

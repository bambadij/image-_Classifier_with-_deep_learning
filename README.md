﻿# Developing an Image Classifier with Deep Learning
In this first part of the project, you'll work through a Jupyter notebook to implement an image classifier with PyTorch.

# Part 2 - Building the command line application
Now that you've built and trained a deep neural network on the flower data set, it's time to convert it into an application that others can use. Your application should be a pair of Python scripts that run from the command line. For testing, you should use the checkpoint you saved in the first part.

## Train
-Train a new network on a data set with:
        
        train.py
-Basic usage: 
        
        python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
-Options: * Set directory to save checkpoints:
    
        .python train.py data_dir --save_dir save_directory 
-Choose architecture:

    .python train.py data_dir --arch "vgg13" 
-Set hyperparameters: 

    .python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 
- Use GPU for training: 

      .python train.py data_dir --gpu

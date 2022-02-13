# Don't let tensorflow see my GPU because it doesn't work properly on WSL lol
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from glob import glob

import numpy
from tensorflow import keras
from keras.constraints import maxnorm
from keras.utils import np_utils

char1_list = [
    "AN",
    "BW",
    "BB",
    "CE",
    "DB",
    "EL",
    "FI",
    "FU",
    "MF",
    "PW",
    "PS",
    "PC",
    "RF",
    "SQ",
    "UM",
    "VA"
]

if __name__ == '__main__':

    # Size of each sample image
    char1_ysize = 80
    char1_xsize = 120

    # Set seed for reproducibility purposes
    seed = 69
    
    # Load the dataset from transformed_char1_data
    total_samples = sum([len(glob(f"transformed_char1_data/{char1}/*.jpg")) for char1 in char1_list])
    print(f"Found {total_samples} samples in total")

    # Normalize each image from 0-255 to between 0-1 by dividing by 255
    
    # The model is eventually going to have 1 neuron for each character
    class_num = len(char1_list)

    # model = keras.Sequential()
    # model.add(keras.layers.Dense(32, activation='relu'))
    # model.add(keras.layers.Dropout(0.2))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dense(class_num, activation='softmax'))

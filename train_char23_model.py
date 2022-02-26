from glob import glob
from pprint import pprint

import numpy as np
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


char23_list = ["AN","BB","BW","CE","DB","EL","FI","FU","MF","N","PC","PS","PW","RF","SQ","UM","VA"]

if __name__ == '__main__':

    print("Initial setup...")

    # Size of each sample image
    img_height, img_width = 12, 48

    epochs = 30
    batch_size = 16

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_height, img_width)
    else:
        input_shape = (img_height, img_width, 3)

    # Set seed for reproducibility
    np.random.seed(69)

    print("Creating model...")

    # Create the model
    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
     
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(char23_list), activation='softmax'))

    print("Compiling model...")

    model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
    )

    print("Creating training data generator...")

    # Training data
    train_dir = 'char23_train'
    num_train_samples = len(glob(f"{train_dir}/**/*.jpg"))
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        color_mode="rgb",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    print("Creating test data generator...")

    # Testing data
    test_dir = 'char23_test'
    num_test_samples = len(glob(f"{test_dir}/**/*.jpg"))
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        color_mode="rgb",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    print("Training model...")

    # Train the model on the data
    model.fit(
        x=train_generator,
        steps_per_epoch=num_train_samples // batch_size,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=num_test_samples // batch_size
    )

    print("Saving model...")

    model.save('char23_model.h5')

    print("Finished.")
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from numpy import argmax
from keras.models import load_model
from glob import glob
import cv2 as cv

from character_list import char1_list

if __name__ == '__main__':
    
    img_height, img_width = 80, 120
    batch_size = 16
    test_dir = 'char1_test'
    model = load_model('char1_model.h5')

    np.random.seed(69)

    # Bulk manual approach
    # Very slow, but good practice in handling the data
    for char1 in char1_list:
        print(f"Running tests for {char1}... ", end="")
        sample_files = sorted(glob(f"{test_dir}/{char1}/*.jpg"))
        total_samples = 0
        successes = 0
        for filename in sample_files:
            total_samples += 1
            image = cv.imread(filename)
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            img_array = np.array(image)
            img_array = img_array.astype('float32')
            img_array = img_array / 255.0
            img_array = img_array.reshape(1,img_height,img_width,1)
            output_labels = model.predict(img_array)[0]
            output_prediction = char1_list[argmax(output_labels)]
            if output_prediction == char1:
                successes += 1
        print(f"{float(successes*100/total_samples):.2f}")

    # Automatically evaluating using generators and built in evaluate()    
    # num_test_samples = len(glob(f"{test_dir}/**/*.jpg"))
    # test_datagen = ImageDataGenerator(rescale=1. / 255)
    # test_generator = test_datagen.flow_from_directory(
    #     test_dir,
    #     color_mode="grayscale",
    #     target_size=(img_height, img_width),
    #     batch_size=1,
    #     class_mode='categorical'
    # )
    # scores = model.evaluate(x=test_generator)
    # print(f"Accuracy = {scores[1]*100:.2f}%")

    # Semi manual approach using a generator but NOT its evaluate() function
    # successes = 0
    # total_cases = 97
    # for i in range(total_cases):
    #     sample = test_generator.next()
    #     print(f"predicting case {i+1}:", end="")
    #     img_data = sample[0]
    #     expected_labels = sample[1][0]
    #     expected_prediction = char1_list[argmax(expected_labels)]
    #     print(f"Expected {expected_prediction}, ", end="")
    #     # img_array = np.array(sample[0])
    #     # img_array = img_array.reshape(img_height,img_width,1)
    #     # img_array *= 255
    #     # cv.imwrite(f"sample.jpg", img_array)
    #     output_labels = model.predict(img_data)[0]
    #     # print(output_labels)
    #     output_prediction = char1_list[argmax(output_labels)]
    #     print(f"got {output_prediction}")
    #     if expected_prediction == output_prediction:
    #         successes += 1
    # print(f"Semi-manual testing: {successes}/{total_cases} success rate")






from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from numpy import argmax
from keras.models import load_model
from glob import glob
import cv2 as cv

char23_list = ["AN","BB","BW","CE","DB","EL","FI","FU","MF","N","PC","PS","PW","RF","SQ","UM","VA"]

if __name__ == '__main__':
    
    img_height, img_width = 12, 48
    batch_size = 4
    test_dir = 'char23_test'
    model = load_model('char23_model.h5')

    np.random.seed(69)

    # Bulk manual approach
    # Very slow, but good practice in handling the data
    for char23 in char23_list:
        print(f"Running tests for {char23}... ", end="")
        sample_files = sorted(glob(f"{test_dir}/{char23}/*.jpg"))
        total_samples = 0
        successes = 0
        for filename in sample_files:
            total_samples += 1
            image = cv.imread(filename)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            img_array = np.array(image)
            img_array = img_array.astype('float32')
            img_array = img_array / 255.0
            img_array = img_array.reshape(1,img_height,img_width,3)
            output_labels = model.predict(img_array)[0]
            output_prediction = char23_list[argmax(output_labels)]
            if output_prediction == char23:
                successes += 1
        print(f"{float(successes*100/total_samples):.2f}")

    # Automatically evaluating using generators and built in evaluate()    
    # num_test_samples = len(glob(f"{test_dir}/**/*.jpg"))
    # test_datagen = ImageDataGenerator(rescale=1. / 255)
    # test_generator = test_datagen.flow_from_directory(
    #     test_dir,
    #     color_mode="rgb",
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
    #     expected_prediction = char23_list[argmax(expected_labels)]
    #     print(f"Expected {expected_prediction}, ", end="")
    #     # img_array = np.array(sample[0])
    #     # img_array = img_array.reshape(img_height,img_width,1)
    #     # img_array *= 255
    #     # cv.imwrite(f"sample.jpg", img_array)
    #     output_labels = model.predict(img_data)[0]
    #     # print(output_labels)
    #     output_prediction = char23_list[argmax(output_labels)]
    #     print(f"got {output_prediction}")
    #     if expected_prediction == output_prediction:
    #         successes += 1
    # print(f"Semi-manual testing: {successes}/{total_cases} success rate")






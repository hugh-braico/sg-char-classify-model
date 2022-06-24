import cv2 as cv
from glob import glob
from random import seed, random
import pathlib

from character_list import char1_list

# Take a small amount of 720p screenshots for each character and generate a 
# large dataset from it by varying blur and crop.
if __name__ == '__main__':

    char1_y1 = 20
    char1_y2 = 100
    char1_x1 = 30
    char1_x2 = 150

    # Set RNG seed for reproducibility
    seed(69)


    for char_name in char1_list:

        print(f"Processing data for {char_name}...")
        raw_files = sorted(glob(f"raw_char1_data/{char_name}/*.png"))
        num_samples = 0
        num_raw_files = len(raw_files)

        # Make output folders if they don't exist
        test_dir  = f"char1_test/{char_name}"
        train_dir = f"char1_train/{char_name}"
        pathlib.Path(test_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(train_dir).mkdir(parents=True, exist_ok=True)

        for file_num, raw_file in enumerate(raw_files):
            image = cv.imread(raw_file)

            # Expand training dataset to reduce overfitting
            # Simulate small variance in crop accuracy
            for xcrop in range(-3, 4, 3):
                for ycrop in range(-3, 4, 3):
                    cropped_image = image[
                        (char1_y1+ycrop):(char1_y2+ycrop), 
                        (char1_x1+xcrop):(char1_x2+xcrop)
                    ]
                    # Simulate small amounts of blurring in different directions
                    for ksizex in range (1, 7, 5):
                        for ksizey in range (1, 7, 5):
                            ksize = (ksizey, ksizex)
                            blurred_image = cv.blur(cropped_image, ksize)
                            mono_image = cv.cvtColor(blurred_image, cv.COLOR_BGR2GRAY)
                            # Put 20% of the data into test rather than train
                            if random() < 0.2:
                                cv.imwrite(f"{test_dir}/{file_num+1}_{xcrop}_{ycrop}_{ksizex}_{ksizey}.jpg", mono_image)
                            else: 
                                cv.imwrite(f"{train_dir}/{file_num+1}_{xcrop}_{ycrop}_{ksizex}_{ksizey}.jpg", mono_image)
                            num_samples += 1

        print(f"generated {num_samples} sample images from {num_raw_files} raw files")

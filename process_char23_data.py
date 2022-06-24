import cv2 as cv
from glob import glob
from random import seed, random
import pathlib

from character_list import char23_list

# Take a small amount of 720p screenshots for each character and generate a 
# large dataset from it by varying blur and crop.
if __name__ == '__main__':

    # Char mini portraits are roughly 48x12
    char2_y1 = 70
    char2_y2 = 82
    char2_x1 = 172
    char2_x2 = 220

    # Set RNG seed for reproducibility
    seed(69)

    for char_name in char23_list:
        print(f"Processing data for {char_name}...")
        raw_files = sorted(glob(f"raw_char2_data/{char_name}/*.png"))
        num_samples = 0
        num_raw_files = len(raw_files)

        # Make output folders if they don't exist
        test_dir  = f"char23_test/{char_name}"
        train_dir = f"char23_train/{char_name}"
        pathlib.Path(test_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(train_dir).mkdir(parents=True, exist_ok=True)

        for file_num, raw_file in enumerate(raw_files):
            image = cv.imread(raw_file)
            # Simulate small variance in crop accuracy
            for xcrop in range(-2, 3, 1):
                for ycrop in range(-2, 3, 1):
                    cropped_image = image[
                        (char2_y1+ycrop):(char2_y2+ycrop), 
                        (char2_x1+xcrop):(char2_x2+xcrop)
                    ]
                    # Simulate small amounts of blurring in different directions
                    for ksizex in range (1, 4, 1):
                        for ksizey in range (1, 4, 1):
                            ksize = (ksizey, ksizex)
                            blurred_image = cv.blur(cropped_image, ksize)
                            if random() < 0.2:
                                cv.imwrite(f"{test_dir}/{file_num+1}_{xcrop}_{ycrop}_{ksizex}_{ksizey}.jpg", blurred_image)
                            else: 
                                cv.imwrite(f"{train_dir}/{file_num+1}_{xcrop}_{ycrop}_{ksizex}_{ksizey}.jpg", blurred_image)
                            num_samples += 1

        print(f"generated {num_samples} sample images from {num_raw_files} raw files")

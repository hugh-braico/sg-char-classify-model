import cv2 as cv
import numpy as np
from glob import glob
from pprint import pformat

if __name__ == '__main__':

    GAME_SIZE = 1093  
    y1 = int(GAME_SIZE*0.0734)
    y2 = y1 + 1

    # "Outer" bars
    # p1x1 = int(GAME_SIZE*0.146)
    # p1x2 = int(GAME_SIZE*0.225)

    # "Inner" bars
    p1x1 = int(GAME_SIZE*0.330)
    p1x2 = int(GAME_SIZE*0.409)

    p2x1 = GAME_SIZE - p1x2
    p2x2 = GAME_SIZE - p1x1

    # Get every single green
    greens_list = []
    for sample_file in glob("green_bars_samples/*.jpg"):
        image = cv.imread(sample_file)
        p1healthbar = image[y1:y2, p1x1:p1x2]
        p2healthbar = image[y1:y2, p2x1:p2x2]
        p1avg = p1healthbar.mean(axis=0).mean(axis=0)
        p2avg = p2healthbar.mean(axis=0).mean(axis=0)
        # print(f"{sample_file}: {p1avg}, {p2avg}")
        greens_list.append(p1avg)
        greens_list.append(p2avg)

    # Convert from dynamic list to array so we can use mean()
    greens_array = np.array(greens_list)
    # print(f"greens_list: {pformat(greens_list)}")

    # Print the mean of all those greens
    avg_green = greens_array.mean(axis=0)
    print(f"Mean of all greens: {avg_green}")

    # Find the greatest deviation between any sample and the mean
    max_error = 0.0
    error_sum = 0.0
    for green in greens_list:
        error = np.linalg.norm(green - avg_green)
        if error > max_error: 
            max_error = error
        error_sum += error

    print(f"Largest deviation: {max_error}")
    print(f"Average deviation: {error_sum / len(greens_list)}")
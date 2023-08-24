from pathlib import Path

from cv2 import cvtColor
from cv2 import imread, imwrite
from cv2 import COLOR_BGR2GRAY

from numpy import ndarray

from pycvl import *


ROOT_PROJECT_PATH = Path(__file__) / '..'
RESOURCES_DIR_PATH = ROOT_PROJECT_PATH / 'resources'


def main():
    test_img_path = RESOURCES_DIR_PATH / 'test_frame_1.jpeg'
    test_img_path_str = str(test_img_path)
    frame = imread(test_img_path_str)

    # Gray test
    gray_frame_1 = cvtColor(frame.copy(), COLOR_BGR2GRAY)
    imwrite('/tmp/gray_frame_1.jpeg', gray_frame_1)
    gray_frame_2: ndarray = grayscale(frame.copy())
    imwrite('/tmp/gray_frame_2.jpeg', gray_frame_2)

    # Threshold test
    # _, threshold_1 = cv2.threshold(gray_frame_1.copy(), 100.0, 255.0, cv2.THRESH_BINARY)
    # imwrite("/tmp/thresh_frame_1.jpeg", threshold_1)
    # threshold_2: ndarray = threshold(gray_frame_2.copy(), 100.0, 255.0)
    # imwrite("/tmp/thresh_frame_2.jpeg", threshold_2)

    # Canny test
    # canny_1 = cv2.Canny(gray_frame_2.copy(), 100.0, 255.0, 1, True)
    # imwrite("/tmp/canny_frame_1.jpeg", canny_1)
    canny_2 = canny(gray_frame_2.copy(), 100.0, 255.0, 3, True)
    imwrite('/tmp/canny_frame_2.jpeg', canny_2)


# Canny by sigma test
    canny_sigma_2 = canny_sigma(gray_frame_2.copy(), 3, 0.05, True)
    imwrite('/tmp/canny_sigma_frame_2.jpeg', canny_sigma_2)

    # Differences test
    res = str(RESOURCES_DIR_PATH.absolute())
    file_path_tmpl = f'{res}/test_frame_%s.jpeg'
    all_frames = [imread(file_path_tmpl % index) for index in range(1, 20)]
    all_frames = [grayscale(img) for img in all_frames if img is not None]
    all_frames = [canny_sigma(img, 3, 0.05, True) for img in all_frames]
    diff_frame_1 = difference(all_frames)
    imwrite('/tmp/differences_frame_1.jpeg', diff_frame_1)
    diff_frame_2 = difference_reduce(all_frames)
    imwrite('/tmp/differences_frame_2.jpeg', diff_frame_2)

    bounds = PyColorBounds(8, 9, 10, 11)
    vibro_frame = vibration(diff_frame_2, 8, 2, bounds)
    imwrite('/tmp/vibro_frame.jpeg', vibro_frame)

    median_1 = median(gray_frame_1)
    median_2 = median(gray_frame_2)

    print()


if __name__ == '__main__':
    main()

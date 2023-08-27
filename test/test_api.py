from os import walk
from pathlib import Path
from typing import List

from cv2 import imread
from numpy import ndarray
from pytest import raises

from pycvl import PyCvlHelper, PyColorBounds


RESOURCES_DIR_PATH = Path(__file__).parent / 'resources'


def test_grayscale():
    loaded_images = load_resource_frames()
    first_frame = loaded_images[0]
    pycvl = PyCvlHelper()
    gray_image = pycvl.grayscale(first_frame)
    assert gray_image.shape == (360, 640)


def test_threshold():
    loaded_images = load_resource_frames()
    first_frame = loaded_images[0]
    pycvl = PyCvlHelper()
    gray_image = pycvl.grayscale(first_frame)
    threshold_image = pycvl.threshold(gray_image, 100, 200)
    assert threshold_image.shape == (360, 640)


def test_threshold_throw():
    with raises(TypeError):
        loaded_images = load_resource_frames()
        first_frame = loaded_images[0]
        pycvl = PyCvlHelper()
        _ = pycvl.threshold(first_frame, 100, 200)


def test_canny():
    loaded_images = load_resource_frames()
    first_frame = loaded_images[0]
    pycvl = PyCvlHelper()
    gray_image = pycvl.grayscale(first_frame)
    canny_image = pycvl.canny(gray_image, 100, 200, 3, True)
    assert canny_image.shape == (360, 640)


def test_canny_sigma():
    loaded_images = load_resource_frames()
    first_frame = loaded_images[0]
    pycvl = PyCvlHelper()
    gray_image = pycvl.grayscale(first_frame)
    canny_image = pycvl.canny_sigma(gray_image, 3, 0.05, True)
    assert canny_image.shape == (360, 640)


def test_canny_throw():
    with raises(TypeError):
        loaded_images = load_resource_frames()
        first_frame = loaded_images[0]
        pycvl = PyCvlHelper()
        _ = pycvl.canny_sigma(first_frame, 3, 0.05, True)


def test_distribution():
    pass


def test_compute_median():
    loaded_images = load_resource_frames()
    first_frame = loaded_images[0]
    pycvl = PyCvlHelper()
    gray_image = pycvl.grayscale(first_frame)
    result = pycvl.median(gray_image)
    assert result == 194.86283854166666


def test_difference():
    loaded_images = load_resource_frames()
    pycvl = PyCvlHelper()
    gray_images = map(lambda x: pycvl.grayscale(x), loaded_images)
    canny_images = map(lambda x: pycvl.canny_sigma(x, 3, 0.5, True), gray_images)
    difference_image = pycvl.difference(list(canny_images))
    assert difference_image.shape == (360, 640)
    assert pycvl.median(difference_image) == 10.491080729166667


def test_difference_reduce():
    loaded_images = load_resource_frames()
    pycvl = PyCvlHelper()
    gray_images = map(lambda x: pycvl.grayscale(x), loaded_images)
    canny_images = map(lambda x: pycvl.canny_sigma(x, 3, 0.5, True), gray_images)
    difference_image = pycvl.difference_reduce(list(canny_images))
    assert difference_image.shape == (360, 640)
    assert pycvl.median(difference_image) == 17.324283854166666


def test_vibration():
    loaded_images = load_resource_frames()
    pycvl = PyCvlHelper()
    gray_images = map(lambda x: pycvl.grayscale(x), loaded_images)
    canny_images = map(lambda x: pycvl.canny_sigma(x, 3, 0.5, True), gray_images)
    difference_image = pycvl.difference(list(canny_images))

    color_bounds = PyColorBounds(8, 9, 10, 11)
    vibration_image = pycvl.vibration(difference_image, color_bounds, 8, 2)
    assert vibration_image.shape == (360, 640, 4)


def load_resource_frames() -> List[ndarray]:
    resource_path = RESOURCES_DIR_PATH.absolute()
    walk_iterator = walk(str(resource_path))
    return [
        imread(f'{dir_path}/{image_file_path}')
        for dir_path, _, file_names in walk_iterator
        for image_file_path in file_names
    ]

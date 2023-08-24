from dataclasses import dataclass
from typing import List

from numpy import ndarray


@dataclass
class PyColorBounds:
    ch1: int
    ch2: int
    ch3: int
    ch4: int


def median(frame: ndarray) -> float:
    """

    :param frame:
    :return:
    """


def grayscale(frame: ndarray) -> ndarray:
    """

    :param frame:
    :return:
    """


def threshold(frame: ndarray, thresh: float, maxval: float) -> ndarray:
    """

    :param frame:
    :param thresh:
    :param maxval:
    :return:
    """


def canny(frame: ndarray, low: float, high: float, size: float, is_l2: bool) -> ndarray:
    """

    :param frame:
    :param low:
    :param high:
    :param size:
    :param is_l2:
    :return:
    """


def canny_sigma(frame: ndarray, size: int, sigma: float, is_l2: bool) -> ndarray:
    """

    :param frame:
    :param size:
    :param sigma:
    :param is_l2:
    :return:
    """


def difference(frames: List[ndarray]) -> ndarray:
    """

    :param frames:
    :return:
    """


def difference_reduce(frames: List[ndarray]) -> ndarray:
    """

    :param frames:
    :return:
    """


def vibration(frame: ndarray, neighbours: int, window_size: int, color_bounds: PyColorBounds) -> ndarray:
    """

    :param frame:
    :param neighbours:
    :param window_size:
    :param color_bounds:
    :return:
    """

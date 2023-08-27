from dataclasses import dataclass
from typing import List

from numpy import ndarray


@dataclass
class PyColorBounds:
    """
    This data class useful to setting up channels bounds values
    which used for setting color on pixels while generating vibration image.
    """
    ch1: int
    ch2: int
    ch3: int
    ch4: int


class PyCvlHelper:

    def __init__(self):
        """
        There is main constructor for current class which provides ability to use CvlCore APIs.
        """

    def median(self, frame: ndarray) -> float:
        """
        This method returns computed median value of passed frame sub-matrix.
        The other words this algorithm iterates over each pixel of image (representation of
        threshold->Canny->diff-image chain transformation result) and calculate amount of nonzero
        pixels around current pixel. The computed value replaced instead pixel value.

        :param frame: (ndarray) a frame sub-matrix to get median.
        :return: (float) a median value.
        """


    def grayscale(self, frame: ndarray) -> ndarray:
        """
        This method transforms within RGB frame like adding/removing the alpha channel, reversing the
        channel order, conversion to/from 16-bit RGB color (R5:G6:B5 or R5:G5:B5), as well as
        conversion to/from grayscale.
        :param frame: (ndarray) an input stream frame to transform.
        :return: (ndarray) a grayscale frame.
        """


    def threshold(self, frame: ndarray, thresh: float, maxval: float) -> ndarray:
        """
        This method returns threshold image from passed bgr-image by passed black/white bounds
        values. The simplest thresholding methods replace each pixel in an image with a black
        pixel if the image intensity less than a fixed value called the threshold if the pixel
        intensity is greater than that threshold. This function is necessary for further image
        transformation to generate pixels vibration image.

        :param frame: (ndarray) a passed video stream frame to transform.
        :param thresh: (int) a black bound-value to swap pixel value to 1.
        :param maxval: (int) a white bound-value to swap pixel value to 0.
        :return: (ndarray) threshold frame.
        """


    def canny(self, frame: ndarray, low: float, high: float, size: int = 3, is_l2: bool = False) -> ndarray:
        """
        This method returns canny image from passed grayscale image by user parameters.
        The Canny edge detector is an edge detection operator that uses a multi-stage algorithm
        to detect a wide range of edges in images. It was developed by John F. Canny in 1986.
        Canny also produced a computational theory of edge detection explaining why the
        technique works.


        :param frame: (ndarray) a passed video stream frame to transform.
        :param low: (int) a first threshold for the hysteresis procedure.
        :param high: (int) a second threshold for the hysteresis procedure.
        :param size: (int) an aperture size of Sobel operator to generate Canny view.
        :param is_l2: (bool) a specifies the equation for finding gradient magnitude.
        :return: (ndarray) Canny frame.
        """


    def canny_sigma(self, frame: ndarray, size: int = 3, sigma: float = 0.05, is_l2: bool = False) -> ndarray:
        """
        This method returns canny image from passed grayscale image computing average value from sigma.
        The Canny edge detector is an edge detection operator that uses a multi-stage algorithm
        to detect a wide range of edges in images. It was developed by John F. Canny in 1986.
        Canny also produced a computational theory of edge detection explaining why the
        technique works.

        :param frame: (ndarray) a passed video stream frame to transform.
        :param size: (int) an aperture size of Sobel operator to generate Canny view.
        :param sigma: (float) a value to vary the percentage thresholds that are determined based on simple statistics.
        :param is_l2: (bool) a specifies the equation for finding gradient magnitude.
        :return: (ndarray) Canny frame.
        """


    def difference(self, frames: List[ndarray]) -> ndarray:
        """
        This recursive method returns result-image of absdiff() method by passed list of frames.
        This method used for getting only changed pixels from frames suite to further vibration
        image generating.

        For example, we have both matrix:

            0 1 0       0 1 0      0 0 0
            1 0 1  and  1 1 1  =>  0 1 0
            0 1 0       0 1 0      0 0 0

        :param frames: (List[ndarray]) a frames collection to get changed pixels.
        :return: (ndarray) a diff-image of suite frames.
        """


    def difference_reduce(self, frames: List[ndarray]) -> ndarray:
        """
        This method returns result-image by reducing resutls of absdiff() for aech pair frames into list.
        This method used for getting only changed pixels from frames suite to further vibration
        image generating.

        For example, we have both matrix:

            0 1 0       0 1 0      0 0 0
            1 0 1  and  1 1 1  =>  0 1 0
            0 1 0       0 1 0      0 0 0

        :param frames: (List[ndarray]) a frames collection to get changed pixels.
        :return: (ndarray) a diff-image of suite frames.
        """


    def vibration(self, frame: ndarray, color_bounds: PyColorBounds, neighbours: int = 8, window_size: int = 2) -> ndarray:
        """
        This method returns image with vibrating pixels (colored by bounds values) by passed image.
        This method uses calculate_vibrating_image method to compute new image presentation
        (see pydoc for calculate_vibrating_image) The vibration image is network procedure which used
        for anxiety triggering by dispersion of statistic values.

        :param frame: (ndarray) a passed diff-image (results of abs) to transform.
        :param color_bounds: (PyColorBounds) a bounds object to set up borders channels.
        :param neighbours: (int) a neighbours count value to filter noise of vibration.
        :param window_size: (int) an offset from central pixels to count neighbours non-null pixels.
        :return: (ndarray) a vibration frame.
        """

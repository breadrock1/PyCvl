from pathlib import Path
from typing import Union

from cv2 import VideoCapture
from cv2 import error, imshow, waitKey
from numpy import ndarray

from pycvl import PyColorBounds
from pycvl import (
    canny_sigma,
    difference_reduce,
    grayscale,
    vibration
)
from queue_list import QueueList


MAX_QUEUE_SIZE = 10
ROOT_DIR_PATH = Path(__file__).parent.parent


class VibrationFrameExample:
    """
    There is class with simple functionality to generate vibration image.
    """

    FrameType = Union[VideoCapture, ndarray]

    def __init__(self, video_file: Path or str):
        """
        The main class constructor.

        :param video_file: (Path or str) path to video file.
        """
        self.video_file_path = str(video_file)

        self.color_bounds = PyColorBounds(ch1=8, ch2=9, ch3=10, ch4=11)
        self.collected_frames = QueueList(max_size=MAX_QUEUE_SIZE)

    def launch_stream(self):
        """
        There is main function to launch and release video stream.
        """
        frame_iteration_enum = 0
        video_capture = VideoCapture(self.video_file_path)

        while video_capture.isOpened():
            try:
                _flag, _frame = video_capture.read()
                # This code block waiting new frame if it hasn't been loaded from stream yet.
                if not _flag:
                    break

                # Need waiting even frame count to exclude noises. By this way we optimized
                # processing of target image, cuz analyzing each frame does not unnecessary.
                if frame_iteration_enum % 2 != 0:
                    continue

                self._processing_frame(_frame)

            except error as err:
                print(f'Caught error from OpenCV core: {err}')

            except Exception as err:
                print(f'Caught runtime  exception: {err}')
                print('Closing current stream...')
                break

            finally:
                frame_iteration_enum += 1
                waitKey(10)

        video_capture.release()

    def _processing_frame(self, frame: FrameType):
        """
        There is method to produce of frame from stream.
        :param frame: (FrameType) video stream frame.
        """
        try:
            grayscale_image = grayscale(frame=frame)
            canny_image = canny_sigma(frame=grayscale_image, size=3, sigma=0.05, is_l2=True)
            self.collected_frames.append(canny_image)
            if len(self.collected_frames) < MAX_QUEUE_SIZE:
                return

            difference_image = difference_reduce(self.collected_frames.to_list)
            vibration_image = vibration(frame=difference_image,
                                        color_bounds=self.color_bounds,
                                        neighbours=8,
                                        window_size=2)

            imshow("Vibration Example", vibration_image)
        except Exception as err:
            print(f'Unknown runtime error: {err}')


if __name__ == '__main__':
    video_file_path = ROOT_DIR_PATH / 'resources' / 'video_1.mp4'
    example = VibrationFrameExample(video_file=video_file_path)
    example.launch_stream()

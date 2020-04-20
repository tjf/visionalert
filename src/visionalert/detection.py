import collections
from dataclasses import dataclass
import logging
import threading

import cv2

logger = logging.getLogger(__name__)

DetectionResult = collections.namedtuple(
    "DetectionResult", ["name", "confidence", "coordinates"]
)
Interest = collections.namedtuple("Interest", ["name", "confidence"])


@dataclass
class Rectangle:
    start_x: int
    start_y: int
    end_x: int
    end_y: int

    @property
    def area(self):
        return (self.end_x - self.start_x) * (self.end_y - self.start_y)


def is_masked(mask, rectangle):
    """
    Evaluates rectangle against mask returning true if the rectangle is entirely
    inside the masked area containing zeros

    :param mask: Numpy 2-dimensional nd_array where the value 255 is considered unmasked.
    :param rectangle: Rectangle to compare against the mask
    """
    if mask is not None:
        return (
            255
            not in mask[
                rectangle.start_y : rectangle.end_y, rectangle.start_x : rectangle.end_x
            ]
        )
    else:
        return False


def matches_interest(interests, detected_object):
    """
    Evaluates whether the detected_object matches the criteria defined by the dict
    of interests from a camera
    :param interests: dict containing an object name to Interest map
    :param detected_object: DetectionResult for evaluation
    :return:
    """
    if interests is None or detected_object.name not in interests:
        return False
    elif detected_object.confidence < interests[detected_object.name].confidence:
        return False
    else:
        return True


# TODO refactor this to get the magic numbers out of it and add some tests.
def annotate_frame(frame, detected_object, color=(0, 255, 0), line_weight=2):
    coords = detected_object.coordinates

    # Draw bounding box
    cv2.rectangle(
        frame,
        (coords.start_x, coords.start_y),
        (coords.end_x, coords.end_y),
        color,
        line_weight,
    )

    # Draw label background and text
    label = (
        f"{detected_object.name.capitalize()}: {int(detected_object.confidence * 100)}% ({coords.area})"
    )
    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    label_ymin = max(coords.start_y, label_size[1] + 10)

    cv2.rectangle(
        frame,
        (coords.start_x, label_ymin - label_size[1] - 10),
        (coords.start_x + label_size[0], label_ymin + base_line - 10),
        (255, 255, 255),
        cv2.FILLED,
    )

    cv2.putText(
        frame,
        label,
        (coords.start_x, label_ymin - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        2,
    )


class Dispatcher:
    """
    Retrieves frames from get_frame_function, checks them for objects via the
    detection_function finally passing them and any valid detections to alert_function.

    :param get_frame_function: Takes zero arguments and returns a Frame
    :param detection_function: Takes a single nd_array parameter and returns a list of DetectionResults
    :param alert_function: Takes a tuple containing a list of verified DetectionResults and the annotated
    Frame that was analyzed.
    :param cameras: dict mapping camera names to a Camera object
    """

    def __init__(self, get_frame_function, detection_function, alert_function, cameras):
        self._get_frame_function = get_frame_function
        self._detection_function = detection_function
        self._alert_function = alert_function
        self._cameras = cameras or {}

        self._thread = threading.Thread(
            name=self.__class__.__name__, daemon=True, target=self._dispatch_loop
        )

    def start(self):
        self._thread.start()

    def join(self):
        self._thread.join()

    def _dispatch_loop(self):
        while True:
            self._process_frame(self._get_frame_function())

    def _process_frame(self, frame):
        try:
            camera = self._cameras[frame.camera_name]
        except KeyError:
            logger.warning(
                f"Camera {frame.camera_name} not registered with dispatcher!"
            )
            return

        valid_detections = [
            detection
            for detection in self._detection_function(frame.data)
            if matches_interest(camera.interests, detection)
            and not is_masked(camera.mask, detection.coordinates)
        ]

        for detected_object in valid_detections:
            annotate_frame(frame.data, detected_object)

        if valid_detections:
            self._alert_function((valid_detections, frame))

import collections
import logging
from threading import BoundedSemaphore, Thread
import time

import cv2
import numpy
from PIL import Image

import visionalert.alert as alert

logger = logging.getLogger(__name__)

DetectionResult = collections.namedtuple(
    "DetectionResult", ["name", "confidence", "coordinates"]
)
Rectangle = collections.namedtuple(
    "Rectangle", ["start_x", "start_y", "end_x", "end_y"]
)


def create_empty_boundedsemaphore(size):
    s = BoundedSemaphore(size)
    for _ in range(size):
        s.acquire()
    return s


def load_mask(filename):
    image = Image.open(filename)
    return Mask(numpy.asarray(image))


# TODO refactor this to get the magic numbers out of it and add some tests.
def annotate_frame(frame, detected_object, color=(0, 255, 0), line_weight=2):
    coords = detected_object.coordinates

    # Annotate bounding box
    cv2.rectangle(
        frame,
        (coords.start_x, coords.start_y),
        (coords.end_x, coords.end_y),
        color,
        line_weight,
    )

    # Annotate label background and text
    label = (
        f"{detected_object.name.capitalize()}: {int(detected_object.confidence * 100)}%"
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


# Please forgive me, I've been writing Java for the last 10 years. :-(


class Mask:
    """
    Masks off a partial region to avoid triggering alarms if an object is detected
    in that area.  Trigger depth indicates how many rows on the bottom of the bounding
    box of an object may pass into a unmasked area before triggering a violation.  This
    allows us to alert when a person steps from the sidewalk onto the driveway, for
    example, before the entire bounding box is inside the mask.

    """

    def __init__(self, mask_ndarray, trigger_depth=5):
        if not isinstance(mask_ndarray, numpy.ndarray) or mask_ndarray.ndim != 2:
            raise ValueError("Incorrect mask format.  Are you using a grayscale image?")
        self._mask = mask_ndarray
        self.trigger_depth = trigger_depth

    def hides(self, rectangle):
        row = max(rectangle.end_y - self.trigger_depth, 0)
        return 255 not in self._mask[row][rectangle.start_x : rectangle.end_x]


class Dispatcher:
    """
    Receives frames from the configured cameras and submits them to the list
    of sensors configured for a given camera.
    """

    def __init__(self, detection_function, max_queue_size=20):
        self._detection_function = detection_function
        self._deque = collections.deque(maxlen=max_queue_size)
        self._semaphore = create_empty_boundedsemaphore(max_queue_size)
        self._sensors = collections.defaultdict(list)
        self._masks = {}
        self._thread = Thread(
            name=self.__class__.__name__, daemon=True, target=self._run
        )
        self._thread.start()

    def submit_frame(self, stream_name, frame):
        try:
            # We use a deque here instead of a queue so we can put bounds on
            # the size of waiting frames and drop older ones if we start to
            # overflow.  Unfortunately there is no blocking 'get' call for
            # deques so we use a semaphore in lieu of busy polling in the
            # detection thread.
            self._deque.appendleft((stream_name, frame))
            self._semaphore.release()  # Let other thread know something is waiting
        except ValueError:
            logger.warning(
                "Object detection input queue overflow detected, discarding oldest frame."
            )

    def _run(self):
        while True:
            self._semaphore.acquire()
            stream_name, frame = self._deque.pop()

            logger.debug(f"Submitting frame for object detection from {stream_name}")
            detections = self._detection_function(frame)
            # detections = [
            #     detection
            #     for detection in self._detection_function(frame)
            #     if not self._masks[stream_name].hides(detection.coordinates)
            # ]

            for each in detections:
                if stream_name in self._masks and self._masks[stream_name].hides(
                    each.coordinates
                ):
                    continue
                for sensor in self._sensors[stream_name]:
                    sensor.submit(frame, each)

    def add_sensor(self, sensor):
        self._sensors[sensor.stream_name].append(sensor)
        logger.debug(f"Adding sensor {sensor} for stream {sensor.stream_name}")

    def add_mask(self, stream_name, mask):
        self._masks[stream_name] = mask


class Sensor:
    """
    Creates detection events when a configured object is detected in frame.  Events
    are defined as continuous detection of an object and conclude when an object is
    no longer detected for 30 seconds.
    """

    def __init__(
        self, stream_name, object_type, minimum_confidence=0.5, alerter=alert.Alerter,
    ) -> None:
        """
        Instances of this are registered with the DetectionDispatcher instance.

        :param stream_name: Name of stream this sensor is for
        :param object_type: Type of object from the label map corresponding to the
        model in use by TensorFlow
        :param minimum_confidence: The minimum confidence score required to trigger
        an alert.
        :param alerter:
        """

        self.stream_name = stream_name
        self.object_type = object_type
        self.minimum_confidence = minimum_confidence
        self._alerter = alerter

        self.seconds_without_detection = 30  # TODO move this

        self.event = alert.Event(self.stream_name)

    def submit(self, frame, item):
        if item.name != self.object_type or item.confidence < self.minimum_confidence:
            return

        if (
            time.time() - self.event.last_event_frame_time
            > self.seconds_without_detection
        ):
            logger.info(f"New detection event triggered on {self.stream_name}!")
            self.event = alert.Event(self.stream_name)
            self._alerter.enqueue_alert(self.event)

        logger.info(
            f"{item.name.capitalize()} detected on stream {self.stream_name} with confidence "
            f"score {item.confidence} at {item.coordinates}"
        )

        self.event.last_event_frame_time = time.time()
        if item.confidence > self.event.confidence:
            annotate_frame(frame, item)
            self.event.update(item.confidence, frame)

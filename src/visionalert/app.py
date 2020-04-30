import argparse
import collections
import logging
import sys
import threading

import numpy
from PIL import Image

from visionalert import tensorflow
from visionalert import load_config, config
from visionalert.alert import Notifier
from visionalert.detection import Dispatcher, Interest
from visionalert.video import Camera

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze and detect objects in your video feeds"
    )
    parser.add_argument(
        "-c", default="config.yml", dest="config", help="location of configuration file"
    )
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    return parser.parse_args()


def init_logging(debug):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s",
    )

    if not debug:  # Hide nasty stack traces unless we're in debug mode
        sys.tracebacklimit = 0


def init_camera(config, frame_action):
    return Camera(
        config["name"],
        config["url"],
        frame_action,
        fps=config.get("fps"),
        mask=init_mask(config["mask"]) if "mask" in config else None,
        interests={
            name: Interest(
                name,
                interest["confidence"],
                interest.get("minimum_area", 0),
                interest.get("maximum_area", sys.maxsize),
            )
            for name, interest in config["interests"].items()
        },
    )


def init_mask(filename):
    image = Image.open(filename)
    return numpy.asarray(image)


def run():
    args = parse_args()
    load_config(args.config)
    init_logging(args.debug)

    input_queue = DiscardingQueue(
        config["input_queue_maximum_frames"],
        overflow_action=lambda: logger.warning(
            "Object detection input queue overflow detected, discarding oldest frame!"
        ),
    )

    cameras = {
        params["name"]: init_camera(params, input_queue.put)
        for params in config["cameras"]
    }

    detector = tensorflow.create_detector(
        config["tensorflow_model_file"], config["tensorflow_label_map"]
    )

    dispatcher = Dispatcher(
        input_queue.get, detector, Notifier().submit_detections, cameras
    )
    dispatcher.start()

    for camera in cameras.values():
        camera.connection_timeout = config["connection_timeout_seconds"]
        camera.read_timeout = config["read_timeout_seconds"]
        camera.start()

    dispatcher.join()


class DiscardingQueue:
    """A bounded queue that discards the oldest items when it overflows"""

    def __init__(self, max_size, overflow_action=None) -> None:
        self._deque = collections.deque(maxlen=max_size)
        self._semaphore = threading.BoundedSemaphore(max_size)
        self._overflow_action = overflow_action
        for _ in range(max_size):
            self._semaphore.acquire()

    def put(self, item):
        try:
            self._deque.appendleft(item)
            self._semaphore.release()
        except ValueError:
            if self._overflow_action:
                self._overflow_action()

    def get(self):
        self._semaphore.acquire()
        return self._deque.pop()


if __name__ == "__main__":
    run()

import argparse
import collections
import logging
import sys
import threading

import numpy
from PIL import Image

from visionalert import tensorflow
from visionalert.alert import Notifier
from visionalert.config import load_config, Config
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
        fps=config["fps"] if "fps" in config else None,
        mask=init_mask(config["mask"]) if "mask" in config else None,
        interests=init_interests(config["interests"]),
    )


def init_mask(filename):
    image = Image.open(filename)
    return numpy.asarray(image)


def init_interests(config):
    return {
        name: Interest(name=name, confidence=v["confidence"])
        for name, v in config.items()
    }


def run():
    args = parse_args()
    load_config(args.config)
    init_logging(args.debug)

    input_queue = DiscardingQueue(
        Config["input_queue_maximum_frames"],
        overflow_action=lambda: logger.warning(
            "Object detection input queue overflow detected, discarding oldest frame!"
        ),
    )

    cameras = {
        params["name"]: init_camera(params, input_queue.put)
        for params in Config["cameras"]
    }

    detector = tensorflow.create_detector(
        Config["tensorflow_model_file"], Config["tensorflow_label_map"]
    )

    dispatcher = Dispatcher(
        input_queue.get, detector, Notifier().submit_detections, cameras
    )
    dispatcher.start()

    for camera in cameras.values():
        camera.connection_timeout = Config["connection_timeout_seconds"]
        camera.read_timeout = Config["read_timeout_seconds"]
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

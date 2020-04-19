import argparse
import logging
import sys
import threading

from visionalert import tensorflow
from visionalert.config import load_config, Config
from visionalert.detection import Dispatcher, Sensor, load_mask
from visionalert.video import Camera

logger = logging.getLogger(__name__)


def convert_frame_to_rgb(func):
    def _wrapper_to_rgb(name, frame):
        func(name, frame.to_ndarray(format="rgb24"))

    return _wrapper_to_rgb


def init_camera(params, dispatcher):
    cam = Camera(
        params["name"],
        params["url"],
        convert_frame_to_rgb(dispatcher.submit_frame),
        fps=params["fps"],
    )

    if "mask" in params:
        logger.info(f"Loading mask {params['mask']} for {cam.name}")
        dispatcher.add_mask(cam.name, load_mask(params["mask"]))

    for sensor_type, sensor_conf in params["interests"].items():
        dispatcher.add_sensor(
            Sensor(cam.name, sensor_type, sensor_conf["confidence"])
        )

    return cam


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze and detect objects in your video feeds"
    )
    parser.add_argument(
        "-c", default="config.yml", dest="config", help="location of configuration file"
    )
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    return parser.parse_args()


def run():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s",
    )

    if not args.debug:  # Hide nasty stack traces unless we're in debug mode
        sys.tracebacklimit = 0

    load_config(args.config)

    detector = tensorflow.create_detector(
        Config["tensorflow_model_file"], Config["tensorflow_label_map"]
    )
    dispatcher = Dispatcher(detector)
    cameras = [init_camera(params, dispatcher) for params in Config["cameras"]]
    for c in cameras:
        c.start()

    threading.Event().wait()
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

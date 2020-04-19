from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import logging
import threading
import time

import boto3
from PIL import Image
import requests

from visionalert.config import Config

logger = logging.getLogger(__name__)


def s3_upload(byte_object, mime_type, s3_filename):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=Config["aws_access_key"],
        aws_secret_access_key=Config["aws_secret_key"],
        endpoint_url=Config["aws_s3_url"],
    )

    s3.upload_fileobj(
        byte_object,
        Config["aws_image_bucket"],
        s3_filename,
        ExtraArgs={"ContentType": mime_type},
    )


def frame_to_jpeg(frame):
    image_bytes = BytesIO()
    image = Image.fromarray(frame)
    image.thumbnail((1024, 1024), Image.ANTIALIAS)
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)  # Rewind the pointer
    return image_bytes


def send_push_notification(title, image_url):
    gotify_key = {"X-Gotify-Key": Config["gotify_key"]}
    gotify_req = {
        "extras": {"client::display": {"contentType": "text/markdown"}},
        "message": f"[![Image]({image_url})]({image_url})",
        "priority": 5,
        "title": f"{title}",
    }
    requests.post(
        f"{Config['gotify_url']}/message", json=gotify_req, headers=gotify_key
    )


class Notifier:
    """
    Responsible for sending alert when an event is received.
    """

    executor = ThreadPoolExecutor(thread_name_prefix="Notifier")
    _event_scoreboard = {}

    @classmethod
    def _get_last_object_event(cls, camera_name, object_name):
        if camera_name not in cls._event_scoreboard:
            cls._event_scoreboard[camera_name] = {}

        if object_name not in cls._event_scoreboard[camera_name]:
            cls._event_scoreboard[camera_name][object_name] = None

        return cls._event_scoreboard[camera_name][object_name]

    @classmethod
    def _set_last_object_event(cls, event):
        cls._event_scoreboard[event.camera_name][event.object_name] = event

    @classmethod
    def submit_detections(cls, data):
        detections, frame = data

        for each in detections:
            event = cls._get_last_object_event(frame.camera_name, each.name)

            if not event or time.time() - event.last_frame_time > 30:
                logger.info(
                    f"New detection event for {each.name} triggered on {frame.camera_name} "
                    f"with confidence {each.confidence * 100:.2f}%"
                )
                new_event = Event(frame.camera_name, each.name)
                new_event.update(each.confidence, frame.data)
                cls._set_last_object_event(new_event)
                cls._enqueue_alert(new_event)

            else:
                logger.info(
                    f"{each.name.capitalize()} still detected on camera {frame.camera_name} "
                    f"with confidence {each.confidence * 100:.2f}% at {each.coordinates}"
                )
                event.update(each.confidence, frame.data)

    @classmethod
    def _enqueue_alert(cls, event):
        cls.executor.submit(cls._send_alert, event)

    @classmethod
    def _send_alert(cls, event):
        try:
            # Wait a few seconds to give the event a chance to potentially
            # get a higher scoring frame.
            time.sleep(5)  # TODO make this configurable

            s3_filename = f"{int(time.time())}.jpg"
            s3_upload(frame_to_jpeg(event.frame), "image/jpg", s3_filename)

            send_push_notification(
                f"{event.camera_name} Motion Detected",
                f"{Config['aws_image_base_url']}/{s3_filename}",
            )

            logger.info(
                f"Sending alert for {event.object_name} on camera {event.camera_name} "
                f"with confidence {event.confidence * 100:.2f}%"
            )

        except Exception as e:
            logger.warning(e)


class Event:
    """
    Updateable container representing a notification Event.  Allows the representative
    frame to be updated if one with a higher confidence is available.
    """

    def __init__(self, camera_name, object_name) -> None:
        self.camera_name = camera_name
        self.object_name = object_name
        self._mutex = threading.RLock()  # Just being cautious here
        self._last_frame_time = 0.0
        self._confidence = 0.0
        self._frame = None

    def update(self, confidence, frame):
        with self._mutex:
            self._last_frame_time = time.time()
            if confidence > self._confidence:
                self._frame = frame
                self._confidence = confidence

    @property
    def confidence(self):
        with self._mutex:
            return self._confidence

    @property
    def frame(self):
        with self._mutex:
            return self._frame

    @property
    def last_frame_time(self):
        with self._mutex:
            return self._last_frame_time

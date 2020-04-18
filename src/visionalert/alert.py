from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import logging
import threading
import time

import boto3
from PIL import Image
import requests

from visionalert.config import Config


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


class Alerter:
    """
    Responsible for sending alert when an event is received.
    """
    executor = ThreadPoolExecutor(thread_name_prefix="Alerter")

    @classmethod
    def enqueue_alert(cls, event):
        cls.executor.submit(cls.execute, event)

    @classmethod
    def execute(cls, event):
        try:
            # Wait a few seconds to give the event a chance to potentially
            # get a higher scoring frame.
            time.sleep(5)  # TODO make this configurable

            s3_filename = f"{int(time.time())}.jpg"
            s3_upload(frame_to_jpeg(event.frame), "image/jpg", s3_filename)

            send_push_notification(
                f"{event.stream_name} Motion Detected",
                f"{Config['aws_image_base_url']}/{s3_filename}",
            )

            logging.info(
                f"Sending alert on stream {event.stream_name} with confidence {event.confidence}"
            )

        except Exception as e:
            logging.warning(e)


class Event:
    """
    Container that is populated by a sensor when an event happens.  As the event
    is being triggered, instances of this are continually updated to capture
    the frame with the highest confidence score until a configured amount of time
    passes without any detections, thus ending the event.
    """
    def __init__(self, stream_name) -> None:
        self.stream_name = stream_name
        self.last_event_frame_time = 0.0

        self._mutex = threading.RLock()  # Just being cautious here
        self._confidence = 0.0
        self._frame = None

    def update(self, confidence, frame):
        with self._mutex:
            self._confidence = confidence
            self._frame = frame

    @property
    def confidence(self):
        with self._mutex:
            return self._confidence

    @property
    def frame(self):
        with self._mutex:
            return self._frame

import time

import numpy
import pytest

from visionalert.config import Config
import visionalert.alert as alert
import visionalert.video as video
import visionalert.detection as detection


@pytest.fixture
def detections():
    frame = video.Frame("camera", numpy.zeros((200, 200, 3), dtype=numpy.uint8))
    detected_objects = [
        detection.DetectionResult("person", 0.8, detection.Rectangle(0, 0, 20, 20)),
        detection.DetectionResult("person", 0.9, detection.Rectangle(0, 0, 20, 20)),
        detection.DetectionResult("person", 0.3, detection.Rectangle(0, 0, 20, 20)),
    ]
    return detected_objects, frame


@pytest.fixture
def enqueuer_args(monkeypatch):
    args = []
    monkeypatch.setattr(
        "visionalert.alert.Notifier._enqueue_alert", lambda *x: args.append(x)
    )
    return args


@pytest.fixture
def patched_send_alert(monkeypatch):
    mock_args = []

    capture = lambda *x: mock_args.append(x)
    monkeypatch.setattr(alert, "s3_upload", capture)
    monkeypatch.setattr(alert, "send_push_notification", capture)
    monkeypatch.setattr(time, "time", lambda: 31337)
    Config._config_dict["aws_image_base_url"] = "http://foo.com"

    return mock_args


@pytest.fixture
def alert_event(detections):
    return alert.Event(detections[1], detections[0][0])


def test_submit_detections_should_enqueue_single_event(enqueuer_args, detections):
    notifier = alert.Notifier()
    notifier.submit_detections(detections)
    notifier.submit_detections(detections)
    assert len(enqueuer_args) == 1


def test_submit_detections_should_enqueue_new_event_after_timeout(
    enqueuer_args, detections
):
    notifier = alert.Notifier(detection_timeout=1)
    notifier.submit_detections(detections)
    time.sleep(1.5)
    notifier.submit_detections(detections)

    assert len(enqueuer_args) == 2
    assert enqueuer_args[0][1] is not enqueuer_args[1][1]  # Should be diff events


def test_enqueued_alert_event_should_have_highest_confidence(enqueuer_args, detections):
    notifier = alert.Notifier()
    notifier.submit_detections(detections)
    assert enqueuer_args[0][1].confidence == 0.9


def test_send_alert_upload_correct_content_type(patched_send_alert, alert_event):
    notifier = alert.Notifier(send_delay=0)
    notifier._send_alert(alert_event)
    assert patched_send_alert[0][1] == "image/jpg"


def test_send_alert_upload_filename_is_time(patched_send_alert, alert_event):
    notifier = alert.Notifier(send_delay=0)
    notifier._send_alert(alert_event)
    assert patched_send_alert[0][2] == "31337.jpg"


def test_send_alert_notification_image_url(patched_send_alert, alert_event):
    notifier = alert.Notifier(send_delay=0)
    notifier._send_alert(alert_event)
    assert patched_send_alert[1][1] == "http://foo.com/31337.jpg"

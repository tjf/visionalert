import numpy as np
import pytest

import visionalert.video as video
import visionalert.detection as detection
from visionalert.detection import Rectangle, DetectionResult, Interest, Dispatcher

SMALL_RECTANGLE = Rectangle(5, 5, 30, 30)  # Area 625
MEDIUM_RECTANGLE = Rectangle(5, 5, 100, 100)  # Area 9025
LARGE_RECTANGLE = Rectangle(5, 5, 300, 300)  # Area 87025


@pytest.mark.parametrize(
    "rectangle, expected",
    [(Rectangle(4, 1, 9, 9), False), (Rectangle(4, 1, 9, 5), True)],
)
def test_is_masked(rectangle, expected):
    mask = np.zeros((10, 10))
    mask[5:, :] = 255  # unmask the bottom half
    assert detection.is_masked(mask, rectangle) == expected


def test_is_masked_when_mask_is_none():
    assert detection.is_masked(None, Rectangle(0, 0, 100, 100)) is False


def test_rectangle_area():
    assert Rectangle(0, 0, 10, 10).area == 100


@pytest.mark.parametrize(
    "detection_result, expected",
    [
        (DetectionResult("person", 0.1, MEDIUM_RECTANGLE), False),
        (DetectionResult("person", 0.8, MEDIUM_RECTANGLE), True),
        (DetectionResult("car", 0.8, SMALL_RECTANGLE), False),
        (DetectionResult("car", 0.8, MEDIUM_RECTANGLE), True),
        (DetectionResult("car", 0.8, LARGE_RECTANGLE), False),
        (DetectionResult("truck", 0.8, MEDIUM_RECTANGLE), False),
    ],
)
def test_matches_interest(detection_result, expected):
    interests = {
        "person": Interest("person", 0.6, minimum_area=0, maximum_area=1000000),
        "car": Interest("car", 0.5, minimum_area=5000, maximum_area=10000),
    }
    assert detection.matches_interest(interests, detection_result) == expected


def test_dispatcher_process_frame(mocker, mock_camera, mock_detection_results):
    alert_function = mocker.Mock()

    detection_function = mocker.Mock()
    detection_function.return_value = mock_detection_results

    mocker.patch("visionalert.detection.annotate_frame")

    frame = video.Frame(mock_camera.name, np.zeros((200, 200, 3)))
    camera_dict = {mock_camera.name: mock_camera}

    d = Dispatcher(None, detection_function, alert_function, camera_dict)
    d._process_frame(frame)

    detection_function.assert_called_once_with(frame.data)
    alert_function.assert_called_once_with(([mock_detection_results[0]], frame))
    detection.annotate_frame.assert_called_once_with(
        frame.data, mock_detection_results[0]
    )


@pytest.fixture()
def mock_camera():
    return video.Camera(
        "test_cam", None, None, interests={"person": Interest("person", 0.6, 0, 10000)}
    )


@pytest.fixture()
def mock_detection_results():
    return [
        DetectionResult("person", 0.8, Rectangle(10, 10, 50, 50)),
        DetectionResult("person", 0.2, Rectangle(10, 10, 50, 50)),
        DetectionResult("cat", 0.8, Rectangle(20, 20, 50, 50)),
    ]

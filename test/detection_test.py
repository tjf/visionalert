import numpy as np
import pytest

import visionalert.video as video
import visionalert.detection as detection
from visionalert.detection import Rectangle, DetectionResult, Interest, Dispatcher

TEST_RECTANGLE = Rectangle(5, 5, 30, 30)


@pytest.mark.parametrize(
    "rectangle, expected",
    [(Rectangle(4, 1, 9, 9), False), (Rectangle(4, 1, 9, 5), True)],
)
def test_is_masked(rectangle, expected):
    mask = np.zeros((10, 10))
    mask[5:, :] = 255  # unmask the bottom half
    result = detection.is_masked(mask, rectangle)
    assert result == expected


def test_is_masked_when_mask_is_none():
    result = detection.is_masked(None, Rectangle(0, 0, 100, 100))
    assert result == False


@pytest.mark.parametrize(
    "detection_result, expected",
    [
        (DetectionResult("person", 0.1, TEST_RECTANGLE), False),
        (DetectionResult("person", 0.6, TEST_RECTANGLE), True),
        (DetectionResult("person", 0.8, TEST_RECTANGLE), True),
        (DetectionResult("cat", 0.8, TEST_RECTANGLE), False),
        (DetectionResult("car", 0.8, TEST_RECTANGLE), True),
    ],
)
def test_matches_interest(detection_result, expected):
    interests = {
        "person": Interest("person", 0.6),
        "car": Interest("car", 0.5),
    }
    result = detection.matches_interest(interests, detection_result)
    assert result == expected


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
        "test_cam", None, None, interests={"person": Interest("person", 0.6)}
    )


@pytest.fixture()
def mock_detection_results():
    return [
        DetectionResult("person", 0.8, Rectangle(10, 10, 50, 50)),
        DetectionResult("person", 0.2, Rectangle(10, 10, 50, 50)),
        DetectionResult("cat", 0.8, Rectangle(20, 20, 50, 50)),
    ]

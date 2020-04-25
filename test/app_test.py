import sys

import numpy
import pytest

import visionalert.app as app
from visionalert.detection import Interest


@pytest.fixture
def camera_dict():
    return {
        "name": "Test Camera",
        "url": "test.com",
        "fps": 5,
        "mask": "mask.jpg",
        "interests": {
            "person": {"confidence": 0.6, "minimum_area": 10000, "maximum_area": 20000}
        },
    }


def test_init_camera_with_all_values(camera_dict, monkeypatch):
    mask = numpy.zeros((200, 200))

    def init_mask(file):
        if file != "mask.jpg":
            raise ValueError
        return mask

    monkeypatch.setattr(app, "init_mask", value=init_mask)

    frame_action = lambda: None
    camera = app.init_camera(camera_dict, frame_action)

    assert camera.name == "Test Camera"
    assert camera.url == "test.com"
    assert camera.fps == 5
    assert camera.mask is mask
    assert camera.interests["person"] == Interest("person", 0.6, 10000, 20000)
    assert camera._frame_action is frame_action


def test_init_camera_with_missing_optional_values(camera_dict):
    del camera_dict["fps"]
    del camera_dict["mask"]
    camera_dict["interests"] = {"person": {"confidence": 0.6}}

    camera = app.init_camera(camera_dict, None)

    assert camera.fps is None
    assert camera.mask is None
    assert camera.interests["person"] == Interest("person", 0.6, 0, sys.maxsize)

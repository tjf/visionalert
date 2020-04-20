from fractions import Fraction

import av
import pytest

import visionalert.video as video


@pytest.fixture()
def mock_av_open(mocker):
    mocker.patch("visionalert.video.av.open")


def test_get_frames_should_pass_open_parameters(mock_av_open):
    _ = [_ for _ in video.get_frames("foo", connection_timeout=1.0, read_timeout=2.0)]
    av.open.assert_called_once_with(
        "foo", options={"rtsp_flags": "prefer_tcp"}, timeout=(1.0, 2.0)
    )


def test_get_frames_should_set_thread_type(mock_av_open):
    _ = [_ for _ in video.get_frames("")]
    assert (
        av.open.return_value.streams.video.__getitem__.return_value.thread_type
        == "AUTO"
    ), "Thread type not set"


def test_get_frames_should_close_container(mock_av_open):
    _ = [_ for _ in video.get_frames("")]
    av.open.return_value.close.assert_called_once()


@pytest.mark.parametrize("fps, expected_count", [(None, 50), (10, 50), (2, 10), (5, 25)])
def test_get_frames_should_decode_mp4_at_requested_fps(fps, expected_count):
    count = 0
    frame = None
    for frame in video.get_frames("fixtures/sample.mp4", fps=fps):
        count += 1

    assert count == expected_count
    assert frame.width == 640
    assert frame.height == 480

from fractions import Fraction

import av
import pytest

import visionalert.video as video
from test.helpers import ReturnThenRun

STREAM_NAME = "Stream Name"


@pytest.fixture()
def mock_av_open(mocker):
    mocker.patch("visionalert.video.av.open")


@pytest.fixture()
def mock_get_10_frames(mocker):
    """
    Mock get_frames to return 10 frames and optionally run
    a function after (useful for stopping threads)
    """

    get_frames_mock = ReturnThenRun(range(10))  # Simulate 10 frames
    mocker.patch("visionalert.video.get_frames")
    video.get_frames.side_effect = get_frames_mock
    return get_frames_mock


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


@pytest.mark.timeout(5)
def test_streamgrabber_should_call_callback_and_retry(mocker, mock_get_10_frames):
    mock_callback = mocker.Mock()

    s = video.StreamGrabber(STREAM_NAME, "", mock_callback)

    mock_get_10_frames.n_times = 2  # Twice as we're testing the retry
    mock_get_10_frames.then_run_function = (
        s.stop
    )  # Stop the thread the second time around

    s.start()
    s.join()

    calls = []
    for _ in range(2):  # 2 times around, 10 frames each
        [calls.append(mocker.call(STREAM_NAME, x)) for x in range(0, 10)]
    mock_callback.assert_has_calls(calls)


@pytest.mark.timeout(5)
def test_streamgrabber_should_call_callback_and_exit(mocker, mock_get_10_frames):
    mock_callback = mocker.Mock()

    s = video.StreamGrabber(STREAM_NAME, "", mock_callback)
    s.retry = False  # So we exit...

    s.start()
    s.join()

    calls = [mocker.call(STREAM_NAME, x) for x in range(0, 10)]
    mock_callback.assert_has_calls(calls)


@pytest.mark.parametrize("fps, expected_count", [(None, 50), (10, 50), (2, 10), (5, 25)])
def test_get_frames_should_decode_mp4_at_requested_fps(fps, expected_count):
    count = 0
    frame = None
    for frame in video.get_frames("fixtures/sample.mp4", requested_fps=fps):
        count += 1

    assert count == expected_count
    assert frame.width == 640
    assert frame.height == 480

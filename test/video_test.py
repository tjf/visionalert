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
    Mock get_frames to return 10 frames at 10 fps and optionally run
    a function after (useful for stopping threads)
    """
    framerate = Fraction(10)

    return_value = [(framerate, x) for x in range(10)]
    get_frames_mock = ReturnThenRun(return_value)  # Simulate 10 frames
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
def test_streamgrabber_should_retry_on_end(mocker, mock_get_10_frames):
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
def test_streamgrabber_should_exit_after_stream(mocker, mock_get_10_frames):
    mock_callback = mocker.Mock()

    s = video.StreamGrabber(STREAM_NAME, "", mock_callback)
    s.retry = False  # So we exit...

    s.start()
    s.join()

    calls = [mocker.call(STREAM_NAME, x) for x in range(0, 10)]
    mock_callback.assert_has_calls(calls)


@pytest.mark.parametrize(
    "fps, frame_index, framerate, expected",
    [
        (5, 0, Fraction(10), False),
        (5, 1, Fraction(10), True),
        (3, 3, Fraction(10), True),
        (2, 5, Fraction(10), False),
        (4, 4, Fraction(10), True),
        (100, 0, Fraction(10), False),
        (100, 1, Fraction(10), False),
        (100, 2, Fraction(10), False),
    ],
)
def test_streamgrabber_fps_skip_frames(fps, frame_index, framerate, expected):
    s = video.StreamGrabber(STREAM_NAME, "", None)
    s.fps = fps
    assert s._skip_frame(frame_index, framerate) == expected


@pytest.mark.parametrize("fps, expected", [(10, 10), (0, None), (10.3, None), (25, 25)])
def test_streamgrabber_allows_fps_values(fps, expected):
    s = video.StreamGrabber(STREAM_NAME, "", None)
    s.fps = fps
    assert s.fps == expected


def test_get_frames_should_decode_mp4():
    count = 0
    frame = None
    for f in video.get_frames("fixtures/sample.mp4"):
        count += 1
        frame = f[1]  # Returns tuple of (codec context, frame)

    assert count == 50
    assert frame.width == 640
    assert frame.height == 480

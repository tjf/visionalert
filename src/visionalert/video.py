import logging
import threading
import time

import av
import cv2
import fpstimer


def get_frames(location, connection_timeout=None, read_timeout=None, fps=None, seek=0):
    """
    Generator that retrieves frames from a libavformat-compatible location.

    :param location: Path or URL containing encoded video.  In short, if it works with FFMPEG, it
    probably will here, too.  If an RTSP URL is provided, we try to force the protocol to TCP due
    to an issue with high resolution video and UDP.  The default UDP buffer size allocated by the
    underlying library just isn't enough to handle an entire high resolution frame.
    :param connection_timeout: TCP connection timeout for remote hosts
    :param read_timeout: Socket read timeout for remote hosts
    :param fps: Return this many frames per second, dropping the others.
    :param seek: Seek this many seconds ahead before returning frames, if possible.
    """

    container = av.open(
        location,
        options={"rtsp_flags": "prefer_tcp", "rtsp_transport": "tcp"},
        timeout=(connection_timeout, read_timeout),
    )

    stream = container.streams.video[0]  # Only care about the first video stream
    stream.thread_type = "AUTO"

    if seek:
        container.seek(int(seek / stream.time_base), stream=stream)

    if fps and fps > stream.guessed_rate:
        raise ValueError(
            f"Requested FPS {fps} is higher than stream supports {float(stream.guessed_rate)}"
        )

    frame_index = -1
    for frame in container.decode(video=0):
        frame_index += 1
        if fps and frame_index % (int(stream.guessed_rate / fps)) >= 1:
            continue

        yield frame

    container.close()


def view_frames(frames, fps):
    """Display frames at given fps using OpenCV"""
    timer = fpstimer.FPSTimer(fps)
    for frame in frames:
        cv2.imshow("Viewer", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        timer.sleep()


class Camera:
    def __init__(self, name, url, on_frame, fps=None):
        self.name = name
        self.url = url
        self.fps = fps
        self.retry_wait = 1
        self.connection_timeout = 3.0
        self.read_timeout = 3.0
        self.on_frame = on_frame

        self._capture_thread = threading.Thread(
            name=f"Camera-{self.name}", daemon=True, target=self._capture_loop
        )

    def start(self):
        self._capture_thread.start()

    def _capture_loop(self):
        while True:
            try:
                logging.info(f"Connecting to camera {self.name}")
                for frame in get_frames(
                    self.url,
                    connection_timeout=self.connection_timeout,
                    read_timeout=self.read_timeout,
                    fps=self.fps,
                ):
                    self.on_frame(self.name, frame)

            except Exception as e:
                logging.error(
                    f"Error encountered reading frames from {self.name}: {e}.  "
                    f"Trying again in {self.retry_wait} second(s)."
                )
                time.sleep(self.retry_wait)

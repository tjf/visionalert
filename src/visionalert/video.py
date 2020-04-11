import logging
import threading
import time

import av
import cv2
import fpstimer


def get_frames(location, connection_timeout=None, read_timeout=None):
    """
    Generator that retrieves frames from an libavformat-compatible location.

    :param location: Path or URL containing encoded video.  In short, if it works with FFMPEG, it
    probably will here, too.  If an RTSP URL is provided, we try to force the protocol to TCP due
    to an issue with high resolution video and UDP.  The default UDP buffer size allocated by the
    underlying library just isn't enough to handle an entire high resolution frame.
    :param connection_timeout: TCP connection timeout for remote hosts
    :param read_timeout: Socket read timeout for remote hosts
    """
    container = av.open(
        location,
        options={"rtsp_flags": "prefer_tcp"},
        timeout=(connection_timeout, read_timeout),
    )
    stream = container.streams.video[0]  # Only care about the first video stream
    stream.thread_type = "AUTO"
    for frame in container.decode(video=0):
        yield stream.guessed_rate, frame
    container.close()


def view_frames(frames, fps):
    """Display frames at given fps using OpenCV"""
    timer = fpstimer.FPSTimer(fps)
    for frame in frames:
        cv2.imshow("Viewer", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        timer.sleep()


class StreamGrabber(threading.Thread):
    def __init__(self, stream_name, location, on_frame):
        super(StreamGrabber, self).__init__(
            name=f"{self.__class__.__name__}-{stream_name}", daemon=True
        )

        self.stream_name = stream_name
        self.location = location
        self.on_frame = on_frame

        self.retry_sleep_seconds = 1
        self.retry = True
        self.connection_timeout = 3.0
        self.read_timeout = 3.0

        self._fps = None
        self._last_frame = None
        self._last_frame_lock = threading.RLock()
        self._stop_requested = False
        self._debug = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        """Runs in a daemon thread to receive and store the most recent frame from the camera"""
        while not self._stop_requested:
            try:
                logging.info(
                    f"Opening stream {self.stream_name} at {self.location}"  # TODO scrub password
                )
                frame_gen = get_frames(
                    self.location,
                    connection_timeout=self.connection_timeout,
                    read_timeout=self.read_timeout,
                )

                self._process_frames(frame_gen)

            except Exception as e:
                logging.error(
                    f"Error encountered reading frames from stream {self.name}: %s.",
                    e,
                    exc_info=self._debug,
                )

            if not self.retry:
                break
            else:
                logging.info(
                    f"Retrying stream {self.stream_name} in {self.retry_sleep_seconds} seconds"
                )
                time.sleep(self.retry_sleep_seconds)

        logging.info(f"Stream {self.stream_name} complete. Exiting.")

    def _process_frames(self, frame_gen):
        current_frame_index = -1
        for framerate, frame in frame_gen:
            current_frame_index += 1

            if self._stop_requested:
                break

            # Skip frames to meet configures fps
            if self._skip_frame(current_frame_index, framerate):
                continue
            else:
                with self._last_frame_lock:
                    self._last_frame = frame
                self.on_frame(self.stream_name, self._last_frame)

    def _skip_frame(self, index, framerate):
        if not self.fps:
            return False
        elif self.fps > framerate:
            if index == 0:
                logging.warning(
                    f"Configured FPS of {self.fps} is higher than stream {self.stream_name} supports ({float(framerate)})"
                )
            return False
        elif self.fps == framerate:
            return False
        elif index % (int(framerate) / self.fps) >= 1:
            return True
        else:
            return False

    @property
    def last_frame(self):
        with self._last_frame_lock:
            return self._last_frame

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, fps):
        if not isinstance(fps, int) or fps < 1:
            logging.warning(
                f"Invalid fps value {fps} for stream {self.name}.  Using native framerate."
            )
        else:
            self._fps = fps

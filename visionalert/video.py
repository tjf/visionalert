import logging
import threading
import time

import av


def get_frames(location, connection_timeout=None, read_timeout=None):
    container = av.open(
        location,
        options={"rtsp_flags": "prefer_tcp"},
        timeout=(connection_timeout, read_timeout),
    )
    container.streams.video[0].thread_type = "AUTO"
    for frame in container.decode(video=0):
        yield frame
    container.close()


class StreamGrabber(threading.Thread):
    def __init__(self, stream_name, location, on_frame):
        super(StreamGrabber, self).__init__(
            name=f"{self.__class__.__name__}-{stream_name}", daemon=True
        )

        self.stream_name = stream_name
        self.location = location
        self.on_frame = on_frame

        self.every_nth = 1
        self.retry_sleep_seconds = 1
        self.retry = True
        self.connection_timeout = 3.0
        self.read_timeout = 3.0

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

            finally:
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
        for frame in frame_gen:
            current_frame_index += 1

            if self._stop_requested:
                break

            # Only process every Nth frame, skip others
            if self.every_nth != 1 and current_frame_index % self.every_nth:
                continue
            else:
                with self._last_frame_lock:
                    self._last_frame = frame
                self.on_frame(self.stream_name, self._last_frame)

    @property
    def last_frame(self):
        with self._last_frame_lock:
            return self._last_frame

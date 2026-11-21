import logging
import os
import queue
import threading
import wave
from const import TMP_DIR

logger = logging.getLogger(__name__)

STREAM_INPUT_FILE = os.path.join(TMP_DIR, "in.wav")
STREAM_OUTPUT_FILE = os.path.join(TMP_DIR, "out.wav")

_STOP = object()


class IORecorder:
    def __init__(self, input_sampling_rate: int, output_sampling_rate: int):
        self.fi = None
        self.fo = None
        self._queue: queue.Queue | None = None
        self._writer_thread: threading.Thread | None = None

    def _clear(self):
        self.close()
        self._clearFile(STREAM_INPUT_FILE)
        self._clearFile(STREAM_OUTPUT_FILE)

    def _clearFile(self, filename: str):
        if os.path.exists(filename):
            logger.info(f"Removing old recording file {filename}")
            os.remove(filename)

    def _writer_loop(self, q: queue.Queue, fi, fo):
        try:
            while True:
                item = q.get()
                if item is _STOP:
                    break
                target, wav = item
                try:
                    if target == "in" and fi is not None:
                        fi.writeframes(wav)
                    elif target == "out" and fo is not None:
                        fo.writeframes(wav)
                except Exception as e:
                    logger.warning(f"IORecorder write error ({target}): {e}")
        finally:
            if fi is not None:
                try:
                    fi.close()
                except Exception:
                    pass
            if fo is not None:
                try:
                    fo.close()
                except Exception:
                    pass

    def open(self, input_sampling_rate: int, output_sampling_rate: int):
        self._clear()

        fi = wave.open(STREAM_INPUT_FILE, "wb")
        fi.setnchannels(1)
        fi.setsampwidth(2)
        fi.setframerate(input_sampling_rate)

        fo = wave.open(STREAM_OUTPUT_FILE, "wb")
        fo.setnchannels(1)
        fo.setsampwidth(2)
        fo.setframerate(output_sampling_rate)

        self.fi = fi
        self.fo = fo
        self._queue = queue.Queue(maxsize=256)
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            args=(self._queue, fi, fo),
            name="IORecorderWriter",
            daemon=True,
        )
        self._writer_thread.start()
        logger.info(f"-------------------------- - - - {STREAM_INPUT_FILE}, {STREAM_OUTPUT_FILE}")

    def write_input(self, wav):
        q = self._queue
        if q is None:
            return
        try:
            q.put_nowait(("in", wav))
        except queue.Full:
            pass

    def write_output(self, wav):
        q = self._queue
        if q is None:
            return
        try:
            q.put_nowait(("out", wav))
        except queue.Full:
            pass

    def close(self):
        q = self._queue
        t = self._writer_thread
        self._queue = None
        self._writer_thread = None

        if q is not None:
            while True:
                try:
                    q.put_nowait(_STOP)
                    break
                except queue.Full:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
        if t is not None and t.is_alive():
            t.join(timeout=2.0)

        self.fi = None
        self.fo = None

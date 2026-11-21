import sounddevice as sd
from dataclasses import dataclass, field
import sys
import threading
import numpy as np

from const import ServerAudioDeviceType
import logging

# from const import SERVER_DEVICE_SAMPLE_RATES

logger = logging.getLogger(__name__)

# Serialize PortAudio lifecycle and device queries across threads because
# sd._terminate() / sd._initialize() temporarily puts PortAudio into an
# uninitialized state where concurrent sd.query_devices() raises PaErrorCode -10000.
PORTAUDIO_LOCK = threading.RLock()


def reinitialize_portaudio() -> None:
    """Safely terminate and re-initialize PortAudio under PORTAUDIO_LOCK."""
    with PORTAUDIO_LOCK:
        try:
            sd._terminate()
        finally:
            sd._initialize()


@dataclass
class ServerAudioDevice:
    index: int = 0
    name: str = ""
    hostAPI: str = ""
    maxInputChannels: int = 0
    maxOutputChannels: int = 0
    default_samplerate: int = 0
    # available_samplerates: list[int] = field(default_factory=lambda: [])


def dummy_callback(data: np.ndarray, frames, times, status):
    pass


def checkSamplingRate(deviceId: int, desiredSamplingRate: int, type: ServerAudioDeviceType):
    with PORTAUDIO_LOCK:
        if type == "input":
            try:
                with sd.InputStream(
                    device=deviceId,
                    callback=dummy_callback,
                    dtype="float32",
                    samplerate=desiredSamplingRate,
                ):
                    pass
                return True
            except Exception as e:  # NOQA
                logger.warning(f"[checkSamplingRate] {e}")
                return False
        else:
            try:
                with sd.OutputStream(
                    device=deviceId,
                    callback=dummy_callback,
                    dtype="float32",
                    samplerate=desiredSamplingRate,
                ):
                    pass
                return True
            except Exception as e:  # NOQA
                logger.warning(f"[checkSamplingRate] {e}")
                return False


def list_audio_device():
    with PORTAUDIO_LOCK:
        try:
            audioDeviceList = sd.query_devices()
        except Exception as e:
            logger.exception(e)
            raise e

        inputAudioDeviceList = [d for d in audioDeviceList if d["max_input_channels"] > 0]
        outputAudioDeviceList = [d for d in audioDeviceList if d["max_output_channels"] > 0]
        hostapis = sd.query_hostapis()

        serverAudioInputDevices: list[ServerAudioDevice] = []
        serverAudioOutputDevices: list[ServerAudioDevice] = []
        for d in inputAudioDeviceList:
            host_api_name = hostapis[d["hostapi"]]["name"]
            if sys.platform == "win32" and host_api_name not in ["Windows WASAPI", "ASIO"]:
                continue
            serverInputAudioDevice: ServerAudioDevice = ServerAudioDevice(
                index=d["index"],
                name=d["name"],
                hostAPI=host_api_name,
                maxInputChannels=d["max_input_channels"],
                maxOutputChannels=d["max_output_channels"],
                default_samplerate=d["default_samplerate"],
            )
            serverAudioInputDevices.append(serverInputAudioDevice)
        for d in outputAudioDeviceList:
            host_api_name = hostapis[d["hostapi"]]["name"]
            if sys.platform == "win32" and host_api_name not in ["Windows WASAPI", "ASIO"]:
                continue
            serverOutputAudioDevice: ServerAudioDevice = ServerAudioDevice(
                index=d["index"],
                name=d["name"],
                hostAPI=host_api_name,
                maxInputChannels=d["max_input_channels"],
                maxOutputChannels=d["max_output_channels"],
                default_samplerate=d["default_samplerate"],
            )
            serverAudioOutputDevices.append(serverOutputAudioDevice)

        return serverAudioInputDevices, serverAudioOutputDevices

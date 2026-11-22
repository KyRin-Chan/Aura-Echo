import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import time
import zipfile
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urljoin, urlsplit, urlunsplit

import aiohttp

from const import TENSORRT_LIBS_DIR, TMP_DIR
from downloader.HttpClient import HttpClient

logger = logging.getLogger(__name__)

TENSORRT_VERSION = "10.8.0.43"
TENSORRT_INDEX_URL = "https://pypi.nvidia.com/tensorrt-cu12-libs/"
TENSORRT_WHEEL_PREFIX = f"tensorrt_cu12_libs-{TENSORRT_VERSION}-"
TENSORRT_WHEEL_SUFFIX = "win_amd64.whl"
REQUIRED_CORE_DLLS = ("nvinfer_10.dll", "nvinfer_plugin_10.dll", "nvonnxparser_10.dll")
PENDING_DELETE_MARKER = ".pending_delete"
MANIFEST_FILE = ".installed.json"


class _WheelAnchorParser(HTMLParser):
    """Minimal HTML anchor collector for PEP 503 simple package index pages."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


def _parse_wheel_from_index(
    html_text: str, base_url: str = TENSORRT_INDEX_URL
) -> Tuple[str, str, Optional[str]]:
    """Parse a PEP 503 simple index HTML page for the matching Windows TensorRT wheel.

    Returns:
        (download_url_without_fragment, wheel_filename, optional_sha256_hex)
    """
    parser = _WheelAnchorParser()
    parser.feed(html_text)
    parser.close()

    prefix_lower = TENSORRT_WHEEL_PREFIX.lower()
    suffix_lower = TENSORRT_WHEEL_SUFFIX.lower()

    for href in parser.hrefs:
        full_url = urljoin(base_url, href)
        parts = urlsplit(full_url)
        filename = os.path.basename(unquote(parts.path))
        filename_lower = filename.lower()
        if filename_lower.startswith(prefix_lower) and filename_lower.endswith(suffix_lower):
            sha256_hex: Optional[str] = None
            if parts.fragment:
                frag_params = parse_qs(parts.fragment)
                sha_list = frag_params.get("sha256")
                if sha_list:
                    sha256_hex = sha_list[0].strip().lower()
            clean_url = urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))
            return clean_url, filename, sha256_hex

    raise RuntimeError(
        f"No matching wheel found for {TENSORRT_WHEEL_PREFIX}*{TENSORRT_WHEEL_SUFFIX} at {base_url}"
    )


class _DownloadCancelled(Exception):
    """Internal sentinel used to abort blocking worker threads cleanly."""


class TensorRTManager:
    """Manages on-demand downloading, installation, and removal of TensorRT runtime DLLs."""

    _instance: Optional["TensorRTManager"] = None

    def __init__(self, install_dir: str = TENSORRT_LIBS_DIR) -> None:
        self.install_dir = install_dir
        self.state: str = "idle"
        self.progress: float = 0.0
        self.downloaded_bytes: int = 0
        self.total_bytes: int = 0
        self.speed_bytes_per_sec: float = 0.0
        self.error: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self._cancel_requested: bool = False

    @classmethod
    def get_instance(cls) -> "TensorRTManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def apply_pending_delete(install_dir: str = TENSORRT_LIBS_DIR) -> bool:
        """Delete deferred TensorRT directory on startup before DLLs can be loaded."""
        marker_path = os.path.join(install_dir, PENDING_DELETE_MARKER)
        if not os.path.exists(marker_path):
            return False

        logger.info(f"Applying deferred deletion of TensorRT runtime directory: {install_dir}")
        shutil.rmtree(install_dir, ignore_errors=True)
        trt_cache_dir = os.path.join(TMP_DIR, "trt_cache")
        if os.path.exists(trt_cache_dir):
            shutil.rmtree(trt_cache_dir, ignore_errors=True)
        return True

    def is_managed_installed(self) -> bool:
        """Return True if the managed TensorRT directory exists with all core DLLs."""
        if not os.path.isdir(self.install_dir):
            return False
        if os.path.exists(os.path.join(self.install_dir, PENDING_DELETE_MARKER)):
            return False

        try:
            entries = {
                entry.lower()
                for entry in os.listdir(self.install_dir)
                if os.path.isfile(os.path.join(self.install_dir, entry))
            }
        except OSError:
            return False

        return all(dll in entries for dll in REQUIRED_CORE_DLLS)

    def _read_manifest_version(self) -> str:
        manifest_path = os.path.join(self.install_dir, MANIFEST_FILE)
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ver = data.get("version")
                if isinstance(ver, str) and ver.strip():
                    return ver.strip()
            except Exception:
                pass
        return TENSORRT_VERSION

    def _compute_installed_size_bytes(self) -> int:
        if not os.path.isdir(self.install_dir):
            return 0
        total = 0
        try:
            for name in os.listdir(self.install_dir):
                if name.lower().endswith(".dll"):
                    full_path = os.path.join(self.install_dir, name)
                    if os.path.isfile(full_path):
                        total += os.path.getsize(full_path)
        except OSError:
            pass
        return total

    def get_status(self) -> Dict[str, Any]:
        from voice_changer.common.deviceManager.DeviceManager import DeviceManager

        dm = DeviceManager.get_instance()
        supported = dm.is_tensorrt_supported()
        installed = dm.is_tensorrt_available()
        managed = self.is_managed_installed()
        pending_delete = os.path.exists(os.path.join(self.install_dir, PENDING_DELETE_MARKER))
        source = "managed" if managed else ("system" if installed else "none")
        size_bytes = self._compute_installed_size_bytes() if (managed or pending_delete) else 0
        version = self._read_manifest_version()

        active_task = self._task is not None and not self._task.done()
        if active_task:
            effective_state = self.state
            progress = self.progress
        elif installed:
            effective_state = "installed"
            progress = 100.0
        elif self.state in ("error", "cancelled"):
            effective_state = self.state
            progress = self.progress
        else:
            effective_state = "idle"
            progress = 0.0

        return {
            "supported": supported,
            "installed": installed,
            "managed": managed,
            "source": source,
            "version": version,
            "targetVersion": TENSORRT_VERSION,
            "installDir": self.install_dir,
            "sizeBytes": size_bytes,
            "pendingDelete": pending_delete,
            "state": effective_state,
            "progress": round(progress, 2),
            "downloadedBytes": self.downloaded_bytes,
            "totalBytes": self.total_bytes,
            "speedBytesPerSec": round(self.speed_bytes_per_sec, 1),
            "error": self.error,
        }

    def _try_remove_install_dir(self) -> None:
        if not os.path.exists(self.install_dir):
            return
        for root, dirs, files in os.walk(self.install_dir, topdown=False):
            for fname in files:
                os.remove(os.path.join(root, fname))
            for dname in dirs:
                os.rmdir(os.path.join(root, dname))
        os.rmdir(self.install_dir)

    async def start_download(self) -> Dict[str, Any]:
        from voice_changer.common.deviceManager.DeviceManager import DeviceManager

        dm = DeviceManager.get_instance()
        if not dm.is_tensorrt_supported():
            raise ValueError(
                "TensorRT on-demand runtime is only supported on Windows with CUDA and TensorrtExecutionProvider."
            )

        if self._task is not None and not self._task.done():
            return self.get_status()

        marker_path = os.path.join(self.install_dir, PENDING_DELETE_MARKER)
        if os.path.exists(marker_path):
            try:
                self._try_remove_install_dir()
            except OSError as e:
                raise RuntimeError(
                    "TensorRT files from this session are pending removal. Please restart AuraEcho before reinstalling."
                ) from e

        if self.is_managed_installed():
            dm.refresh_tensorrt_libs()
            self.state = "installed"
            self.progress = 100.0
            self.speed_bytes_per_sec = 0.0
            self.error = None
            return self.get_status()

        self.progress = 0.0
        self.downloaded_bytes = 0
        self.total_bytes = 0
        self.speed_bytes_per_sec = 0.0
        self.error = None
        self.state = "resolving"
        self._cancel_requested = False

        self._task = asyncio.create_task(self._download_and_install())
        return self.get_status()

    async def _run_cancellable_thread(self, fn, *args):
        fut = asyncio.ensure_future(asyncio.to_thread(fn, *args))
        try:
            return await fut
        except asyncio.CancelledError:
            self._cancel_requested = True
            try:
                await asyncio.shield(fut)
            except Exception:
                pass
            raise

    def _verify_sha256_sync(self, file_path: str, expected_sha256: str) -> None:
        file_size = os.path.getsize(file_path)
        hasher = hashlib.sha256()
        verified_bytes = 0
        chunk_size = 4 * 1024 * 1024
        with open(file_path, "rb") as f:
            while True:
                if self._cancel_requested:
                    raise _DownloadCancelled()
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
                verified_bytes += len(chunk)
                if file_size > 0:
                    self.progress = min(100.0, verified_bytes / file_size * 100.0)
        actual = hasher.hexdigest().lower()
        if actual != expected_sha256.lower():
            raise RuntimeError(
                f"SHA-256 checksum mismatch (expected {expected_sha256.lower()}, got {actual})"
            )

    def _extract_wheel_sync(
        self, part_path: str, staging_dir: str, wheel_filename: str, expected_sha256: Optional[str]
    ) -> None:
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
        os.makedirs(staging_dir, exist_ok=True)
        os.makedirs(self.install_dir, exist_ok=True)

        extracted_names: List[str] = []
        with zipfile.ZipFile(part_path, "r") as zf:
            selected: Dict[str, Tuple[zipfile.ZipInfo, str]] = {}
            for info in zf.infolist():
                if info.is_dir():
                    continue
                normalized = info.filename.replace("\\", "/")
                basename = os.path.basename(normalized)
                if not basename or not basename.lower().endswith(".dll"):
                    continue
                key = basename.lower()
                if key not in selected:
                    selected[key] = (info, basename)

            missing_core = [dll for dll in REQUIRED_CORE_DLLS if dll not in selected]
            if missing_core:
                raise RuntimeError(
                    f"Wheel {wheel_filename} is missing required TensorRT DLL(s): {', '.join(missing_core)}"
                )

            total_uncompressed = sum(info.file_size for info, _ in selected.values())
            extracted_bytes = 0
            chunk_size = 4 * 1024 * 1024

            for info, basename in selected.values():
                if self._cancel_requested:
                    raise _DownloadCancelled()
                dst_stage = os.path.join(staging_dir, basename)
                with zf.open(info, "r") as src_f, open(dst_stage, "wb") as dst_f:
                    while True:
                        if self._cancel_requested:
                            raise _DownloadCancelled()
                        buf = src_f.read(chunk_size)
                        if not buf:
                            break
                        dst_f.write(buf)
                        extracted_bytes += len(buf)
                        if total_uncompressed > 0:
                            self.progress = min(100.0, extracted_bytes / total_uncompressed * 100.0)
                extracted_names.append(basename)

        for basename in extracted_names:
            src_stage = os.path.join(staging_dir, basename)
            dst_final = os.path.join(self.install_dir, basename)
            os.replace(src_stage, dst_final)

        manifest = {
            "version": TENSORRT_VERSION,
            "wheel": wheel_filename,
            "sha256": expected_sha256,
            "files": sorted(extracted_names),
            "totalBytes": self._compute_installed_size_bytes(),
        }
        manifest_path = os.path.join(self.install_dir, MANIFEST_FILE)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    async def _download_and_install(self) -> None:
        from voice_changer.common.deviceManager.DeviceManager import DeviceManager

        download_dir = os.path.join(self.install_dir, ".download")
        staging_dir = os.path.join(self.install_dir, ".staging")
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=120)

        try:
            session = await HttpClient.get_client()

            # Step A: Resolve matching Windows wheel from NVIDIA simple index
            self.state = "resolving"
            async with session.get(TENSORRT_INDEX_URL, allow_redirects=True, timeout=timeout) as idx_res:
                idx_res.raise_for_status()
                html_text = await idx_res.text()
            wheel_url, wheel_filename, expected_sha256 = _parse_wheel_from_index(
                html_text, TENSORRT_INDEX_URL
            )

            # Step B: Download wheel (with Range resume)
            self.state = "downloading"
            os.makedirs(download_dir, exist_ok=True)
            part_path = os.path.join(download_dir, f"{wheel_filename}.part")

            offset = os.path.getsize(part_path) if os.path.exists(part_path) else 0
            skip_download = False

            while not skip_download:
                if self._cancel_requested:
                    raise asyncio.CancelledError()

                headers = {"Range": f"bytes={offset}-"} if offset > 0 else {}
                async with session.get(
                    wheel_url, headers=headers, allow_redirects=True, timeout=timeout
                ) as res:
                    if res.status == 416 and offset > 0:
                        content_range = res.headers.get("Content-Range", "")
                        match = re.search(r"\*/(\d+)", content_range)
                        if match and offset == int(match.group(1)):
                            self.total_bytes = int(match.group(1))
                            self.downloaded_bytes = offset
                            self.progress = 100.0
                            skip_download = True
                            break
                        if os.path.exists(part_path):
                            os.remove(part_path)
                        offset = 0
                        continue

                    res.raise_for_status()

                    if res.status == 206 and offset > 0:
                        content_range = res.headers.get("Content-Range", "")
                        match = re.search(r"/(\d+)", content_range)
                        if match:
                            self.total_bytes = int(match.group(1))
                        else:
                            cl = int(res.headers.get("Content-Length", 0) or 0)
                            self.total_bytes = offset + cl
                        self.downloaded_bytes = offset
                        open_mode = "ab"
                    else:
                        offset = 0
                        self.total_bytes = int(res.headers.get("Content-Length", 0) or 0)
                        self.downloaded_bytes = 0
                        open_mode = "wb"

                    if self.total_bytes > 0:
                        self.progress = min(100.0, self.downloaded_bytes / self.total_bytes * 100.0)

                    last_tick = time.monotonic()
                    bytes_since_tick = 0
                    with open(part_path, open_mode) as f:
                        async for chunk in res.content.iter_chunked(1024 * 1024):
                            if self._cancel_requested:
                                raise asyncio.CancelledError()
                            f.write(chunk)
                            chunk_len = len(chunk)
                            self.downloaded_bytes += chunk_len
                            bytes_since_tick += chunk_len
                            if self.total_bytes > 0:
                                self.progress = min(
                                    100.0, self.downloaded_bytes / self.total_bytes * 100.0
                                )

                            now = time.monotonic()
                            elapsed = now - last_tick
                            if elapsed >= 0.25:
                                inst_speed = bytes_since_tick / elapsed
                                if self.speed_bytes_per_sec <= 0:
                                    self.speed_bytes_per_sec = inst_speed
                                else:
                                    self.speed_bytes_per_sec = (
                                        0.3 * inst_speed + 0.7 * self.speed_bytes_per_sec
                                    )
                                last_tick = now
                                bytes_since_tick = 0
                    break

            # Step C: Verify SHA-256 checksum
            if expected_sha256:
                self.state = "verifying"
                self.speed_bytes_per_sec = 0.0
                self.progress = 0.0
                try:
                    await self._run_cancellable_thread(
                        self._verify_sha256_sync, part_path, expected_sha256
                    )
                except _DownloadCancelled:
                    raise asyncio.CancelledError()
                except RuntimeError:
                    if os.path.exists(part_path):
                        os.remove(part_path)
                    raise

            # Step D: Extract TensorRT DLLs into staging and move atomically into install_dir
            self.state = "extracting"
            self.speed_bytes_per_sec = 0.0
            self.progress = 0.0
            try:
                await self._run_cancellable_thread(
                    self._extract_wheel_sync, part_path, staging_dir, wheel_filename, expected_sha256
                )
            except _DownloadCancelled:
                raise asyncio.CancelledError()

            shutil.rmtree(download_dir, ignore_errors=True)
            shutil.rmtree(staging_dir, ignore_errors=True)

            # Step E: Register DLLs immediately in running process
            if not DeviceManager.get_instance().refresh_tensorrt_libs():
                raise RuntimeError("TensorRT DLLs were extracted but could not be registered.")

            self.state = "installed"
            self.progress = 100.0
            self.speed_bytes_per_sec = 0.0
            self.error = None
            logger.info(f"TensorRT {TENSORRT_VERSION} runtime installed to {self.install_dir}")

        except asyncio.CancelledError:
            self.state = "cancelled"
            self.speed_bytes_per_sec = 0.0
            self.error = None
            shutil.rmtree(download_dir, ignore_errors=True)
            shutil.rmtree(staging_dir, ignore_errors=True)
            if os.path.isdir(self.install_dir) and not os.listdir(self.install_dir):
                try:
                    os.rmdir(self.install_dir)
                except OSError:
                    pass
            logger.info("TensorRT download cancelled.")
        except Exception as e:
            logger.exception(f"Failed to download/install TensorRT runtime: {e}")
            self.state = "error"
            self.speed_bytes_per_sec = 0.0
            self.error = str(e)
            shutil.rmtree(staging_dir, ignore_errors=True)

    async def cancel_download(self) -> Dict[str, Any]:
        self._cancel_requested = True
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        return self.get_status()

    def delete_runtime(self, voice_changer_manager: Any) -> Dict[str, Any]:
        if self._task is not None and not self._task.done():
            raise ValueError("Cannot delete TensorRT runtime while a download is in progress.")

        if voice_changer_manager is not None and getattr(voice_changer_manager.settings, "useTensorRT", 0) == 1:
            raise ValueError("Disable TensorRT Acceleration before deleting the TensorRT runtime.")

        trt_cache_dir = os.path.join(TMP_DIR, "trt_cache")
        if os.path.exists(trt_cache_dir):
            shutil.rmtree(trt_cache_dir, ignore_errors=True)

        if not os.path.exists(self.install_dir):
            self.state = "idle"
            self.progress = 0.0
            self.downloaded_bytes = 0
            self.total_bytes = 0
            self.speed_bytes_per_sec = 0.0
            self.error = None
            return {
                **self.get_status(),
                "status": "deleted",
                "message": "TensorRT runtime was not found on disk.",
            }

        try:
            self._try_remove_install_dir()
            self.state = "idle"
            self.progress = 0.0
            self.downloaded_bytes = 0
            self.total_bytes = 0
            self.speed_bytes_per_sec = 0.0
            self.error = None
            return {
                **self.get_status(),
                "status": "deleted",
                "message": "TensorRT runtime deleted successfully.",
            }
        except OSError as e:
            logger.warning(f"Could not delete TensorRT DLLs immediately ({e}); marking for removal on restart.")
            os.makedirs(self.install_dir, exist_ok=True)
            marker_path = os.path.join(self.install_dir, PENDING_DELETE_MARKER)
            with open(marker_path, "w", encoding="utf-8") as f:
                f.write("pending_delete\n")
            return {
                **self.get_status(),
                "status": "pending_restart",
                "message": "TensorRT DLLs are currently loaded in memory and will be automatically deleted on next restart.",
            }

import hashlib
import logging
import os
from time import perf_counter, time

import torch

from const import ROOT_PATH

logger = logging.getLogger(__name__)

_JIT_CACHE_MAX_FILES = 10
_JIT_CACHE_MAX_BYTES = 1024 * 1024 * 1024  # 1 GiB
_JIT_CACHE_TMP_MAX_AGE_SEC = 600


def get_jit_cache_path(
    namespace: str,
    file: str,
    dev: torch.device,
    target_dtype: torch.dtype,
) -> str | None:
    """Compute a deterministic TorchScript disk cache path for a checkpoint + device + dtype."""
    try:
        abs_path = os.path.abspath(file)
        st = os.stat(abs_path)
        cuda_cap = ""
        if dev.type == "cuda":
            major, minor = torch.cuda.get_device_capability(dev)
            cuda_cap = f"sm_{major}{minor}"
        key = "|".join([
            namespace,
            abs_path,
            str(st.st_size),
            str(st.st_mtime_ns),
            dev.type,
            cuda_cap,
            str(target_dtype),
            torch.__version__,
        ])
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return os.path.join(ROOT_PATH, "tmp_dir", "jit_cache", f"{digest}.ts")
    except Exception:
        return None


def prune_jit_cache(
    cache_dir: str,
    keep_path: str | None = None,
    max_files: int = _JIT_CACHE_MAX_FILES,
    max_bytes: int = _JIT_CACHE_MAX_BYTES,
) -> int:
    """Enforce LRU limits on `.ts` cache entries and clean up stale `.tmp` files."""
    if not os.path.isdir(cache_dir):
        return 0

    now = time()
    keep_norm = os.path.normcase(os.path.abspath(keep_path)) if keep_path else None
    entries: list[tuple[bool, int, int, str]] = []

    try:
        with os.scandir(cache_dir) as it:
            for entry in it:
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    continue

                if entry.name.endswith(".tmp"):
                    if now - st.st_mtime > _JIT_CACHE_TMP_MAX_AGE_SEC:
                        try:
                            os.remove(entry.path)
                        except OSError:
                            pass
                    continue

                if entry.name.endswith(".ts"):
                    is_keep = (
                        keep_norm is not None
                        and os.path.normcase(os.path.abspath(entry.path)) == keep_norm
                    )
                    entries.append((is_keep, st.st_mtime_ns, st.st_size, entry.path))
    except OSError:
        return 0

    # Keep the newly written entry first, then sort remaining entries newest-first (LRU).
    entries.sort(key=lambda item: (item[0], item[1]), reverse=True)

    kept_count = 0
    kept_bytes = 0
    removed_count = 0
    removed_bytes = 0

    for is_keep, _, size, path in entries:
        over_limit = kept_count >= max_files or (kept_bytes + size > max_bytes)
        if not is_keep and over_limit:
            try:
                os.remove(path)
                removed_count += 1
                removed_bytes += size
            except OSError:
                pass
        else:
            kept_count += 1
            kept_bytes += size

    if removed_count > 0:
        logger.info(
            f"Pruned {removed_count} old JIT cache file(s) "
            f"({removed_bytes / (1024 * 1024):.1f} MB) from {cache_dir}"
        )
    return removed_count


def load_jit_cache(
    cache_path: str | None,
    dev: torch.device,
    extra_files: dict[str, str] | None = None,
    required_attr: str | None = None,
) -> tuple[torch.jit.ScriptModule, dict, float] | None:
    """Load a cached TorchScript module from disk, updating its mtime for LRU tracking."""
    if not cache_path or not os.path.isfile(cache_path):
        return None

    t0 = perf_counter()
    loaded_extra = dict(extra_files) if extra_files is not None else {}
    try:
        if extra_files is not None:
            model = torch.jit.load(cache_path, map_location=dev, _extra_files=loaded_extra)
        else:
            model = torch.jit.load(cache_path, map_location=dev)

        if required_attr and not hasattr(model, required_attr):
            raise RuntimeError(f"Cached TorchScript module is missing '{required_attr}' method")

        try:
            os.utime(cache_path, None)
        except OSError:
            pass

        elapsed_ms = (perf_counter() - t0) * 1000.0
        return model, loaded_extra, elapsed_ms
    except Exception as e:
        logger.warning(f"Failed to load cached JIT model ({e}); recompiling.")
        try:
            os.remove(cache_path)
        except OSError:
            pass
        return None


def save_jit_cache(
    model: torch.jit.ScriptModule,
    cache_path: str | None,
    extra_files: dict[str, str] | None = None,
) -> bool:
    """Atomically save a compiled TorchScript module to disk and prune old cache entries."""
    if not cache_path:
        return False

    tmp_path = f"{cache_path}.{os.getpid()}.tmp"
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if extra_files is not None:
            torch.jit.save(model, tmp_path, _extra_files=extra_files)
        else:
            torch.jit.save(model, tmp_path)
        os.replace(tmp_path, cache_path)
        prune_jit_cache(os.path.dirname(cache_path), keep_path=cache_path)
        return True
    except Exception as cache_err:
        logger.warning(f"Failed to save JIT model cache: {cache_err}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return False

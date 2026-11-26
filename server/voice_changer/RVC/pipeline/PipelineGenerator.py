import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import faiss
import faiss.contrib.torch_utils
import torch
from data.ModelSlot import RVCModelSlot
from typing import Callable, Optional

from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.embedder.EmbedderManager import EmbedderManager
from voice_changer.RVC.inferencer.InferencerManager import InferencerManager
from voice_changer.RVC.pipeline.Pipeline import Pipeline
from voice_changer.pitch_extractor.PitchExtractorManager import PitchExtractorManager
from voice_changer.audio_effects.AudioEffectsManager import AudioEffectsManager
from settings import get_settings

import logging
logger = logging.getLogger(__name__)

_TOTAL_STEPS = 5

def createPipeline(
    modelSlot: RVCModelSlot,
    f0Detector: str,
    useONNX: bool,
    force_reload: bool,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    audio_effects_manager: Optional[AudioEffectsManager] = None,
    total_steps: int = _TOTAL_STEPS,
):
    def _report(step: int, label: str):
        if progress_callback:
            try:
                progress_callback(step, total_steps, label)
            except Exception:
                pass

    model_dir = get_settings().model_dir

    if useONNX:
        inferencer_type = modelSlot.modelTypeOnnx
        model_path = os.path.join(model_dir, str(modelSlot.slotIndex), os.path.basename(modelSlot.modelFileOnnx))
    else:
        inferencer_type = modelSlot.modelType
        model_path = os.path.join(model_dir, str(modelSlot.slotIndex), os.path.basename(modelSlot.modelFile))
    index_path = os.path.join(model_dir, str(modelSlot.slotIndex), os.path.basename(modelSlot.indexFile))

    _report(1, "Loading voice model, embedder, pitch extractor & index...")

    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="PipelineLoader") as executor:
        futures = {
            executor.submit(InferencerManager.getInferencer, inferencer_type, model_path): (
                "inferencer",
                "Voice model loaded",
            ),
            executor.submit(EmbedderManager.get_embedder, modelSlot.embedder, force_reload): (
                "embedder",
                "Embedder loaded",
            ),
            executor.submit(PitchExtractorManager.getPitchExtractor, f0Detector, force_reload): (
                "pitchExtractor",
                "Pitch extractor loaded",
            ),
            executor.submit(_loadIndex, index_path): (
                "index",
                "FAISS index loaded",
            ),
        }

        results = {}
        completed = 0
        for fut in as_completed(futures):
            key, label = futures[fut]
            results[key] = fut.result()
            completed += 1
            _report(completed, label)

    inferencer = results["inferencer"]
    embedder = results["embedder"]
    pitchExtractor = results["pitchExtractor"]
    index, index_reconstruct = results["index"]

    pipeline = Pipeline(
        embedder,
        inferencer,
        pitchExtractor,
        index,
        index_reconstruct,
        modelSlot.f0,
        modelSlot.samplingRate,
        modelSlot.embChannels,
        audio_effects_manager=audio_effects_manager,
    )

    return pipeline


def _loadIndex(indexPath: str) -> tuple[faiss.Index | None, torch.Tensor | None]:
    dev = DeviceManager.get_instance().device
    # Indexのロード
    logger.info("Loading index...")
    # ファイル指定があってもファイルがない場合はNone
    if os.path.exists(indexPath) is not True or os.path.isfile(indexPath) is not True:
        logger.warning("Index file not found. Index will not be used.")
        return (None, None)

    logger.info(f"Try loading \"{indexPath}\"...")
    try:
        index: faiss.IndexIVFFlat = faiss.read_index(indexPath)
        if not index.is_trained:
            logger.error("Invalid index. You MUST use added_xxxx.index, not trained_xxxx.index. Index will not be used.")
            return (None, None)
        if index.ntotal == 0:
            logger.error("Index is empty. Index will not be used.")
            return (None, None)
        # BUG: faiss-gpu does not support reconstruct on GPU indices
        # https://github.com/facebookresearch/faiss/issues/2181
        index_reconstruct = index.reconstruct_n(0, index.ntotal).to(dev)
        if sys.platform == 'linux' and '+cu' in torch.__version__ and dev.type == 'cuda':
            index: faiss.GpuIndexIVFFlat = faiss.index_cpu_to_gpus_list(index, gpus=[dev.index])
    except Exception as e: # NOQA
        logger.error("Load index failed. Index will not be used.")
        logger.exception(e)
        return (None, None)

    return index, index_reconstruct

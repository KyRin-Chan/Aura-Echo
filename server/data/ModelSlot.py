from typing import TypeAlias, Union
from const import MAX_SLOT_NUM, EnumInferenceTypes, EmbedderType, VoiceChangerType

from dataclasses import dataclass, asdict, field

import os
import json
import logging
logger = logging.getLogger(__name__)

from voice_changer.utils.ZipUtils import sanitize_filename

@dataclass
class ModelSlot:
    slotIndex: int = -1
    voiceChangerType: VoiceChangerType | None = None
    name: str = ""
    description: str = ""
    credit: str = ""
    termsOfUseUrl: str = ""
    iconFile: str = ""
    speakers: dict = field(default_factory=lambda: {})


@dataclass
class RVCModelSlot(ModelSlot):
    voiceChangerType: VoiceChangerType = "RVC"
    modelFile: str = ""
    modelFileOnnx: str = ""
    indexFile: str = ""
    defaultTune: float = 0.0
    defaultFormantShift: float = 0
    defaultIndexRatio: float = 0
    defaultProtect: float = 0.5
    isONNX: bool = False
    modelType: str = EnumInferenceTypes.pyTorchRVC.value
    modelTypeOnnx: str = EnumInferenceTypes.onnxRVC.value
    samplingRate: int = -1
    f0: bool = True
    embChannels: int = 256
    embOutputLayer: int = 9
    useFinalProj: bool = True
    deprecated: bool = False
    embedder: EmbedderType = "hubert_base"
    speakers: dict = field(default_factory=lambda: {0: "target"})
    version: str = "v2"

ModelSlots: TypeAlias = Union[
    ModelSlot,
    RVCModelSlot,
]


def loadSlotInfo(model_dir: str, slotIndex: int) -> ModelSlots:
    slotDir = os.path.join(model_dir, str(slotIndex))
    jsonFile = os.path.join(slotDir, "params.json")
    if not os.path.exists(jsonFile):
        return ModelSlot()
    with open(jsonFile, encoding="utf-8") as f:
        jsonDict = json.load(f)
    slotInfoKey = list(ModelSlot.__annotations__.keys())
    slotInfo = ModelSlot(**{k: v for k, v in jsonDict.items() if k in slotInfoKey})
    if slotInfo.voiceChangerType == "RVC":
        slotInfoKey.extend(list(RVCModelSlot.__annotations__.keys()))
        return RVCModelSlot(**{k: v for k, v in jsonDict.items() if k in slotInfoKey})
    else:
        return ModelSlot()


def sanitize_and_rename_slot_files(model_dir: str, slotInfo: ModelSlots):
    if not isinstance(slotInfo, RVCModelSlot):
        return
    
    slot_index = slotInfo.slotIndex
    slotDir = os.path.join(model_dir, str(slot_index))
    if not os.path.exists(slotDir):
        return
        
    changed = False
    
    # Check modelFile
    if slotInfo.modelFile:
        sanitized = sanitize_filename(slotInfo.modelFile)
        if sanitized != slotInfo.modelFile:
            src_path = os.path.join(slotDir, slotInfo.modelFile)
            dst_path = os.path.join(slotDir, sanitized)
            if os.path.exists(src_path):
                try:
                    logger.info(f"Renaming unsafe model file: {src_path} -> {dst_path}")
                    if os.path.exists(dst_path):
                        os.remove(dst_path)
                    os.rename(src_path, dst_path)
                    slotInfo.modelFile = sanitized
                    changed = True
                except Exception as e:
                    logger.warning(f"Failed to rename model file: {e}")
            else:
                if os.path.exists(dst_path):
                    slotInfo.modelFile = sanitized
                    changed = True
                    
    # Check modelFileOnnx
    if slotInfo.modelFileOnnx:
        sanitized = sanitize_filename(slotInfo.modelFileOnnx)
        if sanitized != slotInfo.modelFileOnnx:
            src_path = os.path.join(slotDir, slotInfo.modelFileOnnx)
            dst_path = os.path.join(slotDir, sanitized)
            if os.path.exists(src_path):
                try:
                    logger.info(f"Renaming unsafe ONNX file: {src_path} -> {dst_path}")
                    if os.path.exists(dst_path):
                        os.remove(dst_path)
                    os.rename(src_path, dst_path)
                    slotInfo.modelFileOnnx = sanitized
                    changed = True
                except Exception as e:
                    logger.warning(f"Failed to rename ONNX file: {e}")
            else:
                if os.path.exists(dst_path):
                    slotInfo.modelFileOnnx = sanitized
                    changed = True

    # Check indexFile
    if slotInfo.indexFile:
        sanitized = sanitize_filename(slotInfo.indexFile)
        if sanitized != slotInfo.indexFile:
            src_path = os.path.join(slotDir, slotInfo.indexFile)
            dst_path = os.path.join(slotDir, sanitized)
            if os.path.exists(src_path):
                try:
                    logger.info(f"Renaming unsafe index file: {src_path} -> {dst_path}")
                    if os.path.exists(dst_path):
                        os.remove(dst_path)
                    os.rename(src_path, dst_path)
                    slotInfo.indexFile = sanitized
                    changed = True
                except Exception as e:
                    logger.warning(f"Failed to rename index file: {e}")
            else:
                if os.path.exists(dst_path):
                    slotInfo.indexFile = sanitized
                    changed = True

    # Check iconFile
    if slotInfo.iconFile:
        sanitized = sanitize_filename(slotInfo.iconFile)
        if sanitized != slotInfo.iconFile:
            src_path = os.path.join(slotDir, slotInfo.iconFile)
            dst_path = os.path.join(slotDir, sanitized)
            if os.path.exists(src_path):
                try:
                    logger.info(f"Renaming unsafe icon file: {src_path} -> {dst_path}")
                    if os.path.exists(dst_path):
                        os.remove(dst_path)
                    os.rename(src_path, dst_path)
                    slotInfo.iconFile = sanitized
                    changed = True
                except Exception as e:
                    logger.warning(f"Failed to rename icon file: {e}")
            else:
                if os.path.exists(dst_path):
                    slotInfo.iconFile = sanitized
                    changed = True

    if changed:
        saveSlotInfo(model_dir, slot_index, slotInfo)


def loadAllSlotInfo(model_dir: str):
    slotInfos: list[ModelSlots] = []
    for slotIndex in range(MAX_SLOT_NUM):
        slotInfo = loadSlotInfo(model_dir, slotIndex)
        slotInfo.slotIndex = slotIndex  # スロットインデックスは動的に注入
        # Self-healing: sanitize and rename existing files if they contain unsafe characters
        sanitize_and_rename_slot_files(model_dir, slotInfo)
        slotInfos.append(slotInfo)
    return slotInfos


def saveSlotInfo(model_dir: str, slotIndex: int, slotInfo: ModelSlots):
    slotDir = os.path.join(model_dir, str(slotIndex))
    logger.info(f"SlotInfo::: {slotInfo}")
    slotInfoDict = asdict(slotInfo)
    with open(os.path.join(slotDir, "params.json"), "w") as f:
        json.dump(slotInfoDict, f, indent=4)

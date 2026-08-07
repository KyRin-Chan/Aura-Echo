import os
import json
import logging
from typing import Dict, List, Any, Optional
from const import FORMANT_PROFILES_DIR

logger = logging.getLogger(__name__)

class FormantProfileManager:
    _instance: Optional['FormantProfileManager'] = None

    def __init__(self):
        self.profiles_dir = os.path.join(FORMANT_PROFILES_DIR, "profiles")
        os.makedirs(self.profiles_dir, exist_ok=True)

    @classmethod
    def get_instance(cls) -> 'FormantProfileManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_slot_manager(self):
        try:
            from voice_changer.VoiceChangerManager import VoiceChangerManager
            vc_mgr = VoiceChangerManager.get_instance()
            return vc_mgr.modelSlotManager
        except Exception as e:
            logger.warning(f"Could not get ModelSlotManager: {e}")
            return None

    # ---------------- Profiles (Saved in server/formant_profiles/profiles/) ----------------

    def get_all_profiles(self) -> List[Dict[str, Any]]:
        profiles = []
        if not os.path.exists(self.profiles_dir):
            return profiles
        
        for filename in os.listdir(self.profiles_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.profiles_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict) and "id" in data:
                            profiles.append(data)
                except Exception as e:
                    logger.error(f"Error reading profile file {filename}: {e}")
        return profiles

    def get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        filepath = os.path.join(self.profiles_dir, f"profile_{profile_id}.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading profile {profile_id}: {e}")
        return None

    def save_profile(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        profile_id = profile_data.get("id")
        if not profile_id:
            raise ValueError("Profile data must contain an 'id' field")
        
        filepath = os.path.join(self.profiles_dir, f"profile_{profile_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved formant profile profile_{profile_id}.json")
        return profile_data

    def delete_profile(self, profile_id: str) -> bool:
        filepath = os.path.join(self.profiles_dir, f"profile_{profile_id}.json")
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Deleted formant profile profile_{profile_id}.json")
            except Exception as e:
                logger.error(f"Error deleting profile {profile_id}: {e}")
                return False

        # Remove profile reference from model slot params.json
        bindings = self.get_all_bindings()
        for slot_index, binding in bindings.items():
            if binding.get("inputProfileId") == profile_id or binding.get("targetProfileId") == profile_id:
                if binding.get("inputProfileId") == profile_id:
                    binding["inputProfileId"] = ""
                if binding.get("targetProfileId") == profile_id:
                    binding["targetProfileId"] = ""
                binding["active"] = False
                self.save_binding(slot_index, binding)
        return True

    # ---------------- Slot Bindings (Saved directly in model_dir/<slotIndex>/params.json) ----------------

    def get_all_bindings(self) -> Dict[str, Dict[str, Any]]:
        bindings = {}
        slot_mgr = self._get_slot_manager()
        if not slot_mgr:
            return bindings
        
        try:
            slot_infos = slot_mgr.getAllSlotInfo()
            for slot_info in slot_infos:
                slot_index = getattr(slot_info, "slotIndex", -1)
                if slot_index == -1:
                    continue
                input_id = getattr(slot_info, "formantInputProfileId", "")
                target_id = getattr(slot_info, "formantTargetProfileId", "")
                active = getattr(slot_info, "formantProfileActive", False)
                strength = getattr(slot_info, "formantProfileStrength", 0.35)

                if input_id or target_id or active:
                    bindings[str(slot_index)] = {
                        "slotIndex": slot_index,
                        "inputProfileId": input_id,
                        "targetProfileId": target_id,
                        "active": active,
                        "strength": strength
                    }
        except Exception as e:
            logger.error(f"Error getting slot bindings from model params.json: {e}")
        return bindings

    def get_binding(self, slot_index: Any) -> Optional[Dict[str, Any]]:
        slot_mgr = self._get_slot_manager()
        if not slot_mgr:
            return None
        try:
            slot_info = slot_mgr.get_slot_info(int(slot_index))
            if slot_info:
                return {
                    "slotIndex": slot_index,
                    "inputProfileId": getattr(slot_info, "formantInputProfileId", ""),
                    "targetProfileId": getattr(slot_info, "formantTargetProfileId", ""),
                    "active": getattr(slot_info, "formantProfileActive", False),
                    "strength": getattr(slot_info, "formantProfileStrength", 0.35)
                }
        except Exception as e:
            logger.error(f"Error reading binding for slot {slot_index}: {e}")
        return None

    def save_binding(self, slot_index: Any, binding_data: Dict[str, Any]) -> Dict[str, Any]:
        slot_mgr = self._get_slot_manager()
        if not slot_mgr:
            return binding_data
        
        try:
            idx = int(slot_index)
            slot_info = slot_mgr.get_slot_info(idx)
            if slot_info:
                if "inputProfileId" in binding_data:
                    slot_info.formantInputProfileId = binding_data["inputProfileId"]
                if "targetProfileId" in binding_data:
                    slot_info.formantTargetProfileId = binding_data["targetProfileId"]
                if "active" in binding_data:
                    slot_info.formantProfileActive = bool(binding_data["active"])
                if "strength" in binding_data:
                    slot_info.formantProfileStrength = float(binding_data["strength"])

                slot_mgr.save_model_slot(idx, slot_info)
                logger.info(f"Saved formant binding directly into model params.json for slot {idx}")
        except Exception as e:
            logger.error(f"Error saving binding for slot {slot_index} into params.json: {e}")
        return binding_data

    def delete_binding(self, slot_index: Any) -> bool:
        slot_mgr = self._get_slot_manager()
        if not slot_mgr:
            return False
        try:
            idx = int(slot_index)
            slot_info = slot_mgr.get_slot_info(idx)
            if slot_info:
                slot_info.formantInputProfileId = ""
                slot_info.formantTargetProfileId = ""
                slot_info.formantProfileActive = False
                slot_mgr.save_model_slot(idx, slot_info)
                logger.info(f"Reset formant binding in params.json for slot {idx}")
                return True
        except Exception as e:
            logger.error(f"Error deleting binding for slot {slot_index}: {e}")
            return False
        return True

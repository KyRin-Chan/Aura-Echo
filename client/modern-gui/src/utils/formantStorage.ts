export interface FormantProfile {
  id: string;
  name: string;
  type: 'input' | 'target';
  envelope: number[];
  sr: number;
  f0?: number;
  centroid?: number;
}

export interface ModelFormantBinding {
  slotIndex: number | string;
  inputProfileId?: string;
  targetProfileId?: string;
  active?: boolean;
  strength?: number;
  recommendedPitch?: number;
  recommendedFormantShift?: number;
}

const PROFILE_LIST_KEY = 'formant_profile_list';
const PROFILE_KEY_PREFIX = 'formant_profile_';
const BINDING_LIST_KEY = 'formant_binding_slots';
const BINDING_KEY_PREFIX = 'formant_binding_slot_';

// ---------------- Profile Storage ----------------

export const getFormantProfileList = (): string[] => {
  try {
    const raw = localStorage.getItem(PROFILE_LIST_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (e) {
    console.error('Error reading formant profile list:', e);
    return [];
  }
};

export const getFormantProfile = (id: string): FormantProfile | null => {
  try {
    const raw = localStorage.getItem(`${PROFILE_KEY_PREFIX}${id}`);
    return raw ? JSON.parse(raw) : null;
  } catch (e) {
    console.error(`Error reading formant profile ${id}:`, e);
    return null;
  }
};

export const getAllFormantProfiles = (): FormantProfile[] => {
  const ids = getFormantProfileList();
  const profiles: FormantProfile[] = [];
  for (const id of ids) {
    const profile = getFormantProfile(id);
    if (profile) profiles.push(profile);
  }
  return profiles;
};

export const syncProfilesFromServer = async (): Promise<FormantProfile[]> => {
  try {
    const res = await fetch('/formant_profiles');
    if (res.ok) {
      const serverProfiles: FormantProfile[] = await res.json();
      if (Array.isArray(serverProfiles)) {
        const ids: string[] = [];
        for (const p of serverProfiles) {
          if (p && p.id) {
            localStorage.setItem(`${PROFILE_KEY_PREFIX}${p.id}`, JSON.stringify(p));
            ids.push(p.id);
          }
        }
        localStorage.setItem(PROFILE_LIST_KEY, JSON.stringify(ids));
        return serverProfiles;
      }
    }
  } catch (e) {
    console.warn('Failed to sync formant profiles from server:', e);
  }
  return getAllFormantProfiles();
};

export const saveFormantProfile = (profile: FormantProfile): void => {
  try {
    localStorage.setItem(`${PROFILE_KEY_PREFIX}${profile.id}`, JSON.stringify(profile));
    const ids = getFormantProfileList();
    if (!ids.includes(profile.id)) {
      ids.push(profile.id);
      localStorage.setItem(PROFILE_LIST_KEY, JSON.stringify(ids));
    }
    // Sync to server asynchronously
    fetch('/formant_profiles', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(profile)
    }).catch((e) => console.warn('Failed to save profile to server:', e));
  } catch (e) {
    console.error('Error saving formant profile:', e);
  }
};

export const deleteFormantProfile = (id: string): void => {
  try {
    localStorage.removeItem(`${PROFILE_KEY_PREFIX}${id}`);
    const ids = getFormantProfileList().filter((item) => item !== id);
    localStorage.setItem(PROFILE_LIST_KEY, JSON.stringify(ids));

    // Sync to server asynchronously
    fetch(`/formant_profiles/${encodeURIComponent(id)}`, {
      method: 'DELETE'
    }).catch((e) => console.warn('Failed to delete profile from server:', e));
  } catch (e) {
    console.error(`Error deleting formant profile ${id}:`, e);
  }
};

// ---------------- Binding Storage ----------------

export const getFormantBindingList = (): (number | string)[] => {
  try {
    const raw = localStorage.getItem(BINDING_LIST_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (e) {
    console.error('Error reading formant binding list:', e);
    return [];
  }
};

export const getFormantBinding = (slotIndex: number | string): ModelFormantBinding | null => {
  try {
    const raw = localStorage.getItem(`${BINDING_KEY_PREFIX}${slotIndex}`);
    return raw ? JSON.parse(raw) : null;
  } catch (e) {
    console.error(`Error reading formant binding for slot ${slotIndex}:`, e);
    return null;
  }
};

export const getAllFormantBindings = (): Record<string | number, ModelFormantBinding> => {
  const slots = getFormantBindingList();
  const bindings: Record<string | number, ModelFormantBinding> = {};
  for (const slotIndex of slots) {
    const binding = getFormantBinding(slotIndex);
    if (binding) bindings[slotIndex] = binding;
  }
  return bindings;
};

export const syncBindingsFromServer = async (): Promise<Record<string | number, ModelFormantBinding>> => {
  try {
    const res = await fetch('/formant_bindings');
    if (res.ok) {
      const serverBindings: Record<string, ModelFormantBinding> = await res.json();
      if (serverBindings && typeof serverBindings === 'object') {
        const slots: (string | number)[] = [];
        for (const [slotKey, b] of Object.entries(serverBindings)) {
          if (b && b.slotIndex !== undefined) {
            localStorage.setItem(`${BINDING_KEY_PREFIX}${b.slotIndex}`, JSON.stringify(b));
            slots.push(b.slotIndex);
          }
        }
        localStorage.setItem(BINDING_LIST_KEY, JSON.stringify(slots));
        return serverBindings;
      }
    }
  } catch (e) {
    console.warn('Failed to sync formant bindings from server:', e);
  }
  return getAllFormantBindings();
};

export const saveFormantBinding = (
  slotIndex: number | string,
  partialBinding: Partial<ModelFormantBinding>
): void => {
  try {
    const existing = getFormantBinding(slotIndex) || { slotIndex };
    const updated: ModelFormantBinding = {
      ...existing,
      ...partialBinding,
      slotIndex
    };
    localStorage.setItem(`${BINDING_KEY_PREFIX}${slotIndex}`, JSON.stringify(updated));

    const slots = getFormantBindingList();
    if (!slots.includes(slotIndex)) {
      slots.push(slotIndex);
      localStorage.setItem(BINDING_LIST_KEY, JSON.stringify(slots));
    }

    // Sync to server asynchronously
    fetch('/formant_bindings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updated)
    }).catch((e) => console.warn('Failed to save binding to server:', e));
  } catch (e) {
    console.error(`Error saving formant binding for slot ${slotIndex}:`, e);
  }
};

export const deleteFormantBinding = (slotIndex: number | string): void => {
  try {
    localStorage.removeItem(`${BINDING_KEY_PREFIX}${slotIndex}`);
    const slots = getFormantBindingList().filter((s) => s !== slotIndex);
    localStorage.setItem(BINDING_LIST_KEY, JSON.stringify(slots));

    // Sync to server asynchronously
    fetch(`/formant_bindings/${encodeURIComponent(String(slotIndex))}`, {
      method: 'DELETE'
    }).catch((e) => console.warn('Failed to delete binding from server:', e));
  } catch (e) {
    console.error(`Error deleting formant binding for slot ${slotIndex}:`, e);
  }
};

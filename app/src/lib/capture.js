// Take a photo with the camera or pick one from the library. Falls back to a file picker in a browser.
import { Capacitor } from '@capacitor/core';
import { Camera, CameraResultType, CameraSource } from '@capacitor/camera';

function pickFile(useCamera) {
  return new Promise((resolve, reject) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    if (useCamera) input.capture = 'environment';
    input.onchange = () => {
      const file = input.files?.[0];
      if (!file) return resolve(null);
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error('Could not read that file.'));
      reader.readAsDataURL(file);
    };
    input.click();
  });
}

export async function takePhoto(source) {
  if (!Capacitor.isNativePlatform()) return pickFile(source === 'camera');
  try {
    const photo = await Camera.getPhoto({
      quality: 90,
      width: 1024,
      correctOrientation: true,
      resultType: CameraResultType.DataUrl,
      source: source === 'camera' ? CameraSource.Camera : CameraSource.Photos
    });
    return photo.dataUrl ?? null;
  } catch (err) {
    // The user closing the picker is not an error.
    if (/cancel/i.test(String(err?.message ?? err))) return null;
    throw err;
  }
}

const KEY = { camera: 'camera', photos: 'photos' };
const OK = new Set(['granted', 'limited']);

// 'granted' | 'denied' | 'prompt'
export async function getPermission(source) {
  if (!Capacitor.isNativePlatform()) return 'granted';
  const state = (await Camera.checkPermissions())[KEY[source]];
  if (OK.has(state)) return 'granted';
  return state === 'denied' ? 'denied' : 'prompt';
}

// Shows the system permission dialog. Returns true when access was granted.
export async function askPermission(source) {
  if (!Capacitor.isNativePlatform()) return true;
  const result = await Camera.requestPermissions({ permissions: [KEY[source]] });
  return OK.has(result[KEY[source]]);
}

export function friendlyError(err, source) {
  const msg = String(err?.message ?? err ?? '');
  if (/simulator|not available|no camera|unavailable/i.test(msg)) {
    return 'No camera is available on this device (simulators do not have one). Try "Choose from photos" instead.';
  }
  if (/denied|permission/i.test(msg)) return deniedMessage(source);
  return msg || 'Something went wrong. Please try again.';
}

export function deniedMessage(source) {
  return source === 'camera'
    ? 'Camera access is turned off. Open Settings, tap LeafLens and switch on Camera.'
    : 'Photo access is turned off. Open Settings, tap LeafLens and allow Photos.';
}

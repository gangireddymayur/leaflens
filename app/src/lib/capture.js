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

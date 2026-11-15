// Turns the current scan into a PDF and lets the user save or share it. Nothing leaves the device
// unless the user picks a destination in the share sheet.
import { Capacitor } from '@capacitor/core';
import { Filesystem, Directory } from '@capacitor/filesystem';
import { Share } from '@capacitor/share';
import { buildReportPdf } from './reportPdf.js';

// Re-encode any image (data URL or file URL) as a modest JPEG so the PDF stays small.
async function photoToJpeg(src, max = 900) {
  const img = await new Promise((resolve, reject) => {
    const el = new Image();
    el.onload = () => resolve(el);
    el.onerror = () => reject(new Error('Could not read the photo for the report.'));
    el.src = src;
  });
  const scale = Math.min(1, max / Math.max(img.naturalWidth, img.naturalHeight));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(img.naturalWidth * scale);
  canvas.height = Math.round(img.naturalHeight * scale);
  canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
  return { dataUrl: canvas.toDataURL('image/jpeg', 0.85), width: canvas.width, height: canvas.height };
}

export async function downloadReport({ image, result }) {
  const now = new Date();
  const photo = await photoToJpeg(image);
  const doc = await buildReportPdf({ photo, result, now });
  const stamp = now.toISOString().slice(0, 16).replace(/[-:T]/g, '');
  const filename = `LeafLens-report-${stamp}.pdf`;

  if (!Capacitor.isNativePlatform()) {
    doc.save(filename);
    return 'saved';
  }
  const data = doc.output('datauristring').split(',')[1];
  const file = await Filesystem.writeFile({ path: filename, data, directory: Directory.Cache });
  try {
    await Share.share({ title: 'LeafLens report', dialogTitle: 'Save or share your report', url: file.uri });
  } catch (err) {
    if (!/cancel/i.test(String(err?.message ?? err))) throw err;
  }
  return 'shared';
}

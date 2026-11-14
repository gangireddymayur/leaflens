// On-device inference: the ONNX model runs inside the app, so no photo ever leaves the phone.
import * as ort from 'onnxruntime-web/wasm';
import ortWasmUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url';
import ortModuleUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url';
import { describeClass } from './labels.js';

const BASE = import.meta.env.BASE_URL;
const SIZE = 224;
const RESIZE = 256;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

// WebView on iOS has no SharedArrayBuffer, so run single-threaded.
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = { wasm: ortWasmUrl, mjs: ortModuleUrl };

let ready;

export function loadModel() {
  ready ??= (async () => {
    const [labels, session] = await Promise.all([
      fetch(`${BASE}model/labels.json`).then((r) => r.json()),
      ort.InferenceSession.create(`${BASE}model/plant_model.onnx`, { executionProviders: ['wasm'] })
    ]);
    // Warm-up run so the first real scan is not slower than the rest.
    const warm = new ort.Tensor('float32', new Float32Array(3 * SIZE * SIZE), [1, 3, SIZE, SIZE]);
    await session.run({ input: warm });
    return { labels, session };
  })();
  return ready;
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Could not read that image.'));
    img.src = src;
  });
}

// Same steps as training: resize shorter side to 256, centre-crop 224, normalise with ImageNet stats.
export function preprocess(img) {
  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  const scale = RESIZE / Math.min(w, h);
  const rw = Math.max(RESIZE, Math.round(w * scale));
  const rh = Math.max(RESIZE, Math.round(h * scale));

  const resized = document.createElement('canvas');
  resized.width = rw;
  resized.height = rh;
  const rctx = resized.getContext('2d', { willReadFrequently: true });
  rctx.imageSmoothingEnabled = true;
  rctx.imageSmoothingQuality = 'high';
  rctx.drawImage(img, 0, 0, rw, rh);

  const left = Math.round((rw - SIZE) / 2);
  const top = Math.round((rh - SIZE) / 2);
  const { data } = rctx.getImageData(left, top, SIZE, SIZE);

  const out = new Float32Array(3 * SIZE * SIZE);
  const plane = SIZE * SIZE;
  for (let i = 0; i < plane; i++) {
    out[i] = (data[i * 4] / 255 - MEAN[0]) / STD[0];
    out[plane + i] = (data[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
    out[2 * plane + i] = (data[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
  }
  return out;
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

export async function classify(src) {
  const { labels, session } = await loadModel();
  const img = await loadImage(src);
  const input = new ort.Tensor('float32', preprocess(img), [1, 3, SIZE, SIZE]);

  const t0 = performance.now();
  const { logits } = await session.run({ input });
  const ms = performance.now() - t0;

  const probs = softmax(logits.data);
  const top = probs
    .map((p, i) => ({ p, i }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 3)
    .map(({ p, i }) => ({ ...describeClass(labels[i]), probability: p }));

  return { top, ms, rawProbs: probs };
}

if (import.meta.env.DEV) window.__classify = classify;

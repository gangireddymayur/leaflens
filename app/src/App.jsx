import React, { useCallback, useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Camera, Image as ImageIcon, Leaf, ShieldCheck, WifiOff, Cpu, RotateCcw, AlertTriangle, CheckCircle2, Sparkles, Bug, Share2, Eye, Wrench, ChevronLeft } from 'lucide-react';
import { classify, loadModel } from './lib/classifier.js';
import { Capacitor } from '@capacitor/core';
import { App as CapApp } from '@capacitor/app';
import { takePhoto } from './lib/capture.js';
import { HEALTHY_TIPS, knowledgeFor } from './lib/knowledge.js';

const BASE = import.meta.env.BASE_URL;
const SAMPLES = [
  { file: 'tomato-early-blight.jpg', label: 'Tomato' },
  { file: 'apple-apple-scab.jpg', label: 'Apple' },
  { file: 'grape-black-rot.jpg', label: 'Grape' },
  { file: 'potato-late-blight.jpg', label: 'Potato' },
  { file: 'corn-maize-common-rust.jpg', label: 'Corn' },
  { file: 'strawberry-leaf-scorch.jpg', label: 'Strawberry' },
  { file: 'peach-bacterial-spot.jpg', label: 'Peach' },
  { file: 'tomato-healthy.jpg', label: 'Tomato' }
];

const spring = { type: 'spring', stiffness: 260, damping: 28 };
const screenMotion = {
  initial: { opacity: 0, y: 24, scale: 0.985 },
  animate: { opacity: 1, y: 0, scale: 1 },
  exit: { opacity: 0, y: -16, scale: 0.99 },
  transition: spring
};

function useModel() {
  const [state, setState] = useState('loading');
  useEffect(() => {
    let alive = true;
    loadModel()
      .then(() => alive && setState('ready'))
      .catch(() => alive && setState('error'));
    return () => { alive = false; };
  }, []);
  return state;
}


function TopBar({ title, onBack }) {
  return (
    <header className="topbar">
      <button className="icon-btn" aria-label="Back" onClick={onBack}><ChevronLeft size={24} strokeWidth={2.4} /></button>
      <span className="topbar-title">{title}</span>
      <span className="topbar-spacer" />
    </header>
  );
}

function Home({ onPick, onSample, modelState }) {
  const busy = modelState !== 'ready';
  return (
    <motion.main className="screen home" {...screenMotion}>
      <div className="brand">
        <span className="brand-mark"><Leaf size={20} strokeWidth={2.4} /></span>
        <span className="brand-name">LeafLens</span>
      </div>

      <div className="hero">
        <h1>Know your plant's health <em>in a snap.</em></h1>
        <p className="lead">Point your camera at a leaf. A model trained on 54,000 leaf photos checks it right on your phone.</p>
      </div>

      <div className="chips">
        <span className="chip"><Cpu size={14} /> On-device AI</span>
        <span className="chip"><WifiOff size={14} /> Works offline</span>
        <span className="chip"><ShieldCheck size={14} /> Photos never leave your phone</span>
      </div>

      <div className="actions">
        <button className="btn primary" disabled={busy} onClick={() => onPick('camera')}>
          <Camera size={20} /> {modelState === 'loading' ? 'Warming up the model…' : 'Scan a leaf'}
        </button>
        <button className="btn ghost" disabled={busy} onClick={() => onPick('photos')}>
          <ImageIcon size={20} /> Choose from photos
        </button>
        {modelState === 'error' && <p className="error-note">The model could not load. Please restart the app.</p>}
      </div>

      <section className="samples">
        <div className="section-title"><Sparkles size={15} /> No leaf nearby? Try a sample</div>
        <div className="sample-row">
          {SAMPLES.map((s) => (
            <button key={s.file} className="sample" disabled={busy} onClick={() => onSample(`${BASE}samples/${s.file}`)}>
              <img src={`${BASE}samples/${s.file}`} alt={`${s.label} leaf sample`} loading="lazy" />
              <span>{s.label}</span>
            </button>
          ))}
        </div>
      </section>

      <p className="footnote">Model: MobileNetV3-Large · 38 conditions · 14 crops</p>
    </motion.main>
  );
}

function Analyzing({ image, onBack }) {
  return (
    <motion.main className="screen analyzing" {...screenMotion}>
      <TopBar title="Analyzing" onBack={onBack} />
      <div className="photo scanning">
        <img src={image} alt="Your leaf" />
        <div className="scan-line" />
      </div>
      <p className="analyzing-text">Analyzing on your device…</p>
    </motion.main>
  );
}

function Ring({ value, healthy }) {
  const r = 34;
  const c = 2 * Math.PI * r;
  return (
    <svg className={`ring ${healthy ? 'ok' : 'warn'}`} viewBox="0 0 80 80" aria-label={`${Math.round(value * 100)} percent confidence`}>
      <circle cx="40" cy="40" r={r} className="ring-bg" />
      <motion.circle
        cx="40" cy="40" r={r} className="ring-fg"
        strokeDasharray={c}
        initial={{ strokeDashoffset: c }}
        animate={{ strokeDashoffset: c * (1 - value) }}
        transition={{ duration: 0.9, ease: 'easeOut' }}
      />
      <text x="40" y="45" textAnchor="middle" className="ring-text">{Math.round(value * 100)}%</text>
    </svg>
  );
}


function About({ best }) {
  const info = knowledgeFor(best.raw);
  if (best.healthy) {
    return (
      <motion.section className="card about" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.24 }}>
        <div className="section-title"><Leaf size={15} /> Keeping it healthy</div>
        <ul className="tips">{HEALTHY_TIPS.map((t) => <li key={t}>{t}</li>)}</ul>
      </motion.section>
    );
  }
  if (!info) return null;
  return (
    <motion.section className="card about" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.24 }}>
      <div className="section-title"><Bug size={15} /> About this condition</div>
      <div className="tags">
        <span className="tag">{info.kind}</span>
        <span className={`tag risk-${info.risk}`}>{info.spreadLabel}</span>
      </div>
      <div className="about-block"><h3><Bug size={14} /> What causes it</h3><p>{info.cause}</p></div>
      <div className="about-block"><h3><Share2 size={14} /> How it spreads</h3><p>{info.spread}</p></div>
      <div className="about-block"><h3><Eye size={14} /> What to look for</h3><p>{info.signs}</p></div>
      <div className="about-block"><h3><Wrench size={14} /> What you can do</h3><ul className="tips">{info.manage.map((m) => <li key={m}>{m}</li>)}</ul></div>
      <p className="expert">General guidance only. Confirm with your local agricultural officer before treating, because the right product and timing depend on your crop, region and season.</p>
    </motion.section>
  );
}

function Result({ image, result, onReset }) {
  const [best, ...others] = result.top;
  const unsure = best.probability < 0.6;
  const healthy = best.healthy && !unsure;
  return (
    <motion.main className="screen result" {...screenMotion}>
      <TopBar title="Scan result" onBack={onReset} />
      <div className="photo small"><img src={image} alt="Your leaf" /></div>

      <motion.section className="card main-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.08 }}>
        {unsure ? (
          <div className="status warn"><AlertTriangle size={16} /> Not sure about this one</div>
        ) : healthy ? (
          <div className="status ok"><CheckCircle2 size={16} /> Looks healthy</div>
        ) : (
          <div className="status warn"><AlertTriangle size={16} /> Possible issue detected</div>
        )}
        <div className="main-row">
          <div>
            <div className="crop">{best.crop}</div>
            <h2>{best.condition}</h2>
          </div>
          <Ring value={best.probability} healthy={healthy} />
        </div>
        {unsure && (
          <p className="hint">The model is not confident. Try a clearer, close-up photo of a single leaf in good light.</p>
        )}
      </motion.section>

      <motion.section className="card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.16 }}>
        <div className="section-title">Other possibilities</div>
        {others.map((o, i) => (
          <div className="bar-row" key={o.raw}>
            <div className="bar-label"><span>{o.crop} · {o.condition}</span><b>{(o.probability * 100).toFixed(1)}%</b></div>
            <div className="bar"><motion.i initial={{ width: 0 }} animate={{ width: `${Math.max(2, o.probability * 100)}%` }} transition={{ duration: 0.8, delay: 0.3 + i * 0.1 }} /></div>
          </div>
        ))}
      </motion.section>

      {!unsure && <About best={best} />}

      <div className="meta">
        <span><Cpu size={13} /> {Math.round(result.ms)} ms on this device</span>
        <span><WifiOff size={13} /> No internet used</span>
      </div>
      <p className="disclaimer">Learning project, not agronomic advice. The model only knows 38 leaf conditions across 14 crops and was trained on lab photos, so real-field accuracy will be lower. Photos of anything other than a leaf are unreliable.</p>

      <button className="btn primary" onClick={onReset}><RotateCcw size={18} /> Scan another leaf</button>
    </motion.main>
  );
}

export default function App() {
  const modelState = useModel();
  const [screen, setScreen] = useState('home');
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const runId = useRef(0);

  const run = useCallback(async (src) => {
    if (!src) return;
    const id = ++runId.current;
    setImage(src);
    setError(null);
    setScreen('analyzing');
    try {
      const started = performance.now();
      const res = await classify(src);
      // Keep the scanning animation visible long enough to feel deliberate.
      const wait = Math.max(0, 900 - (performance.now() - started));
      await new Promise((r) => setTimeout(r, wait));
      if (id !== runId.current) return;
      setResult(res);
      setScreen('result');
    } catch (e) {
      if (id !== runId.current) return;
      setError(e.message ?? 'Something went wrong.');
      setScreen('home');
    }
  }, []);

  // Leaving a scan: cancel any analysis still running and return to the home screen.
  const goHome = useCallback(() => {
    runId.current += 1;
    setScreen('home');
  }, []);

  // Android's system back button should go back inside the app, not close it.
  const screenRef = useRef(screen);
  screenRef.current = screen;
  useEffect(() => {
    if (!Capacitor.isNativePlatform()) return undefined;
    const handle = CapApp.addListener('backButton', () => {
      if (screenRef.current === 'home') CapApp.exitApp();
      else goHome();
    });
    return () => { handle.then((h) => h.remove()); };
  }, [goHome]);

  const pick = useCallback(async (source) => {
    try {
      run(await takePhoto(source));
    } catch (e) {
      setError(e.message ?? 'Could not open the camera.');
    }
  }, [run]);

  return (
    <div className="app">
      <div className="bg-glow g1" />
      <div className="bg-glow g2" />
      <AnimatePresence mode="wait">
        {screen === 'home' && <Home key="home" modelState={modelState} onPick={pick} onSample={run} />}
        {screen === 'analyzing' && <Analyzing key="analyzing" image={image} onBack={goHome} />}
        {screen === 'result' && result && <Result key="result" image={image} result={result} onReset={goHome} />}
      </AnimatePresence>
      <AnimatePresence>
        {error && screen === 'home' && (
          <motion.div className="toast" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} onClick={() => setError(null)}>
            {error}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

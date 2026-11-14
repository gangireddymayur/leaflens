import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import './styles.css';
import { Capacitor } from '@capacitor/core';
import { StatusBar, Style } from '@capacitor/status-bar';

// Light status-bar text on our dark background (otherwise the clock is invisible on Android).
if (Capacitor.getPlatform() === 'android') document.documentElement.classList.add('android');
if (Capacitor.isNativePlatform()) {
  // Draw behind the status bar (edge to edge) and keep its text light.
  StatusBar.setOverlaysWebView({ overlay: true }).catch(() => {});
  StatusBar.setStyle({ style: Style.Dark }).catch(() => {});
}

createRoot(document.getElementById('root')).render(<App />);

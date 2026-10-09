import { useEffect, useRef, useState } from 'react';
import { fluidSimulation } from '../lib/fluidSimulation.js';

// Decide once whether the heavy WebGL fluid should run, or the light gradient fallback.
function shouldUseFluid() {
  if (typeof window === 'undefined') return false;
  const reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduced) return false;
  const isMobile = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
  if (isMobile) return false;                       // auto-fallback on phones/tablets
  const cores = navigator.hardwareConcurrency || 4;
  const mem = navigator.deviceMemory || 4;
  if (cores <= 2 || mem <= 2) return false;         // low-power desktop
  // WebGL available?
  try {
    const c = document.createElement('canvas');
    const gl = c.getContext('webgl2') || c.getContext('webgl') || c.getContext('experimental-webgl');
    if (!gl) return false;
  } catch { return false; }
  return true;
}

export default function FluidBackground() {
  const canvasRef = useRef(null);
  const [useFluid] = useState(shouldUseFluid);

  useEffect(() => {
    if (!useFluid || !canvasRef.current) return;
    let destroy;
    try { destroy = fluidSimulation(canvasRef.current); }
    catch (e) { /* if WebGL init fails, the gradient fallback stays visible underneath */ }
    return () => { if (destroy) destroy(); };
  }, [useFluid]);

  return (
    <div className="dx-bg" aria-hidden="true">
      {/* gradient fallback — always rendered; the canvas (when used) paints over it */}
      <div className="dx-bg-fallback" />
      {useFluid && <canvas ref={canvasRef} className="dx-bg-canvas" />}
    </div>
  );
}

// Fan-out to the three local Python model services + draw the annotated overlay.
import axios from 'axios';
import FormData from 'form-data';
import { createCanvas, loadImage } from '@napi-rs/canvas';

const MODELS = {
  occlusal: process.env.MODEL_OCCLUSAL || 'http://127.0.0.1:8001/analyze',
  tooth: process.env.MODEL_TOOTH || 'http://127.0.0.1:8002/analyze',
  gingivitis: process.env.MODEL_GINGIVITIS || 'http://127.0.0.1:8003/analyze',
};
const MODEL_KEYS = {
  occlusal: process.env.MODEL_OCCLUSAL_KEY || process.env.MODEL_API_KEY || '',
  tooth: process.env.MODEL_TOOTH_KEY || process.env.MODEL_API_KEY || '',
  gingivitis: process.env.MODEL_GINGIVITIS_KEY || process.env.MODEL_API_KEY || '',
};
const HEALTH = Object.fromEntries(Object.entries(MODELS).map(([k, v]) => [k, v.replace(/\/analyze$/, '/health')]));
const TIMEOUT = 90000;

async function callOne(name, url, buffer, filename) {
  try {
    const fd = new FormData();
    fd.append('file', buffer, { filename: filename || 'photo.jpg', contentType: 'image/jpeg' });
    const headers = fd.getHeaders();
    if (MODEL_KEYS[name]) headers.Authorization = `Bearer ${MODEL_KEYS[name]}`;
    const r = await axios.post(url, fd, { headers, timeout: TIMEOUT, maxBodyLength: Infinity });
    return [name, r.data];
  } catch (e) {
    return [name, { _unavailable: true, _error: String(e.message || e).slice(0, 200) }];
  }
}

export async function runModels(buffer, filename) {
  const entries = await Promise.all(Object.entries(MODELS).map(([n, u]) => callOne(n, u, buffer, filename)));
  return Object.fromEntries(entries);
}

export async function checkHealth() {
  const out = {};
  await Promise.all(Object.entries(HEALTH).map(async ([n, u]) => {
    try { const r = await axios.get(u, { timeout: 4000 }); out[n] = r.status === 200; }
    catch { out[n] = false; }
  }));
  return out;
}

// --- overlay ---
const COL = { caries: '#D82828', gum: '#FF8C00', tooth: '#2882C8' };

function pxBox(b, W, H) {
  let [x0, y0, x1, y1] = b.map(Number);
  if (Math.max(x0, y0, x1, y1) <= 1.5) { x0 *= W; y0 *= H; x1 *= W; y1 *= H; } // normalised
  return [x0, y0, x1, y1];
}
function labelBox(ctx, x0, y0, text, color) {
  ctx.font = '600 15px sans-serif';
  const w = ctx.measureText(text).width + 8;
  ctx.fillStyle = color;
  ctx.fillRect(x0, Math.max(y0 - 20, 0), w, 20);
  ctx.fillStyle = '#fff';
  ctx.fillText(text, x0 + 4, Math.max(y0 - 6, 14));
}

export async function annotate(buffer, results) {
  const img = await loadImage(buffer);
  const W = img.width, H = img.height;
  const canvas = createCanvas(W, H);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  ctx.lineJoin = 'round';

  const t = results.tooth;
  if (t && !t._unavailable) for (const d of t.teeth || []) {
    const b = d.bbox || d.box; if (!b) continue;
    const [x0, y0, x1, y1] = pxBox(b, W, H);
    ctx.strokeStyle = COL.tooth; ctx.lineWidth = 1.5; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    labelBox(ctx, x0, y0, String(d.type || 'tooth'), COL.tooth);
  }
  const g = results.gingivitis;
  if (g && !g._unavailable) for (const d of g.regions || []) {
    const b = d.bbox || d.box; if (!b) continue;
    const [x0, y0, x1, y1] = pxBox(b, W, H);
    ctx.strokeStyle = COL.gum; ctx.lineWidth = 3; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    labelBox(ctx, x0, y0, `gingivitis ${(d.confidence || 0).toFixed(2)}`, COL.gum);
  }
  const o = results.occlusal;
  if (o && !o._unavailable) for (const d of o.caries || []) {
    const b = d.box || d.bbox; if (!b) continue;
    const [x0, y0, x1, y1] = pxBox(b, W, H);
    ctx.strokeStyle = COL.caries; ctx.lineWidth = 3; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    labelBox(ctx, x0, y0, `caries ${(d.confidence || 0).toFixed(2)}`, COL.caries);
  }
  return canvas.toBuffer('image/jpeg');
}

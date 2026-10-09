import React, { useEffect, useState } from 'react';
import { getHealth } from '../lib/api.js';

const SERVICES = [
  ['caries', 'Caries screening', 'YOLO detector plus MobileNet-V3 severity grader'],
  ['tooth', 'Tooth type detection', 'Four tooth types; no tooth numbering is inferred'],
  ['gingivitis', 'Gingivitis screening', 'Single-class gum-region detector'],
];

export default function HealthView() {
  const [health, setHealth] = useState(null);
  const [error, setError] = useState('');
  async function refresh() {
    try { setHealth(await getHealth()); setError(''); }
    catch (e) { setError(e.message); }
  }
  useEffect(() => { refresh(); const timer = setInterval(refresh, 15000); return () => clearInterval(timer); }, []);

  return <div>
    <div className="mb-5"><h1 className="text-[28px] font-bold tracking-tight mb-1">System health</h1><p className="text-muted text-sm m-0">Service status refreshes every 15 seconds.</p></div>
    <div className="grid sm:grid-cols-2 gap-3 mb-4">
      <article className="card p-5"><div className="text-xs text-muted uppercase tracking-wide">Main backend</div><div className="font-semibold mt-1">{health?.status === 'ok' ? 'Available' : 'Checking…'}</div></article>
      <article className="card p-5"><div className="text-xs text-muted uppercase tracking-wide">Patient database</div><div className="font-semibold mt-1">{health?.database === 'available' ? 'Available' : health ? 'Unavailable' : 'Checking…'}</div></article>
    </div>
    <section className="card divide-y divide-line">
      {SERVICES.map(([key, name, detail]) => {
        const status = health?.models?.[key]?.status;
        const up = status === 'available';
        return <div key={key} className="p-4 flex items-center gap-3">
          <span className={`w-3 h-3 rounded-full shrink-0 ${up ? 'bg-routine' : 'bg-review'}`} />
          <div className="flex-1"><b className="text-sm">{name}</b><p className="text-muted text-xs m-0 mt-0.5">{detail}</p></div>
          <span className={`text-xs font-semibold ${up ? 'text-routine' : 'text-review'}`}>{status || 'Checking…'}</span>
        </div>;
      })}
    </section>
    <p className="text-muted text-sm mt-4">Email service: {health?.email_configured ? 'configured' : 'Email service is not configured.'}</p>
    {error && <p role="alert" className="text-urgent text-sm">{error}</p>}
    <button className="btn-ghost mt-2" onClick={refresh}>Refresh status</button>
  </div>;
}

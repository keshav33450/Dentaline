import React, { useEffect, useMemo, useState } from 'react';
import { getQueue } from '../lib/api.js';

const RISKS = ['all', 'high', 'moderate', 'incomplete', 'low'];
const TONE = { high: 'text-urgent bg-urgent/10', moderate: 'text-review bg-review/10', incomplete: 'text-muted bg-line', low: 'text-routine bg-routine/10' };

export default function QueueView({ onOpen }) {
  const [rows, setRows] = useState([]);
  const [filter, setFilter] = useState('all');
  const [query, setQuery] = useState('');
  const [error, setError] = useState('');

  useEffect(() => { getQueue().then(setRows).catch((e) => setError(e.message)); }, []);

  const counts = useMemo(() => Object.fromEntries(['high', 'moderate', 'incomplete', 'low'].map((key) => [key, rows.filter((r) => r.risk_level === key).length])), [rows]);
  const shown = rows.filter((row) => (filter === 'all' || row.risk_level === filter) && (!query || `${row.patient_name} ${row.patient_id}`.toLowerCase().includes(query.toLowerCase())));

  return <div>
    <div className="flex items-baseline gap-3 mb-2"><div><h1 className="text-[28px] font-bold tracking-tight mb-1">Screening history</h1><p className="text-muted text-sm m-0">Recent results for your patients.</p></div><span className="text-muted text-sm ml-auto">{rows.length} total</span></div>
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 my-4">{Object.entries(counts).map(([key, count]) => <div key={key} className="card p-3"><b className={`text-sm uppercase ${key === 'high' ? 'text-urgent' : key === 'moderate' ? 'text-review' : key === 'low' ? 'text-routine' : 'text-muted'}`}>{key}</b><span className="block font-display text-2xl font-bold">{count}</span></div>)}</div>
    <input className="field mb-3" placeholder="Search patient name or ID" value={query} onChange={(e) => setQuery(e.target.value)} />
    <div className="flex gap-2 mb-3 flex-wrap">{RISKS.map((value) => <button key={value} onClick={() => setFilter(value)} className={`text-xs px-3 py-2 rounded-full border ${filter === value ? 'bg-ink text-white border-ink' : 'bg-surface text-muted border-line'}`}>{value === 'all' ? 'All' : value}</button>)}</div>
    {error && <p role="alert" className="text-urgent text-sm">{error}</p>}
    {!shown.length ? <div className="text-center text-muted py-10 text-sm">No screenings match this search.</div> : shown.map((item) => {
      const gum = item.gingivitis || {};
      const caries = item.caries || {};
      const when = item.created_at ? new Date(item.created_at).toLocaleString() : 'Date unavailable';
      return <button key={item._id || item.id} onClick={() => onOpen(item._id || item.id)} className="w-full text-left card p-4 mb-2.5 flex items-center gap-3 hover:border-teal/40 transition-colors">
        <span className={`text-[10px] font-bold uppercase px-2.5 py-1 rounded-full ${TONE[item.risk_level] || TONE.incomplete}`}>{item.risk_level || 'incomplete'}</span>
        <span className="flex-1 min-w-0"><b className="block text-sm truncate">{item.patient_name || 'Patient'} · {item.patient_id}</b><small className="block text-muted text-xs mt-0.5 truncate">Caries: {caries.status === 'available' ? caries.severity : 'Check unavailable'} · Gingivitis: {gum.status === 'available' ? (gum.found ? 'Detected' : 'Not detected') : 'Check unavailable'}</small></span>
        <span className="text-muted text-[11px] whitespace-nowrap">{when}</span>
      </button>;
    })}
  </div>;
}

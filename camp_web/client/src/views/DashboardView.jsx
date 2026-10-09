import React, { useEffect, useState } from 'react';
import { downloadCsv, getStats } from '../lib/api.js';
import { Bars } from '../components/ui.jsx';

export default function DashboardView() {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => { getStats().then(setStats).catch((e) => setError(e.message)); }, []);

  if (error) return <div className="card p-5 text-urgent text-sm">{error}</div>;
  if (!stats) return <div className="text-muted py-10 text-center text-sm">Loading dashboard…</div>;

  const today = stats.risk_today || {};
  const kpis = [
    { value: stats.total_patients, label: 'registered patients', color: 'text-ink' },
    { value: stats.screened_today, label: 'screened today', color: 'text-ink' },
    { value: today.high || 0, label: 'high risk today', color: 'text-urgent' },
    { value: today.moderate || 0, label: 'moderate risk today', color: 'text-review' },
    { value: stats.reports_generated, label: 'reports generated', color: 'text-ink' },
    { value: stats.reports_sent, label: 'reports sent', color: 'text-routine' },
  ];

  return <div>
    <div className="mb-5"><h1 className="text-[28px] font-bold tracking-tight mb-1">Doctor dashboard</h1><p className="text-muted text-sm m-0">Counts reflect records stored for your account.</p></div>
    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-4">{kpis.map((item) => <div key={item.label} className="card p-4 sm:p-5"><div className={`font-display font-bold text-3xl sm:text-4xl leading-none ${item.color}`}>{item.value}</div><div className="text-muted text-xs sm:text-sm mt-2">{item.label}</div></div>)}</div>
    <div className="grid md:grid-cols-2 gap-3">
      <section className="card p-5"><h2 className="font-semibold mb-2">Risk distribution · all screenings</h2><Bars rows={[
        { l: 'High', v: stats.risk?.high || 0, color: '#C0392B' },
        { l: 'Moderate', v: stats.risk?.moderate || 0, color: '#E08A1E' },
        { l: 'Low', v: stats.risk?.low || 0, color: '#1E9E6A' },
      ]} /></section>
      <section className="card p-5"><h2 className="font-semibold mb-2">Detected indicators · all screenings</h2><Bars rows={[
        { l: 'Caries', v: stats.caries_detected || 0, color: '#D82828' },
        { l: 'Gingivitis', v: stats.gingivitis_detected || 0, color: '#FF8C00' },
      ]} /></section>
    </div>
    <button className="btn-primary mt-4" onClick={() => downloadCsv().catch((e) => setError(e.message))}>Export screening history (CSV)</button>
    <p className="text-muted text-xs mt-3">Risk totals exclude incomplete screenings where no positive result was found and at least one model was unavailable.</p>
  </div>;
}

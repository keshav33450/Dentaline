import React from 'react';

export const PRIO = {
  URGENT: { c: 'text-urgent', bg: 'bg-urgent/10', dot: 'bg-urgent', ring: 'border-l-urgent', ic: '🔴' },
  REVIEW: { c: 'text-review', bg: 'bg-review/10', dot: 'bg-review', ring: 'border-l-review', ic: '🟠' },
  ROUTINE: { c: 'text-routine', bg: 'bg-routine/10', dot: 'bg-routine', ring: 'border-l-routine', ic: '🟢' },
};

export function PriorityPill({ p, size = 'md' }) {
  const s = PRIO[p] || PRIO.ROUTINE;
  const pad = size === 'lg' ? 'text-[17px] px-4 py-2' : 'text-[12px] px-2.5 py-1';
  return <span className={`pill ${s.bg} ${s.c} ${pad}`}>{size === 'lg' && s.ic} {p}</span>;
}

export function Finding({ icon, label, sub, value, bar }) {
  return (
    <div className="flex items-center gap-3 py-3 border-t border-line first:border-t-0">
      <div className="w-8 text-center text-[19px]">{icon}</div>
      <div className="flex-1 min-w-0">
        <b className="text-[15px] font-semibold">{label}</b>
        {bar != null ? (
          <div className="h-1.5 rounded bg-line overflow-hidden mt-1.5 w-32">
            <div className="h-full bg-review" style={{ width: `${bar}%` }} />
          </div>
        ) : sub ? <small className="block text-muted text-[12.5px]">{sub}</small> : null}
      </div>
      <div className="font-display font-bold text-[15px] whitespace-nowrap">{value}</div>
    </div>
  );
}

export function Spinner({ label }) {
  return (
    <div className="text-center py-14 text-muted">
      <div className="spin w-10 h-10 border-[3px] border-line border-t-teal rounded-full mx-auto mb-3.5" />
      {label}
    </div>
  );
}

export function Toast({ msg }) {
  if (!msg) return null;
  return (
    <div className="fixed left-1/2 -translate-x-1/2 bottom-24 z-50 bg-ink text-white px-4 py-3 rounded-xl text-sm max-w-[90vw] text-center shadow-card rise">
      {msg}
    </div>
  );
}

export function Bars({ rows, max }) {
  const m = Math.max(1, max ?? Math.max(...rows.map((r) => r.v), 1));
  return (
    <div className="flex flex-col gap-2.5 mt-1.5">
      {rows.map((r) => (
        <div key={r.l} className="flex items-center gap-2.5 text-[13.5px]">
          <span className="w-20 text-muted">{r.l}</span>
          <span className="flex-1 h-5 rounded-md bg-line overflow-hidden">
            <i className="block h-full rounded-md" style={{ width: `${(r.v / m) * 100}%`, background: r.color || '#0E8C7A' }} />
          </span>
          <span className="w-7 text-right font-display font-bold">{r.v}</span>
        </div>
      ))}
    </div>
  );
}

import React, { useEffect, useState } from 'react';
import { createPatient, getPatientHistory, getPatients } from '../lib/api.js';

const riskTone = { high: 'text-urgent bg-urgent/10', moderate: 'text-review bg-review/10', low: 'text-routine bg-routine/10', incomplete: 'text-muted bg-line' };

export default function PatientsView({ onStartScreen, onOpenScreening, toast }) {
  const [patients, setPatients] = useState([]);
  const [search, setSearch] = useState('');
  const [active, setActive] = useState(null);
  const [history, setHistory] = useState([]);
  const [form, setForm] = useState({ name: '', age: '', phone: '', email: '' });
  const [error, setError] = useState('');

  async function refresh() {
    try { setPatients(await getPatients(search)); }
    catch (e) { setError(e.message); }
  }
  useEffect(() => { refresh(); }, [search]);

  async function openPatient(patient) {
    setActive(patient);
    try { setHistory((await getPatientHistory(patient.patient_id)).screenings); }
    catch (e) { setError(e.message); setHistory([]); }
  }

  async function submit(event) {
    event.preventDefault(); setError('');
    try {
      const created = await createPatient({ ...form, age: Number(form.age) });
      setForm({ name: '', age: '', phone: '', email: '' });
      await refresh();
      await openPatient(created);
      toast(`Patient ${created.patient_id} registered.`);
    } catch (e) { setError(e.message); }
  }

  return (
    <div>
      <div className="mb-5 flex items-end gap-3">
        <div><h1 className="text-[28px] font-bold tracking-tight mb-1">Patients</h1><p className="text-muted text-sm m-0">Register a patient and review their screening history.</p></div>
      </div>
      <div className="grid lg:grid-cols-[minmax(0,1fr)_minmax(300px,0.9fr)] gap-4 items-start">
        <section className="card p-5">
          <h2 className="text-lg font-semibold mb-3">Patient registry</h2>
          <input className="field mb-3" placeholder="Search name, ID, phone, or e-mail" value={search} onChange={(e) => setSearch(e.target.value)} />
          <div className="flex flex-col gap-2 max-h-[520px] overflow-y-auto">
            {!patients.length && <p className="text-muted text-sm py-5 text-center">No matching patients.</p>}
            {patients.map((patient) => (
              <button key={patient.patient_id} onClick={() => openPatient(patient)} className={`text-left p-3 rounded-xl border transition-colors ${active?.patient_id === patient.patient_id ? 'border-teal bg-teal/5' : 'border-line hover:bg-bg/60'}`}>
                <b className="block text-sm">{patient.name}</b>
                <span className="text-muted text-xs">{patient.patient_id} · {patient.age} years</span>
              </button>
            ))}
          </div>
        </section>

        <div className="flex flex-col gap-4">
          <section className="card p-5">
            <h2 className="text-lg font-semibold mb-3">Register patient</h2>
            <form onSubmit={submit} className="flex flex-col gap-3">
              <input className="field" placeholder="Patient name" autoComplete="name" required minLength={2} maxLength={120} value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
              <div className="grid grid-cols-2 gap-3">
                <input className="field" type="number" min="0" max="120" placeholder="Age" required value={form.age} onChange={(e) => setForm({ ...form, age: e.target.value })} />
                <input className="field" placeholder="Phone" autoComplete="tel" required value={form.phone} onChange={(e) => setForm({ ...form, phone: e.target.value })} />
              </div>
              <input className="field" type="email" placeholder="E-mail" autoComplete="email" required value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
              <button className="btn-primary">Register patient</button>
            </form>
          </section>

          {active && <section className="card p-5">
            <div className="flex justify-between items-start gap-3">
              <div><h2 className="text-lg font-semibold m-0">{active.name}</h2><p className="text-muted text-sm mt-1">{active.patient_id} · {active.age} years</p></div>
              <button className="btn-primary !w-auto !px-3 !py-2 text-sm" onClick={() => onStartScreen(active)}>New screening</button>
            </div>
            <p className="text-sm text-muted">{active.phone}<br />{active.email}</p>
            <h3 className="font-semibold text-sm mt-5 mb-2">Screening history</h3>
            {!history.length && <p className="text-muted text-sm">No screenings recorded yet.</p>}
            <div className="flex flex-col gap-2 max-h-[300px] overflow-y-auto">
              {history.map((item) => <button key={item._id || item.id} onClick={() => onOpenScreening(item._id || item.id)} className="text-left border border-line rounded-xl p-3 hover:bg-bg/50">
                <div className="flex items-center justify-between gap-2"><b className="text-sm">{new Date(item.created_at).toLocaleString()}</b><span className={`text-[10px] uppercase tracking-wide px-2 py-1 rounded-full font-bold ${riskTone[item.risk_level] || riskTone.incomplete}`}>{item.risk_level || 'incomplete'}</span></div>
                <p className="text-muted text-xs mb-0 mt-1">Caries: {item.caries?.status === 'available' ? item.caries.severity : 'Check unavailable'} · Gingivitis: {item.gingivitis?.status === 'available' ? (item.gingivitis.found ? 'Yes' : 'Not detected') : 'Check unavailable'} · Report {item.report_generated ? 'generated' : 'not generated'}</p>
              </button>)}
            </div>
          </section>}
        </div>
      </div>
      {error && <p role="alert" className="text-urgent text-sm mt-3">{error}</p>}
    </div>
  );
}

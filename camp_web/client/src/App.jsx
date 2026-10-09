import React, { useEffect, useRef, useState } from 'react';
import { clearSession, getHealth, getProfile, getScreening, getSession, logout, saveSession } from './lib/api.js';
import AuthView from './views/AuthView.jsx';
import ScreenView from './views/ScreenView.jsx';
import PatientsView from './views/PatientsView.jsx';
import QueueView from './views/QueueView.jsx';
import DashboardView from './views/DashboardView.jsx';
import HealthView from './views/HealthView.jsx';
import { Toast } from './components/ui.jsx';

const TABS = [['screen', '📷', 'Screen'], ['patients', '👥', 'Patients'], ['queue', '📋', 'History'], ['camp', '📊', 'Dashboard'], ['health', '⚙️', 'Health']];

export default function App() {
  const [session, setSession] = useState(getSession());
  const [tab, setTab] = useState('screen');
  const [health, setHealth] = useState(null);
  const [toastMsg, setToastMsg] = useState('');
  const [override, setOverride] = useState(null);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [queueKey, setQueueKey] = useState(0);
  const toastT = useRef();

  function toast(message) { setToastMsg(message); clearTimeout(toastT.current); toastT.current = setTimeout(() => setToastMsg(''), 3200); }

  useEffect(() => {
    if (!session?.token) return;
    getProfile().then(({ doctor }) => {
      const next = { ...session, doctor }; saveSession(next); setSession(next);
    }).catch(() => { setSession(null); });
  // Session is checked only when the token changes.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.token]);

  useEffect(() => {
    const expired = () => setSession(null);
    window.addEventListener('dentalx:session-expired', expired);
    return () => window.removeEventListener('dentalx:session-expired', expired);
  }, []);

  useEffect(() => {
    const load = () => getHealth().then(setHealth).catch(() => setHealth(null));
    load(); const i = setInterval(load, 15000); return () => clearInterval(i);
  }, []);

  function signedIn(value) {
    saveSession(value); setSession(value); setTab('screen');
  }

  async function signOut() {
    try { await logout(); } catch { /* the local session is still cleared */ }
    clearSession(); setSession(null); setOverride(null); setSelectedPatient(null);
  }

  async function openScreening(id) {
    try {
      const screening = await getScreening(id);
      setOverride(screening); setSelectedPatient({ patient_id: screening.patient_id, name: screening.patient_name });
      setTab('screen'); window.scrollTo(0, 0);
    } catch (e) { toast(e.message); }
  }

  if (!session?.token) return <AuthView onSignedIn={signedIn} />;

  const models = health?.models || {};
  const activeCount = ['caries', 'tooth', 'gingivitis'].filter((key) => models[key]?.status === 'available').length;

  return <div className="min-h-full pb-[88px]">
    <header className="sticky top-0 z-20 bg-white/85 backdrop-blur-xl border-b border-line/50 shadow-sm">
      <div className="max-w-[980px] mx-auto px-4 sm:px-5 h-[64px] flex items-center justify-between gap-3">
        <div className="flex items-center gap-3 font-display font-bold text-[19px] tracking-tight text-ink">
          <span className="w-9 h-9 rounded-xl bg-gradient-premium shadow-premium grid place-items-center text-white text-[16px]">Dx</span>
          <span>DentalX Camp</span>
        </div>
        <div className="flex items-center gap-3">
          <button onClick={() => setTab('health')} className="flex items-center gap-1.5 px-2 sm:px-3 py-1.5 rounded-full bg-white border border-line shadow-sm" title="Model services">
            {['caries', 'tooth', 'gingivitis'].map((key) => <span key={key} className={`w-2.5 h-2.5 rounded-full ${models[key]?.status === 'available' ? 'bg-routine' : 'bg-review'}`} />)}
            <small className="text-muted font-medium text-[11px] ml-1">{health ? `${activeCount}/3 AI` : 'Checking'}</small>
          </button>
          <div className="hidden sm:block text-right"><b className="block text-xs">{session.doctor?.name || 'Doctor'}</b><small className="text-muted text-[11px]">{session.doctor?.email}</small></div>
          <button onClick={signOut} className="text-xs font-semibold text-muted hover:text-ink">Sign out</button>
        </div>
      </div>
    </header>

    <div className="bg-[#102a26] text-[#cfe6e0] text-[12px] text-center py-1.5 px-4 leading-snug">AI screening aid only — not a diagnosis. Final assessment must be performed by a qualified dental professional.</div>

    <main className="max-w-[980px] mx-auto px-4 sm:px-5 pt-5">
      {tab === 'screen' && <ScreenView key={override?.id || 'fresh'} caseOverride={override} initialPatient={selectedPatient} onClearOverride={() => setOverride(null)} onDone={() => { setQueueKey((k) => k + 1); setOverride(null); setTab('queue'); }} toast={toast} />}
      {tab === 'patients' && <PatientsView onStartScreen={(patient) => { setSelectedPatient(patient); setOverride(null); setTab('screen'); window.scrollTo(0, 0); }} onOpenScreening={openScreening} toast={toast} />}
      {tab === 'queue' && <QueueView key={queueKey} onOpen={openScreening} />}
      {tab === 'camp' && <DashboardView />}
      {tab === 'health' && <HealthView />}
    </main>

    <nav className="fixed bottom-0 inset-x-0 z-30 bg-white/90 backdrop-blur-xl border-t border-line/50 flex pt-2 pb-[max(10px,env(safe-area-inset-bottom))] shadow-[0_-4px_24px_rgba(0,0,0,0.02)]">
      {TABS.map(([id, icon, label]) => <button key={id} onClick={() => { setTab(id); if (id !== 'screen') setOverride(null); window.scrollTo(0, 0); }} className={`flex-1 flex flex-col items-center gap-1 py-1 text-[10px] sm:text-xs font-semibold transition-all ${tab === id ? 'text-teal' : 'text-muted/75 hover:text-muted'}`}>
        <span className={`text-[21px] sm:text-[23px] leading-none ${tab === id ? 'drop-shadow-sm' : ''}`}>{icon}</span>{label}
      </button>)}
    </nav>
    <Toast msg={toastMsg} />
  </div>;
}

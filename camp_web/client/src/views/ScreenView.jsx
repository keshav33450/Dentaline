import React, { useEffect, useRef, useState } from 'react';
import { createPatient, downloadReport, emailCase, getPatients, saveReview, screen } from '../lib/api.js';
import { Spinner } from '../components/ui.jsx';

const MAX_UPLOAD_BYTES = 4 * 1024 * 1024;
const ASSESSMENTS = ['', 'No immediate concern', 'Further examination required', 'Treatment recommended', 'Referral required'];

// Triage verdict presentation — mirrors the backend risk rule exactly.
const VERDICT = {
  high: {
    label: 'URGENT', band: 'See the dentist first',
    rule: 'Moderate/advanced caries, or gingivitis confidence over 60%.',
    ring: 'text-urgent', chip: 'bg-urgent text-white', soft: 'bg-urgent/10 border-urgent/30', bar: 'bg-urgent',
  },
  moderate: {
    label: 'REVIEW', band: 'Dentist review advised',
    rule: 'Mild caries, or any gingivitis detected.',
    ring: 'text-review', chip: 'bg-review text-white', soft: 'bg-review/10 border-review/30', bar: 'bg-review',
  },
  low: {
    label: 'ROUTINE', band: 'Routine check',
    rule: 'No caries or gingivitis above screening threshold.',
    ring: 'text-routine', chip: 'bg-routine text-white', soft: 'bg-routine/10 border-routine/30', bar: 'bg-routine',
  },
  incomplete: {
    label: 'INCOMPLETE', band: 'One or more checks unavailable',
    rule: 'A model did not return a result — re-run or review manually.',
    ring: 'text-muted', chip: 'bg-line text-ink', soft: 'bg-line/60 border-line', bar: 'bg-muted',
  },
};

const SEVERITY_LEVEL = { 'no caries': 0, none: 0, mild: 1, moderate: 2, advanced: 3 };

// Instantly-readable status badges for each finding card.
const BADGE = {
  clear:  { text: 'Clear',   cls: 'bg-routine/15 text-routine' },
  found:  { text: 'Found',   cls: 'bg-review/15 text-review' },
  urgent: { text: 'Urgent',  cls: 'bg-urgent/15 text-urgent' },
  info:   { text: 'Detected', cls: 'bg-teal/15 text-teal' },
};

async function uploadableImage(file) {
  if (file.size <= MAX_UPLOAD_BYTES) return file;
  if (typeof createImageBitmap !== 'function') throw new Error('This photo is too large to upload. Choose a smaller image.');
  const bitmap = await createImageBitmap(file);
  try {
    let scale = Math.min(1, 2560 / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Could not prepare this photo for upload.');
    for (let attempt = 0; attempt < 4; attempt += 1) {
      canvas.width = Math.round(bitmap.width * scale);
      canvas.height = Math.round(bitmap.height * scale);
      context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      for (const quality of [0.84, 0.74, 0.64, 0.54]) {
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', quality));
        if (blob && blob.size <= MAX_UPLOAD_BYTES) return blob;
      }
      scale *= 0.78;
    }
    throw new Error('This photo could not be compressed enough to upload. Choose a smaller image.');
  } finally { bitmap.close(); }
}

export default function ScreenView({ onDone, toast, caseOverride, onClearOverride, initialPatient }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState(0);
  const [result, setResult] = useState(caseOverride || null);
  const [patients, setPatients] = useState([]);
  const [patient, setPatient] = useState(initialPatient || null);
  const [showNewPatient, setShowNewPatient] = useState(false);
  const [patientForm, setPatientForm] = useState({ name: '', age: '', phone: '', email: '' });
  const [notes, setNotes] = useState(caseOverride?.clinical_notes || '');
  const [assessment, setAssessment] = useState(caseOverride?.assessment || '');
  const [reviewSaved, setReviewSaved] = useState(!!caseOverride?.reviewed);
  const [reviewBusy, setReviewBusy] = useState(false);
  const [selectedTooth, setSelectedTooth] = useState(null);
  // which checks the user wants to run on the uploaded photo
  const [checks, setChecks] = useState({ caries: true, tooth: true, gingivitis: true });
  const inputRef = useRef();

  useEffect(() => { if (initialPatient) setPatient(initialPatient); }, [initialPatient]);
  useEffect(() => { getPatients().then(setPatients).catch((e) => toast(e.message)); }, []);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  function chooseFile(f) {
    if (!f) return;
    if (!f.type.startsWith('image/')) { toast('Choose an image file.'); return; }
    setFile(f); setPreview(URL.createObjectURL(f)); setResult(null);
  }
  function pick(e) { chooseFile(e.target.files[0]); }
  function toggleCheck(k) { setChecks((c) => ({ ...c, [k]: !c[k] })); setResult(null); }
  const anyCheck = checks.caries || checks.tooth || checks.gingivitis;

  async function addPatient(event) {
    event.preventDefault();
    try {
      const created = await createPatient({ ...patientForm, age: Number(patientForm.age) });
      setPatient(created); setPatients((items) => [created, ...items]); setShowNewPatient(false);
      setPatientForm({ name: '', age: '', phone: '', email: '' });
      toast(`Patient ${created.patient_id} registered.`);
    } catch (e) { toast(e.message); }
  }

  async function run() {
    if (!file || !patient) return;
    if (!anyCheck) { toast('Select at least one check to run.'); return; }
    setBusy(true); setResult(null); setStage(0);
    const ticker = setInterval(() => setStage((s) => (s < 3 ? s + 1 : s)), 700);
    try {
      const image = await uploadableImage(file);
      const fd = new FormData();
      fd.append('file', image, image === file ? file.name : file.name.replace(/\.[^.]+$/, '') + '.jpg');
      fd.append('patient_id', patient.patient_id);
      // tell the backend which checks to run
      const wanted = ['caries', 'tooth', 'gingivitis'].filter((k) => checks[k]);
      fd.append('models', wanted.join(','));
      const data = await screen(fd);
      setResult(data); setNotes(''); setAssessment(''); setReviewSaved(false); setSelectedTooth(null);
    } catch (e) { toast(e.message); }
    finally { clearInterval(ticker); setBusy(false); }
  }

  async function saveDoctorReview() {
    if (!result?.id) return;
    setReviewBusy(true);
    try { await saveReview(result.id, { clinical_notes: notes, assessment }); setReviewSaved(true); toast('Doctor review saved.'); }
    catch (e) { toast(e.message); }
    finally { setReviewBusy(false); }
  }

  async function sendEmail() {
    try { const response = await emailCase(result.id); toast(response.status || 'Report sent.'); }
    catch (e) { toast(e.message); }
  }

  function reset() {
    setFile(null); setPreview(null); setResult(null); setNotes(''); setAssessment('');
    setSelectedTooth(null);
    setChecks({ caries: true, tooth: true, gingivitis: true });
    if (inputRef.current) inputRef.current.value = '';
    if (caseOverride) onClearOverride?.();
  }

  const r = result;
  const gum = r?.gingivitis;
  const caries = r?.caries;
  const tooth = r?.tooth;
  const v = VERDICT[r?.risk_level] || VERDICT.incomplete;

  const sevText = caries?.status === 'available' ? (caries.severity || 'No caries') : 'Unavailable';
  const sevLevel = SEVERITY_LEVEL[(sevText || '').toLowerCase()] ?? 0;
  const gumConf = gum?.confidence != null ? Math.round(gum.confidence * 100) : null;
  const toothTypes = tooth?.status === 'available' ? Object.entries(tooth.breakdown || {}) : [];

  return (
    <div className="pb-4">
      {!result && !busy && <>
        <div className="mb-5"><h1 className="text-[28px] font-bold tracking-tight mb-1">New Screening</h1><p className="text-muted text-sm m-0">Choose a registered patient, then capture one intraoral photo.</p></div>

        <section className="card p-4 mb-4">
          <div className="flex items-center justify-between gap-3 mb-2"><label htmlFor="patient-select" className="text-sm font-semibold">Patient</label><button className="text-sm text-teal font-semibold" onClick={() => setShowNewPatient(!showNewPatient)}>{showNewPatient ? 'Choose existing' : '+ Register patient'}</button></div>
          {!showNewPatient ? <select id="patient-select" className="field" value={patient?.patient_id || ''} onChange={(e) => setPatient(patients.find((p) => p.patient_id === e.target.value) || null)}>
            <option value="">Select a patient</option>{patients.map((p) => <option key={p.patient_id} value={p.patient_id}>{p.name} · {p.patient_id}</option>)}
          </select> : <form onSubmit={addPatient} className="grid sm:grid-cols-2 gap-2">
            <input className="field" placeholder="Patient name" required minLength={2} value={patientForm.name} onChange={(e) => setPatientForm({ ...patientForm, name: e.target.value })} />
            <input className="field" type="number" min="0" max="120" placeholder="Age" required value={patientForm.age} onChange={(e) => setPatientForm({ ...patientForm, age: e.target.value })} />
            <input className="field" placeholder="Phone" required value={patientForm.phone} onChange={(e) => setPatientForm({ ...patientForm, phone: e.target.value })} />
            <input className="field" type="email" placeholder="E-mail" required value={patientForm.email} onChange={(e) => setPatientForm({ ...patientForm, email: e.target.value })} />
            <button className="btn-primary sm:col-span-2">Register patient</button>
          </form>}
          {patient && <p className="text-xs text-muted mt-2 mb-0">{patient.name} · {patient.age} years · {patient.patient_id}</p>}
        </section>

        <label onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); chooseFile(e.dataTransfer.files[0]); }} className="card flex flex-col items-center justify-center p-9 text-center border-2 border-dashed border-teal/20 cursor-pointer hover:border-teal/50 hover:bg-teal/5 transition-all group">
          <div className="w-16 h-16 rounded-full bg-teal/10 flex items-center justify-center text-[28px] mb-3 group-hover:scale-110 transition-transform">📷</div>
          <h3 className="text-[17px] font-medium mb-1">{file ? 'Photo selected (tap to change)' : 'Upload or capture an intraoral photo'}</h3>
          <p className="text-muted text-[13px] m-0">Drop an image here, choose a file, or use your mobile camera</p>
          <input ref={inputRef} type="file" accept="image/*" capture="environment" className="hidden" onChange={pick} />
        </label>
        {preview && <img src={preview} alt="Selected intraoral photo preview" className="w-full rounded-xl2 mt-4 shadow-card object-contain max-h-[360px] bg-[#0c1a17]" />}

        {/* ---- choose what to check ---- */}
        <div className="mt-4">
          <p className="text-sm font-semibold mb-2">What do you want to check? <span className="text-muted font-normal">· tap to toggle</span></p>
          <div className="grid grid-cols-3 gap-2.5">
            {[
              ['caries', 'Caries', '#D82828', '🦷'],
              ['gingivitis', 'Gingivitis', '#2882C8', '🩸'],
              ['tooth', 'Tooth type', '#D26E1E', '🔎'],
            ].map(([key, label, color, icon]) => {
              const on = checks[key];
              return (
                <button key={key} type="button" onClick={() => toggleCheck(key)}
                  className={`relative rounded-xl2 border-2 p-3 text-center transition-all ${on ? 'bg-white shadow-card' : 'bg-bg opacity-55'}`}
                  style={{ borderColor: on ? color : '#E5E7EB' }}>
                  <div className="text-xl mb-0.5">{icon}</div>
                  <div className="font-semibold text-[13px] leading-tight" style={{ color: on ? color : '#6B7280' }}>{label}</div>
                  <div className="text-[10px] text-muted mt-0.5">{on ? 'will run' : 'off'}</div>
                  {on && <span className="absolute top-1.5 right-1.5 w-4 h-4 rounded-full flex items-center justify-center text-white text-[10px] font-bold" style={{ background: color }}>✓</span>}
                </button>
              );
            })}
          </div>
        </div>

        <button className="btn-primary mt-4 text-[17px]" disabled={!file || !patient || !anyCheck} onClick={run}>
          {anyCheck ? `Run ${[checks.caries && 'caries', checks.gingivitis && 'gingivitis', checks.tooth && 'tooth'].filter(Boolean).length === 3 ? 'all checks' : 'selected checks'}` : 'Select a check'}
        </button>
      </>}

      {busy && <div>
        <Spinner label="Running three AI models on-device…" />
        <div className="grid sm:grid-cols-3 gap-2 text-sm text-center">
          {[['Caries + ICDAS', 'Model 1'], ['Tooth type', 'Model 2'], ['Gingivitis', 'Model 3']].map(([name, m], i) => (
            <div key={name} className={`card p-3 transition-all ${stage > i ? 'border-routine/40' : ''}`}>
              <div className="text-[13px] font-semibold">{name}</div>
              <small className={`block ${stage > i ? 'text-routine' : 'text-muted'}`}>{stage > i ? '✓ done' : stage === i ? 'analysing…' : m}</small>
            </div>
          ))}
        </div>
      </div>}

      {r && <div className="rise">
        {/* ---- header ---- */}
        <div className="flex items-start justify-between gap-3 mb-4">
          <div>
            <h1 className="text-2xl font-bold mb-0.5">Screening report</h1>
            <p className="text-muted text-sm m-0">{r.patient_name} · {r.patient_id}{r.created_at ? ` · ${new Date(r.created_at).toLocaleString()}` : ''}</p>
          </div>
        </div>

        {/* ---- TRIAGE VERDICT banner ---- */}
        <section className={`rounded-xl2 border ${v.soft} p-5 mb-4`}>
          <div className="flex items-center gap-4">
            <div className={`flex-shrink-0 w-16 h-16 rounded-full border-4 ${v.ring} border-current flex items-center justify-center`}>
              <span className={`font-display font-extrabold text-lg ${v.ring}`}>{v.label[0]}</span>
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className={`font-display font-bold text-xs tracking-wide px-2.5 py-1 rounded-full ${v.chip}`}>{v.label} PRIORITY</span>
                <span className="font-semibold text-[15px]">{v.band}</span>
              </div>
              <p className="text-[13px] text-muted mt-1 mb-0 leading-snug"><b className="text-ink/70">Rule applied:</b> {v.rule}</p>
            </div>
          </div>
        </section>

        {/* ---- annotated image ---- */}
        {r.annotated_image_b64 && <>
          <img src={`data:image/jpeg;base64,${r.annotated_image_b64}`} alt="Annotated screening image with colored detection outlines" className="w-full rounded-xl2 shadow-card bg-[#0c1a17]" />
          <div className="flex flex-wrap items-center gap-2 my-3 text-[12px]">
            <Chip color="#D82828">Caries</Chip><Chip color="#2882C8">Gingivitis</Chip><Chip color="#D26E1E">Teeth</Chip>
            {r.image_size && <span className="text-muted text-[11px] ml-auto">{r.image_size.width}×{r.image_size.height}px · YOLO11 on-device</span>}
          </div>
        </>}

        {/* ---- finding cards (only the checks that were run) ---- */}
        <div className="grid sm:grid-cols-3 gap-3">
          {/* Caries */}
          {caries?.status !== 'not_run' && <FindingCard title="Caries" model="Model 1 · ICDAS" status={caries?.status}
            accent="#D82828"
            badge={caries?.status !== 'available' ? null : sevLevel >= 2 ? BADGE.urgent : sevLevel === 1 ? BADGE.found : BADGE.clear}
            headline={caries?.status === 'available' ? sevText : 'Unavailable'}
            headlineClass={sevLevel >= 2 ? 'text-urgent' : sevLevel === 1 ? 'text-review' : 'text-routine'}>
            {caries?.status === 'available' ? <>
              <Meter label="Severity" value={sevLevel} max={3}
                ticks={['None', 'Mild', 'Mod', 'Adv']} barClass={sevLevel >= 2 ? 'bg-urgent' : sevLevel === 1 ? 'bg-review' : 'bg-routine'} />
              <Row k="Regions found" val={`${caries.found ?? 0}`} />
              {Array.isArray(caries.confidence) && caries.confidence.filter(Boolean).length > 0 &&
                <Row k="Top confidence" val={`${Math.round(Math.max(...caries.confidence.filter((x) => x != null)) * 100)}%`} />}
            </> : <Unavailable />}
          </FindingCard>}

          {/* Gingivitis */}
          {gum?.status !== 'not_run' && <FindingCard title="Gingivitis" model="Model 3" status={gum?.status}
            accent="#2882C8"
            badge={gum?.status !== 'available' ? null : !gum.found ? BADGE.clear : (gumConf != null && gumConf > 60 ? BADGE.urgent : BADGE.found)}
            headline={gum?.status === 'available' ? (gum.found ? 'Detected' : 'Not detected') : 'Unavailable'}
            headlineClass={gum?.status === 'available' && gum.found ? (gumConf != null && gumConf > 60 ? 'text-urgent' : 'text-review') : 'text-routine'}>
            {gum?.status === 'available' ? <>
              {gum.found && gumConf != null ? <Meter label="Confidence" value={gumConf} max={100} suffix="%"
                barClass={gumConf > 60 ? 'bg-urgent' : 'bg-review'} threshold={60} />
                : <p className="text-[12.5px] text-muted mt-1 mb-0">No inflammation above the 50% screening threshold.</p>}
              {gum.found && <>
                <Row k="Inflamed regions" val={`${gum.regions?.length ?? 0}`} />
                <Row k="Over 60% → urgent" val={gumConf != null && gumConf > 60 ? 'Yes' : 'No'} />
              </>}
            </> : <Unavailable />}
          </FindingCard>}

          {/* Tooth type */}
          {tooth?.status !== 'not_run' && <FindingCard title="Tooth type" model="Model 2" status={tooth?.status}
            accent="#D26E1E"
            badge={tooth?.status !== 'available' ? null : BADGE.info}
            headline={tooth?.status === 'available' ? `${tooth.count ?? 0} teeth` : 'Unavailable'}
            headlineClass="text-ink">
            {tooth?.status === 'available' ? <>
              {toothTypes.length > 0 ? <div className="mt-1.5 flex flex-col gap-1.5">
                {toothTypes.map(([type, n]) => (
                  <div key={type} className="flex items-center justify-between text-[13px]">
                    <span className="capitalize text-muted">{type}</span>
                    <span className="font-semibold">{n}</span>
                  </div>
                ))}
              </div> : <p className="text-[12.5px] text-muted mt-1 mb-0">Count available; type breakdown not returned.</p>}
            </> : <Unavailable />}
          </FindingCard>}
        </div>

        {/* ---- plain-language summary ---- */}
        <div className="card p-4 mt-3">
          <h3 className="font-semibold text-[13px] text-muted uppercase tracking-wide mb-1.5">Summary</h3>
          <p className="text-sm leading-relaxed text-ink/90 m-0">{r.summary_text}</p>
        </div>

        {/* ---- per-tooth detail (optional) ---- */}
        {tooth?.status === 'available' && tooth.teeth?.length > 0 &&
          <div className="card p-4 mt-3">
            <h3 className="font-semibold text-sm mb-2">Detected tooth regions <span className="text-muted font-normal">· tap to inspect</span></h3>
            <div className="flex flex-wrap gap-2">
              {tooth.teeth.map((item, i) => (
                <button key={`${item.type}-${i}`} onClick={() => setSelectedTooth(selectedTooth === i ? null : i)}
                  className={`px-3 py-1.5 rounded-lg border text-xs capitalize transition-colors ${selectedTooth === i ? 'border-teal text-teal bg-teal/5' : 'border-line text-muted'}`}>
                  {item.type}
                </button>
              ))}
            </div>
            {selectedTooth != null && <p className="text-xs text-muted mt-2 mb-0">
              {tooth.teeth[selectedTooth]?.type} · confidence {Math.round((tooth.teeth[selectedTooth]?.confidence || 0) * 100)}% · tooth number not provided by model.
            </p>}
          </div>}

        {/* ---- doctor review ---- */}
        <section className="card p-5 mt-4">
          <h2 className="text-lg font-semibold mb-1">Doctor review</h2><p className="text-muted text-xs mb-3">AI results are screening support. Record your clinical assessment.</p>
          <label className="text-sm font-medium block">Assessment<select className="field mt-1.5" value={assessment} onChange={(e) => { setAssessment(e.target.value); setReviewSaved(false); }}><option value="">Select assessment</option>{ASSESSMENTS.slice(1).map((item) => <option key={item}>{item}</option>)}</select></label>
          <label className="text-sm font-medium block mt-3">Clinical notes<textarea className="field mt-1.5 min-h-24 resize-y" maxLength={5000} placeholder="Notes for the patient's record" value={notes} onChange={(e) => { setNotes(e.target.value); setReviewSaved(false); }} /></label>
          <button className="btn-ghost mt-3" disabled={reviewBusy} onClick={saveDoctorReview}>{reviewBusy ? 'Saving…' : 'Save doctor review'}</button>
          {!reviewSaved && <p className="text-xs text-muted mt-2 mb-0">Save an assessment before downloading or sending the report.</p>}
        </section>

        <p className="text-xs text-muted leading-relaxed mt-4">AI screening aid only — not a diagnosis. Final assessment must be performed by a qualified dental professional.</p>
        <div className="grid grid-cols-2 gap-3 mt-4">
          <button className="btn-primary" disabled={!reviewSaved} onClick={() => downloadReport(r.id || r._id).catch((e) => toast(e.message))}>Download PDF</button>
          <button className="btn-ghost" disabled={!reviewSaved} onClick={sendEmail}>Send to patient</button>
        </div>
        <div className="flex gap-3 mt-3"><button className="btn-ghost" onClick={reset}>{caseOverride ? 'Back' : 'New screening'}</button>{!caseOverride && <button className="btn-primary" onClick={() => { toast('Screening is in the dentist queue.'); reset(); onDone?.(); }}>Done</button>}</div>
      </div>}
    </div>
  );
}

function FindingCard({ title, model, status, accent, badge, headline, headlineClass = 'text-ink', children }) {
  return (
    <div className="card p-4 flex flex-col border-t-4" style={{ borderTopColor: accent }}>
      <div className="flex items-center justify-between mb-2">
        <span className="font-semibold text-[15px] flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-[3px]" style={{ background: accent }} />{title}
        </span>
        {badge && <span className={`font-display font-bold text-[10px] tracking-wide uppercase px-2 py-0.5 rounded-full ${badge.cls}`}>{badge.text}</span>}
      </div>
      <div className={`font-display font-bold text-xl leading-none ${headlineClass}`}>{headline}</div>
      <div className="text-[11px] text-muted mt-0.5 mb-2">{model}{status && status !== 'available' ? ' · check unavailable' : ''}</div>
      <div className="mt-auto">{children}</div>
    </div>
  );
}

function Meter({ label, value, max, ticks, suffix = '', barClass = 'bg-review', threshold = null }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div className="mt-1">
      <div className="flex items-center justify-between text-[11px] text-muted mb-1">
        <span>{label}</span><span className="font-semibold text-ink">{ticks ? ticks[value] ?? value : `${value}${suffix}`}</span>
      </div>
      <div className="relative h-2 rounded bg-line overflow-hidden">
        <div className={`h-full ${barClass} transition-all`} style={{ width: `${pct}%` }} />
        {threshold != null && <div className="absolute top-0 bottom-0 w-px bg-ink/40" style={{ left: `${threshold}%` }} />}
      </div>
    </div>
  );
}

function Row({ k, val }) {
  return <div className="flex items-center justify-between text-[13px] mt-1.5"><span className="text-muted">{k}</span><span className="font-semibold">{val}</span></div>;
}

function Unavailable() {
  return <p className="text-[12.5px] text-muted mt-1 mb-0">This model did not return a result. Review the photo manually.</p>;
}

function Chip({ color, children }) {
  return <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-line bg-surface text-muted"><span className="w-2.5 h-2.5 rounded-[3px]" style={{ background: color }} />{children}</span>;
}

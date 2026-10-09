import Reveal from '../components/Reveal.jsx';

/* DentalX landing — the scrollable sections below the hero.
   Same visual language as the hero: dark base, coral accent, glass pills,
   Space Grotesk headings, Reveal fade-ups. Scoped under .dx-sections. */

const MODELS = [
  {
    tag: 'Model 1',
    accent: '#FF5C49',
    name: 'Caries detection',
    desc: 'Finds tooth decay on an occlusal photo and grades its ICDAS severity — No / Mild / Moderate / Advanced.',
    stat: '0.82', statlabel: 'test mAP@50 · 1.00 external sensitivity',
  },
  {
    tag: 'Model 2',
    accent: '#7FB7D9',
    name: 'Tooth type',
    desc: 'Classifies every tooth as incisor, canine, premolar or molar — robust across frontal, upper and lower views.',
    stat: '0.95', statlabel: 'mAP@50 · strongest, view-robust',
  },
  {
    tag: 'Model 3',
    accent: '#3FB984',
    name: 'Gingivitis',
    desc: 'Detects inflamed gum regions — erythema, edema and bleeding along the gum line.',
    stat: '0.80', statlabel: 'mAP@50 · honest small-data baseline',
  },
];

const STEPS = [
  { n: '01', title: 'Capture', desc: 'A volunteer takes one intraoral photo on a tablet — plus optional patient details.' },
  { n: '02', title: 'Analyse', desc: 'Three AI models run on the photo at once, fully on-device. No image leaves the machine.' },
  { n: '03', title: 'Triage', desc: 'A transparent rule sorts the patient into URGENT, REVIEW or ROUTINE for the dentist.' },
  { n: '04', title: 'Report', desc: 'An annotated image, priority badge and plain-language summary — with an optional PDF e-mailed to the patient.' },
];

const TRIAGE = [
  { tag: 'Urgent', color: '#FF5C49', rule: 'Moderate / Advanced caries, or gingivitis over 60%', act: 'Dentist sees this patient first' },
  { tag: 'Review', color: '#E0A23E', rule: 'Mild caries, or any gingivitis detected', act: 'Dentist review advised' },
  { tag: 'Routine', color: '#3FB984', rule: 'None of the above', act: 'Routine check' },
];

const RESULTS = [
  { v: '0.95', l: 'Tooth-type mAP@50 — consistent across all views' },
  { v: '1.00', l: 'Caries external-validation lesion sensitivity' },
  { v: '3', l: 'Independent models on a single photo' },
  { v: '100%', l: 'Offline — no patient image leaves the device' },
];

function Section({ id, kicker, title, children }) {
  return (
    <section id={id} className="dx-sec">
      <div className="dx-sec-inner">
        <Reveal className="dx-sec-head" fromY={24}>
          <span className="dx-kicker">{kicker}</span>
          <h2 className="dx-sec-title">{title}</h2>
        </Reveal>
        {children}
      </div>
    </section>
  );
}

export default function LandingSections() {
  return (
    <div className="dx-sections">
      {/* ---- MODELS ---- */}
      <Section id="models" kicker="The three models" title="One photo, three AI checks">
        <div className="dx-grid dx-grid-3">
          {MODELS.map((m, i) => (
            <Reveal key={m.name} className="dx-card" delay={i * 90} fromY={28}>
              <span className="dx-chip" style={{ background: m.accent }}>{m.tag}</span>
              <h3 className="dx-card-title">{m.name}</h3>
              <p className="dx-card-desc">{m.desc}</p>
              <div className="dx-card-stat">
                <span className="dx-stat-num" style={{ color: m.accent }}>{m.stat}</span>
                <span className="dx-stat-label">{m.statlabel}</span>
              </div>
            </Reveal>
          ))}
        </div>
      </Section>

      {/* ---- HOW IT WORKS ---- */}
      <Section id="how" kicker="How it works" title="From photo to decision in seconds">
        <div className="dx-grid dx-grid-4">
          {STEPS.map((s, i) => (
            <Reveal key={s.n} className="dx-step" delay={i * 80} fromY={28}>
              <span className="dx-step-n">{s.n}</span>
              <h3 className="dx-step-title">{s.title}</h3>
              <p className="dx-card-desc">{s.desc}</p>
            </Reveal>
          ))}
        </div>

        <Reveal className="dx-triage" delay={120} fromY={24}>
          <p className="dx-triage-head">The triage rule — shown in the app</p>
          <div className="dx-triage-row">
            {TRIAGE.map((t) => (
              <div key={t.tag} className="dx-triage-card" style={{ borderTopColor: t.color }}>
                <span className="dx-chip" style={{ background: t.color }}>{t.tag}</span>
                <p className="dx-triage-rule">{t.rule}</p>
                <p className="dx-triage-act">{t.act}</p>
              </div>
            ))}
          </div>
          <p className="dx-triage-note">The triage only orders the queue — it never replaces the dentist. Every finding is confirmed clinically.</p>
        </Reveal>
      </Section>

      {/* ---- THE APP ---- */}
      <Section id="app" kicker="The application" title="DentalX Camp — built for the field">
        <div className="dx-grid dx-grid-3">
          <Reveal className="dx-card" fromY={28}>
            <h3 className="dx-card-title">Screen</h3>
            <p className="dx-card-desc">One photo in, an annotated image, priority badge and plain-language summary out.</p>
          </Reveal>
          <Reveal className="dx-card" delay={90} fromY={28}>
            <h3 className="dx-card-title">Queue</h3>
            <p className="dx-card-desc">A list auto-sorted URGENT → REVIEW → ROUTINE, searchable and filterable.</p>
          </Reveal>
          <Reveal className="dx-card" delay={180} fromY={28}>
            <h3 className="dx-card-title">Dashboard</h3>
            <p className="dx-card-desc">Live camp totals, priority and severity breakdown, gingivitis rate, one-click CSV export.</p>
          </Reveal>
        </div>
        <Reveal className="dx-sec-cta" delay={120} fromY={20}>
          <a className="dx-pill dx-pill-coral" href="#/app">Open the screening app</a>
          <span className="dx-stack">React · Node · MongoDB · runs offline</span>
        </Reveal>
      </Section>

      {/* ---- RESULTS ---- */}
      <Section id="results" kicker="Results" title="Real numbers, honestly reported">
        <div className="dx-grid dx-grid-4">
          {RESULTS.map((r, i) => (
            <Reveal key={r.l} className="dx-result" delay={i * 80} fromY={24}>
              <span className="dx-result-num">{r.v}</span>
              <span className="dx-result-label">{r.l}</span>
            </Reveal>
          ))}
        </div>
        <Reveal className="dx-sec-note" delay={120} fromY={16}>
          Research prototype &amp; AI screening aid — not a diagnosis. Every finding must be confirmed by a dentist.
        </Reveal>
      </Section>

      {/* ---- PAGE FOOTER ---- */}
      <footer className="dx-pagefoot">
        <span className="dx-brand-sm">
          <svg className="dx-mark" viewBox="0 0 24 24" fill="none" aria-hidden="true" width="22" height="22">
            <rect x="3" y="3" width="18" height="18" rx="6" fill="#FF5C49" />
            <path d="M8.5 8.5h7M12 8.5v7" stroke="#0E1512" strokeWidth="2.2" strokeLinecap="round" />
          </svg>
          DentalX
        </span>
        <span className="dx-foot-copy">AI-assisted dental screening &amp; triage for community camps.</span>
      </footer>
    </div>
  );
}

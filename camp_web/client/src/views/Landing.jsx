import FluidBackground from '../components/FluidBackground.jsx';
import WordReveal from '../components/WordReveal.jsx';
import Reveal from '../components/Reveal.jsx';
import LandingSections from '../components/LandingSections.jsx';
import '../landing.css';
import '../sections.css';

export default function Landing() {
  return (
    <div className="dx-page">
      <section className="dx-hero">
        <FluidBackground />
        <div className="dx-scrim" aria-hidden="true" />

        {/* NAV */}
        <Reveal as="header" className="dx-nav" delay={150} fromY={-12}>
          <a className="dx-brand" href="/">
            <svg className="dx-mark" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <rect x="3" y="3" width="18" height="18" rx="6" fill="#FF5C49" />
              <path d="M8.5 8.5h7M12 8.5v7" stroke="#0E1512" strokeWidth="2.2" strokeLinecap="round" />
            </svg>
            DentalX
          </a>
          <nav className="dx-navpill" aria-label="Primary">
            <a href="#models">Models</a>
            <a href="#how">How it works</a>
            <a href="#app">The app</a>
            <a href="#results">Results</a>
          </nav>
          <a className="dx-pill dx-pill-light" href="#/app">Open the app</a>
        </Reveal>

        {/* CENTER */}
        <div className="dx-center">
          <Reveal as="p" className="dx-badge" delay={320} fromY={20}>
            <span className="dx-dot" /> Secure &amp; Offline
          </Reveal>

          <h1 className="dx-h1">
            <WordReveal
              text="Instant AI Dental Triage."
              baseDelay={480} stagger={85} duration={720} fromY={26}
              coralFrom={2}
            />
          </h1>

          <p className="dx-sub">
            <WordReveal
              text="Upload one photo. Instantly detect caries, teeth, and gum inflammation. 100% offline."
              baseDelay={1150} stagger={22} duration={600} fromY={14}
            />
          </p>

          <Reveal className="dx-cta-row" delay={1450} fromY={20}>
            <a className="dx-pill dx-pill-coral" href="#/app">Open the screening app</a>
            <a className="dx-pill dx-pill-glass" href="#models">The three models</a>
          </Reveal>
        </div>

        {/* SCROLL CUE */}
        <Reveal as="div" className="dx-scrollcue" delay={1800} fromY={0}>
          <span>Scroll</span>
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" aria-hidden="true">
            <path d="M12 5v14M6 13l6 6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </Reveal>
      </section>

      <LandingSections />
    </div>
  );
}

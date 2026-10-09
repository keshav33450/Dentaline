import { useEffect, useState } from 'react';

// Splits text on spaces and reveals word-by-word (translateY + opacity),
// each word delayed by baseDelay + i*stagger. `coralFrom` colors words from
// that index onward with the accent. Honors prefers-reduced-motion (instant).
export default function WordReveal({ text, baseDelay = 0, stagger = 85, duration = 720, fromY = 26, coralFrom = null, className = '' }) {
  const [go, setGo] = useState(false);
  useEffect(() => {
    const reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced) { setGo(true); return; }
    const t = setTimeout(() => setGo(true), baseDelay);
    return () => clearTimeout(t);
  }, [baseDelay]);

  const words = text.split(' ');
  return (
    <span className={className}>
      {words.map((w, i) => (
        <span
          key={i}
          style={{
            display: 'inline-block',
            willChange: 'transform, opacity',
            transform: go ? 'translateY(0)' : `translateY(${fromY}px)`,
            opacity: go ? 1 : 0,
            transition: `transform ${duration}ms cubic-bezier(0.22,1,0.36,1) ${i * stagger}ms, opacity ${duration}ms cubic-bezier(0.22,1,0.36,1) ${i * stagger}ms`,
            color: coralFrom != null && i >= coralFrom ? 'var(--action)' : undefined,
          }}
        >
          {w}{i < words.length - 1 ? ' ' : ''}
        </span>
      ))}
    </span>
  );
}

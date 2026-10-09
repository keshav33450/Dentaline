import { useEffect, useState } from 'react';

// Reveals a whole block (fade + translateY) after `delay` ms. Honors reduced-motion.
export default function Reveal({ delay = 0, fromY = 20, duration = 700, className = '', style = {}, children, as = 'div' }) {
  const [go, setGo] = useState(false);
  useEffect(() => {
    const reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced) { setGo(true); return; }
    const t = setTimeout(() => setGo(true), delay);
    return () => clearTimeout(t);
  }, [delay]);
  const Tag = as;
  return (
    <Tag
      className={className}
      style={{
        ...style,
        transform: go ? 'translateY(0)' : `translateY(${fromY}px)`,
        opacity: go ? 1 : 0,
        transition: `transform ${duration}ms cubic-bezier(0.2,0,0,1), opacity ${duration}ms cubic-bezier(0.2,0,0,1)`,
      }}
    >
      {children}
    </Tag>
  );
}

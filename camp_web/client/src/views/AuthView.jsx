import React, { useState } from 'react';
import { login, register } from '../lib/api.js';

export default function AuthView({ onSignedIn }) {
  const [mode, setMode] = useState('login');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  async function submit(event) {
    event.preventDefault(); setBusy(true); setError('');
    try {
      const session = mode === 'register'
        ? await register(name, email, password)
        : await login(email, password);
      onSignedIn(session);
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  return (
    <div className="min-h-screen grid place-items-center px-4 py-10 bg-[radial-gradient(ellipse_at_top,_#dff3ee,_transparent_55%)]">
      <div className="w-full max-w-md card p-7 sm:p-9 shadow-card">
        <div className="flex items-center gap-3 mb-7">
          <span className="w-11 h-11 rounded-2xl bg-gradient-premium grid place-items-center text-white font-bold">Dx</span>
          <div><b className="font-display text-xl">DentalX Camp</b><p className="text-muted text-sm m-0">Doctor workspace</p></div>
        </div>
        <h1 className="text-2xl font-bold mb-1">{mode === 'login' ? 'Welcome back' : 'Create your doctor account'}</h1>
        <p className="text-muted text-sm mb-6">Sign in to access your patients and screening history.</p>
        <form onSubmit={submit} className="flex flex-col gap-4">
          {mode === 'register' && <label className="text-sm font-medium">Full name<input className="field mt-1.5" autoComplete="name" required minLength={2} maxLength={100} value={name} onChange={(e) => setName(e.target.value)} /></label>}
          <label className="text-sm font-medium">E-mail<input className="field mt-1.5" type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} /></label>
          <label className="text-sm font-medium">Password<input className="field mt-1.5" type="password" autoComplete={mode === 'login' ? 'current-password' : 'new-password'} required minLength={mode === 'register' ? 12 : 1} value={password} onChange={(e) => setPassword(e.target.value)} />{mode === 'register' && <small className="block text-muted mt-1">Use at least 12 characters.</small>}</label>
          {error && <div role="alert" className="rounded-xl bg-urgent/10 text-urgent px-3.5 py-3 text-sm">{error}</div>}
          <button className="btn-primary mt-1" disabled={busy}>{busy ? 'Please wait…' : mode === 'login' ? 'Sign in' : 'Create account'}</button>
        </form>
        <p className="text-center text-sm text-muted mt-5">
          {mode === 'login' ? 'New to DentalX? ' : 'Already have an account? '}
          <button className="text-teal font-semibold underline underline-offset-2" onClick={() => { setMode(mode === 'login' ? 'register' : 'login'); setError(''); }}>
            {mode === 'login' ? 'Create account' : 'Sign in'}
          </button>
        </p>
        <p className="text-[11px] leading-relaxed text-center text-muted border-t border-line pt-4 mt-5">
          AI screening aid only — not a diagnosis. Final assessment must be performed by a qualified dental professional.
        </p>
      </div>
    </div>
  );
}

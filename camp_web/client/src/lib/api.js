const base = '/api';
const TOKEN = 'dentalx_session';

export function getSession() {
  try { return JSON.parse(sessionStorage.getItem(TOKEN) || 'null'); }
  catch { return null; }
}

export function saveSession(session) { sessionStorage.setItem(TOKEN, JSON.stringify(session)); }
export function clearSession() { sessionStorage.removeItem(TOKEN); }

async function request(path, options = {}) {
  const session = getSession();
  const headers = new Headers(options.headers || {});
  if (session?.token) headers.set('Authorization', `Bearer ${session.token}`);
  if (options.body && !(options.body instanceof FormData)) headers.set('Content-Type', 'application/json');
  const response = await fetch(`${base}${path}`, { ...options, headers });
  if (response.status === 401) {
    clearSession();
    window.dispatchEvent(new Event('dentalx:session-expired'));
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || body.error || `Request failed (${response.status})`);
  }
  const type = response.headers.get('content-type') || '';
  return type.includes('application/json') ? response.json() : response;
}

export const getHealth = () => request('/health');
export const login = (email, password) => request('/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) });
export const register = (name, email, password) => request('/auth/register', { method: 'POST', body: JSON.stringify({ name, email, password }) });
export const logout = () => request('/auth/logout', { method: 'POST' });
export const getProfile = () => request('/auth/me');
export const screen = (formData) => request('/screen', { method: 'POST', body: formData });
export const getQueue = () => request('/queue');
export const getScreening = (id) => request(`/screenings/${encodeURIComponent(id)}`);
export const getStats = () => request('/stats');
export const getPatients = (search = '') => request(`/patients?search=${encodeURIComponent(search)}`);
export const createPatient = (patient) => request('/patients', { method: 'POST', body: JSON.stringify(patient) });
export const getPatientHistory = (id) => request(`/patients/${encodeURIComponent(id)}/screenings`);
export const saveReview = (id, review) => request(`/screenings/${encodeURIComponent(id)}/review`, { method: 'PATCH', body: JSON.stringify(review) });
export const emailCase = (id) => request(`/screenings/${encodeURIComponent(id)}/email`, { method: 'POST' });

export async function downloadReport(id) {
  const response = await request(`/screenings/${encodeURIComponent(id)}/report.pdf`);
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement('a');
  link.href = url; link.download = `DentalX_${id}.pdf`; link.click();
  URL.revokeObjectURL(url);
}

export async function downloadCsv() {
  const response = await request('/export.csv');
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement('a');
  link.href = url; link.download = 'dentalx_screenings.csv'; link.click();
  URL.revokeObjectURL(url);
}

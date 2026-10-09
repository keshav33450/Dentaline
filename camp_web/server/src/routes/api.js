import express from 'express';
import multer from 'multer';
import Case from '../models/Case.js';
import { runModels, annotate, checkHealth } from '../lib/models.js';
import { interpret } from '../lib/triage.js';
import { buildPdf } from '../lib/pdf.js';
import { sendReport, smtpConfigured } from '../lib/email.js';

const router = express.Router();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 20 * 1024 * 1024 } });

const localDay = () => {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
};

// health of model services + smtp
router.get('/health', async (_req, res) => {
  const h = await checkHealth();
  res.json({ ...h, smtp_configured: smtpConfigured() });
});

// screen a patient
router.post('/screen', upload.single('file'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'no image' });
  const buf = req.file.buffer;
  let results, report, annotated;
  try {
    results = await runModels(buf, req.file.originalname);
    report = interpret(results);
    annotated = await annotate(buf, results);
  } catch (e) {
    return res.status(500).json({ error: 'processing failed: ' + (e.message || e) });
  }

  const b = req.body || {};
  const doc = await Case.create({
    day: localDay(),
    patient_name: b.patient_name || '', age: b.age || '', sex: b.sex || '',
    email: b.email || '', camp_id: b.camp_id || '',
    priority: report.priority,
    tooth_count: report.tooth_count, tooth_breakdown: report.tooth_breakdown,
    caries_count: report.caries_count, caries_severity: report.caries_severity,
    caries_highest: report.caries_highest,
    gingivitis: report.gingivitis, gingivitis_conf: report.gingivitis_conf,
    summary_text: report.summary_text, available: report.available,
    annotated, original: buf, raw: results,
  });

  const resp = {
    id: doc._id, ...report,
    annotated_image_b64: annotated.toString('base64'),
    emailed: false, email_status: null,
  };

  // optional email
  if (String(b.send_pdf).toLowerCase() === 'true' && b.email) {
    const pdf = await buildPdf(doc);
    const r = await sendReport(b.email, doc, pdf);
    resp.emailed = r.ok; resp.email_status = r.status;
    if (r.ok) { doc.emailed = true; await doc.save(); }
  }
  res.json(resp);
});

// queue (priority-sorted) — lightweight (no image blobs)
router.get('/queue', async (_req, res) => {
  const rows = await Case.aggregate([
    { $addFields: { _order: { $switch: { branches: [
      { case: { $eq: ['$priority', 'URGENT'] }, then: 0 },
      { case: { $eq: ['$priority', 'REVIEW'] }, then: 1 },
    ], default: 2 } } } },
    { $sort: { _order: 1, ts: -1 } },
    { $project: { annotated: 0, original: 0, raw: 0, _order: 0 } },
  ]);
  res.json(rows);
});

// one case (with annotated image)
router.get('/case/:id', async (req, res) => {
  const d = await Case.findById(req.params.id).lean().catch(() => null);
  if (!d) return res.status(404).json({ error: 'not found' });
  const out = { ...d };
  if (d.annotated) out.annotated_image_b64 = Buffer.from(d.annotated.buffer || d.annotated).toString('base64');
  delete out.annotated; delete out.original; delete out.raw;
  res.json(out);
});

// email a stored case
router.post('/case/:id/email', async (req, res) => {
  const d = await Case.findById(req.params.id).catch(() => null);
  if (!d) return res.status(404).json({ error: 'not found' });
  if (!d.email) return res.status(400).json({ ok: false, status: 'no e-mail on file' });
  const pdf = await buildPdf(d);
  const r = await sendReport(d.email, d, pdf);
  if (r.ok) { d.emailed = true; await d.save(); }
  res.json(r);
});

// download PDF
router.get('/case/:id/pdf', async (req, res) => {
  const d = await Case.findById(req.params.id).catch(() => null);
  if (!d) return res.status(404).json({ error: 'not found' });
  const pdf = await buildPdf(d);
  res.setHeader('Content-Type', 'application/pdf');
  res.setHeader('Content-Disposition', `attachment; filename="screening_${String(d._id).slice(-8)}.pdf"`);
  res.send(pdf);
});

// today's stats
router.get('/stats', async (_req, res) => {
  const day = localDay();
  const rows = await Case.find({ day }).select('priority caries_severity gingivitis').lean();
  const priority = { URGENT: 0, REVIEW: 0, ROUTINE: 0 };
  const severity = {};
  let gum = 0;
  for (const r of rows) {
    priority[r.priority] = (priority[r.priority] || 0) + 1;
    severity[r.caries_severity] = (severity[r.caries_severity] || 0) + 1;
    if (r.gingivitis) gum++;
  }
  res.json({ day, total: rows.length, priority, severity, gingivitis_rate: rows.length ? +(gum / rows.length).toFixed(3) : 0 });
});

// CSV export
router.get('/export.csv', async (_req, res) => {
  const rows = await Case.find().sort({ ts: -1 })
    .select('patient_name age sex email camp_id priority tooth_count caries_severity caries_count gingivitis gingivitis_conf emailed summary_text ts').lean();
  const head = ['id', 'datetime', 'patient_name', 'age', 'sex', 'email', 'camp_id', 'priority', 'tooth_count', 'caries_severity', 'caries_count', 'gingivitis', 'gingivitis_conf', 'emailed', 'summary'];
  const esc = (v) => `"${String(v ?? '').replace(/"/g, '""')}"`;
  const lines = [head.join(',')];
  for (const r of rows) {
    lines.push([
      String(r._id).slice(-8), new Date(r.ts).toLocaleString(), r.patient_name, r.age, r.sex, r.email, r.camp_id,
      r.priority, r.tooth_count, r.caries_severity, r.caries_count, r.gingivitis ? 'Yes' : 'No',
      (r.gingivitis_conf || 0).toFixed(3), r.emailed ? 'Yes' : 'No', r.summary_text,
    ].map(esc).join(','));
  }
  res.setHeader('Content-Type', 'text/csv');
  res.setHeader('Content-Disposition', 'attachment; filename="dentalx_camp_report.csv"');
  res.send(lines.join('\n'));
});

export default router;

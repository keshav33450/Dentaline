// Optional e-mail of the PDF report via SMTP (nodemailer). Off until configured.
import nodemailer from 'nodemailer';

function configured() {
  return !!(process.env.SMTP_HOST && process.env.SMTP_USER && process.env.SMTP_PASS);
}
export function smtpConfigured() { return configured(); }

export async function sendReport(toAddr, c, pdfBuffer) {
  if (!configured()) return { ok: false, status: 'SMTP not configured (set SMTP_* env vars)' };
  if (!toAddr) return { ok: false, status: 'no recipient e-mail' };
  try {
    const transport = nodemailer.createTransport({
      host: process.env.SMTP_HOST,
      port: Number(process.env.SMTP_PORT || 587),
      secure: Number(process.env.SMTP_PORT) === 465,
      auth: { user: process.env.SMTP_USER, pass: process.env.SMTP_PASS },
    });
    const camp = process.env.CAMP_NAME || 'DentalX Community Dental Camp';
    await transport.sendMail({
      from: process.env.SMTP_FROM || process.env.SMTP_USER,
      to: toAddr,
      subject: `Your dental screening report — ${camp}`,
      text: `Dear ${c.patient_name || 'participant'},\n\n`
        + `Attached is your dental screening report from ${camp}.\n`
        + `Screening priority: ${c.priority}.\n\n`
        + 'This is an AI screening aid, not a diagnosis. Please have the findings confirmed by a dentist.\n\n'
        + 'Regards,\nDentalX Camp team',
      attachments: [{ filename: `dental_screening_${String(c._id || '').slice(-8)}.pdf`, content: pdfBuffer }],
    });
    return { ok: true, status: 'sent' };
  } catch (e) {
    return { ok: false, status: String(e.message || e).slice(0, 200) };
  }
}

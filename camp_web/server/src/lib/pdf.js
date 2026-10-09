// One-page PDF screening report (pdfkit). Returns a Buffer.
import PDFDocument from 'pdfkit';

const CAMP = () => process.env.CAMP_NAME || 'DentalX Community Dental Camp';
const PCOL = { URGENT: '#C0392B', REVIEW: '#E08A1E', ROUTINE: '#1E9E6A' };

export function buildPdf(c) {
  return new Promise((resolve, reject) => {
    try {
      const doc = new PDFDocument({ size: 'A4', margin: 50 });
      const chunks = [];
      doc.on('data', (d) => chunks.push(d));
      doc.on('end', () => resolve(Buffer.concat(chunks)));

      // header
      doc.fillColor('#14302B').font('Helvetica-Bold').fontSize(18).text(CAMP());
      doc.moveDown(0.2).fillColor('#5C726C').font('Helvetica').fontSize(11)
        .text('DentalX Camp — AI-assisted dental screening report');
      doc.moveDown(0.8);

      // priority badge
      const prio = c.priority || 'ROUTINE';
      const by = doc.y;
      doc.roundedRect(50, by, 150, 26, 5).fill(PCOL[prio] || '#777');
      doc.fillColor('#fff').font('Helvetica-Bold').fontSize(13).text(`PRIORITY: ${prio}`, 60, by + 7);
      doc.fillColor('#14302B').font('Helvetica').fontSize(11);
      doc.y = by + 40;

      // patient line
      doc.text(`Patient: ${c.patient_name || '-'}    Age: ${c.age || '-'}    Sex: ${c.sex || '-'}`);
      const when = c.ts ? new Date(c.ts) : new Date();
      doc.text(`Date: ${when.toLocaleString()}    Case ID: ${String(c._id || '').slice(-8)}`);
      doc.moveDown(0.8);

      // findings
      doc.font('Helvetica-Bold').fontSize(12).text('Findings'); doc.moveDown(0.3);
      doc.font('Helvetica').fontSize(11);
      const tc = (c.tooth_count === null || c.tooth_count === undefined) ? 'n/a' : c.tooth_count;
      const cc = (c.caries_count === null || c.caries_count === undefined) ? 'n/a' : c.caries_count;
      const gum = c.gingivitis ? `Yes (${Math.round((c.gingivitis_conf || 0) * 100)}%)` : (c.gingivitis === false ? 'No' : 'n/a');
      doc.list([
        `Teeth detected: ${tc}`,
        `Caries: ${cc} region(s) — severity ${c.caries_severity || 'n/a'}`,
        `Gum inflammation: ${gum}`,
      ], { bulletRadius: 2 });
      doc.moveDown(0.6);

      // summary
      doc.font('Helvetica-Bold').fontSize(12).text('Summary'); doc.moveDown(0.3);
      doc.font('Helvetica').fontSize(11).text(c.summary_text || '', { width: 495 });
      doc.moveDown(0.6);

      // annotated image — cap height so the report stays one page
      if (c.annotated && c.annotated.length) {
        const avail = 760 - doc.y;                 // space left above the footer
        const h = Math.max(120, Math.min(260, avail));
        try { doc.image(c.annotated, { fit: [300, h] }); } catch { /* ignore */ }
      }

      // footer disclaimer
      doc.fontSize(9).fillColor('#5C726C').font('Helvetica-Oblique')
        .text('AI screening aid — not a diagnosis. Findings must be confirmed by a dentist.',
          50, 790, { width: 495, align: 'left' });

      doc.end();
    } catch (e) { reject(e); }
  });
}

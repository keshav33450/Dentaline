// Merge the three model outputs into one report + triage priority.
// Ported from the verified Python version; same transparent rules.

const SEV_RANK = { no_caries: 0, none: 0, mild: 1, caries: 1, moderate: 2, advanced: 3 };

export function interpret(results) {
  const avail = {};
  for (const k of ['occlusal', 'tooth', 'gingivitis']) {
    avail[k] = !(results[k] && results[k]._unavailable);
  }
  const out = { available: avail };

  // Model 2: tooth type
  const tooth = results.tooth || {};
  if (tooth._unavailable) { out.tooth_count = null; out.tooth_breakdown = {}; }
  else {
    out.tooth_count = tooth.n_teeth ?? (tooth.teeth ? tooth.teeth.length : 0);
    out.tooth_breakdown = tooth.counts || {};
  }

  // Model 1: occlusal caries  (caries[].severity, summary{}, highest)
  const occ = results.occlusal || {};
  if (occ._unavailable) { out.caries_count = null; out.caries_severity = 'unavailable'; out.caries_highest = ''; }
  else {
    const caries = occ.caries || [];
    out.caries_count = caries.length;
    let worst = 'none';
    for (const d of caries) {
      const s = String(d.severity || '').toLowerCase();
      if ((SEV_RANK[s] || 0) > (SEV_RANK[worst] || 0)) worst = s;
    }
    const pretty = { none: 'None', mild: 'Mild', caries: 'Present', moderate: 'Moderate', advanced: 'Advanced' };
    out.caries_severity = pretty[worst] || (worst.charAt(0).toUpperCase() + worst.slice(1));
    out.caries_highest = occ.highest || '';
  }

  // Model 3: gingivitis
  const gum = results.gingivitis || {};
  if (gum._unavailable) { out.gingivitis = null; out.gingivitis_conf = 0; }
  else {
    out.gingivitis = !!gum.gingivitis_detected;
    out.gingivitis_conf = Number(gum.max_confidence || 0);
  }

  // triage
  const sevRank = (out.caries_severity && !['unavailable', null].includes(out.caries_severity))
    ? (SEV_RANK[out.caries_severity.toLowerCase()] || 0) : 0;
  const gconf = out.gingivitis_conf || 0;
  let prio;
  if (sevRank >= 2 || gconf > 0.6) prio = 'URGENT';
  else if (sevRank === 1 || out.gingivitis) prio = 'REVIEW';
  else prio = 'ROUTINE';
  out.priority = prio;

  // plain-language summary
  const parts = [];
  if (out.tooth_count) parts.push(`${out.tooth_count} teeth detected`);
  if (!['unavailable', 'None', null].includes(out.caries_severity)) parts.push(`possible caries (${out.caries_severity.toLowerCase()})`);
  else if (out.caries_severity === 'None') parts.push('no obvious caries');
  if (out.gingivitis) parts.push(`signs of gum inflammation (${Math.round(gconf * 100)}% confidence)`);
  else if (out.gingivitis === false) parts.push('no obvious gum inflammation');
  const body = parts.length ? parts.join(', ') : 'screening completed';
  const verdict = { URGENT: 'Recommend the dentist see this patient first.', REVIEW: 'Recommend a dentist review.', ROUTINE: 'No urgent findings; routine check advised.' }[prio];
  out.summary_text = `Screening found ${body}. ${verdict}`;
  return out;
}

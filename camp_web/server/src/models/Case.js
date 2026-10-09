import mongoose from 'mongoose';

const CaseSchema = new mongoose.Schema({
  ts: { type: Date, default: Date.now, index: true },
  day: { type: String, index: true },              // YYYY-MM-DD (local camp day)
  patient_name: String,
  age: String,
  sex: String,
  email: String,
  camp_id: String,
  priority: { type: String, enum: ['URGENT', 'REVIEW', 'ROUTINE'], index: true },
  tooth_count: Number,            // null-safe: stored as number or null
  tooth_breakdown: Object,
  caries_count: Number,
  caries_severity: String,
  caries_highest: String,
  gingivitis: Boolean,
  gingivitis_conf: Number,
  summary_text: String,
  available: Object,
  annotated: Buffer,              // annotated JPEG (for queue thumbnails / PDF)
  original: Buffer,               // original JPEG
  emailed: { type: Boolean, default: false },
  raw: Object,                    // raw model outputs
}, { versionKey: false });

export default mongoose.model('Case', CaseSchema);

"""Phase VI - automated PDF screening report (reportlab)."""
from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import cv2

from .labels import SEVERITY_LABEL


def make_pdf(result: dict, overlay_bgr, out: str | Path | None = None, *, patient_code: str = "",
             clinic: str = "Department of Pediatric and Preventive Dentistry", photo_name: str = "") -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    small = ss["BodyText"].clone("small", fontSize=8, leading=10)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                            title="Occlusal caries AI screening report")
    el = [Paragraph("<b>Occlusal Caries - AI Screening Report</b>", ss["Title"]),
          Paragraph(f"{clinic}<br/>Patient code: <b>{patient_code or '-'}</b> &nbsp;&nbsp; Photo: {photo_name or '-'} "
                    f"&nbsp;&nbsp; Date: {dt.datetime.now():%d-%m-%Y %H:%M}", ss["BodyText"]), Spacer(1, 4 * mm)]

    q = result["quality"]
    qtxt = "Photo quality: <font color='green'><b>acceptable</b></font>" if q["ok"] else \
        f"Photo quality: <font color='red'><b>{'; '.join(q['reasons'])}</b></font> - consider retaking the photo"
    el += [Paragraph(qtxt, ss["BodyText"]), Spacer(1, 3 * mm)]

    rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    ok, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    h, w = overlay_bgr.shape[:2]
    maxw, maxh = 180 * mm, 95 * mm
    s = min(maxw / w, maxh / h)
    el += [Image(io.BytesIO(png.tobytes()), width=w * s, height=h * s), Spacer(1, 4 * mm)]

    rows = [["Tooth", "Type", "Caries severity", "Confidence", "Screening note", "Review"]]
    sev_col = {"No Caries": colors.HexColor("#3cb44b"), "Mild": colors.HexColor("#f0d000"),
               "Moderate": colors.HexColor("#ff8c00"), "Advanced": colors.HexColor("#dc2828")}
    for t in result["teeth"]:
        rows.append([f"#{t['tooth']}", t["tooth_type"].title(), t["severity_label"],
                     f"{t['severity_confidence']:.0%}", Paragraph(t["note"], small), "Yes" if t["needs_review"] else "-"])
    tbl = Table(rows, colWidths=[14 * mm, 20 * mm, 28 * mm, 20 * mm, 82 * mm, 14 * mm], repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
             ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    for i, t in enumerate(result["teeth"], 1):
        style.append(("BACKGROUND", (2, i), (2, i), sev_col[t["severity_label"]]))
    tbl.setStyle(TableStyle(style))
    el += [Paragraph("<b>Tooth-wise findings</b> (teeth numbered left to right on the photo)", ss["Heading3"]), tbl, Spacer(1, 4 * mm)]

    sm = result["summary"]
    counts = ", ".join(f"{SEVERITY_LABEL[k]}: {v}" for k, v in sm["severity_counts"].items())
    el += [Paragraph("<b>Clinical screening summary</b>", ss["Heading3"]),
           Paragraph(f"Posterior teeth analysed: <b>{sm['teeth_analysed']}</b> (premolars {sm['premolars']}, molars {sm['molars']}). "
                     f"{counts}. Highest severity: <b>{sm['highest_severity'] or '-'}</b>. "
                     f"Teeth flagged for clinician review: <b>{sm['needs_review']}</b>.", ss["BodyText"]),
           Spacer(1, 3 * mm),
           Paragraph("Severity categories derived from ICDAS: No Caries (0), Mild - enamel lesion (1-2), "
                     "Moderate - early dentin involvement (3-4), Advanced - deep dentin / cavitation (5-6).", small),
           Spacer(1, 2 * mm), Paragraph(f"<i>{result['disclaimer']}</i>", small),
           Spacer(1, 8 * mm), Paragraph("Clinician: ______________________ &nbsp;&nbsp; Signature: ______________", ss["BodyText"])]
    doc.build(el)
    data = buf.getvalue()
    if out:
        Path(out).write_bytes(data)
    return data

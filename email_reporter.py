"""
email_reporter.py — Automated Email Report Sender

Sends the PDF report + summary stats via email (SMTP).
Supports Gmail App Passwords, Outlook, and custom SMTP servers.

Usage
-----
reporter = EmailReporter(EmailConfig(
    smtp_host="smtp.gmail.com",
    smtp_port=587,
    sender_email="your@gmail.com",
    sender_password="app-password-here",   # Gmail App Password
))
reporter.send(
    to=["municipality@chennai.gov.in"],
    report_data=data,
    pdf_path="reports/report.pdf",
)
"""

from __future__ import annotations

import logging
import smtplib
import ssl
from dataclasses import dataclass, field
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class EmailConfig:
    smtp_host:       str   = "smtp.gmail.com"
    smtp_port:       int   = 587
    sender_email:    str   = ""
    sender_password: str   = ""       # Gmail App Password (not account password)
    use_tls:         bool  = True
    sender_name:     str   = "Road-AI Monitor"
    timeout_sec:     int   = 30

    def validate(self) -> None:
        if not self.sender_email:
            raise ValueError("sender_email is required.")
        if not self.sender_password:
            raise ValueError("sender_password is required.")


# ── HTML email template ───────────────────────────────────────────────────────

def _build_html(data) -> str:
    score  = getattr(data, "avg_health_score", 0)
    band   = ("Good"     if score >= 80 else
              "Moderate" if score >= 60 else
              "Poor"     if score >= 40 else "Critical")
    color  = ("#2ecc71" if score >= 80 else
              "#f39c12" if score >= 60 else
              "#e74c3c" if score >= 40 else "#8e44ad")

    trend_sym = {"improving": "↑", "stable": "→", "worsening": "↓"}.get(
        getattr(data, "score_trend", "stable"), "→"
    )

    rows = [
        ("Health Score",      f"{score:.1f} / 100"),
        ("Condition",         band),
        ("Total Detections",  str(getattr(data, "total_detections", 0))),
        ("Flagged Frames",    str(len(getattr(data, "flagged_frames", [])))),
        ("Est. Repair Cost",  f"₹{getattr(data, 'total_cost_inr', 0):,.0f}"),
        ("Budget Tier",       getattr(data, "budget_tier", "—")),
        ("Urgency",           f"Within {getattr(data, 'urgency_days', 30)} day(s)"),
        ("Trend",             f"{trend_sym} {getattr(data, 'score_trend', 'stable').title()}"),
        ("Distance",          f"{getattr(data, 'total_distance_km', 0):.2f} km"),
        ("Surveyed",          getattr(data, "surveyed_date", datetime.now().strftime("%d %B %Y"))),
    ]

    rows_html = "".join(
        f"""<tr style="background:{'#0f1117' if i%2==0 else '#13161e'}">
              <td style="padding:8px 14px;color:#7c8299;font-size:12px;
                         border-right:1px solid #21262d">{k}</td>
              <td style="padding:8px 14px;color:#e8eaf2;font-size:12px;
                         font-weight:600">{v}</td>
            </tr>"""
        for i, (k, v) in enumerate(rows)
    )

    class_rows = ""
    for cls, cnt in sorted(
        getattr(data, "class_breakdown", {}).items(), key=lambda x: -x[1]
    ):
        class_rows += f"""
        <tr>
          <td style="padding:6px 14px;color:#e8eaf2;font-size:11px">{cls}</td>
          <td style="padding:6px 14px;color:#f39c12;font-size:11px;
                     font-weight:700;text-align:right">{cnt}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body style="margin:0;padding:0;background:#090c10;font-family:'Segoe UI',sans-serif">
  <div style="max-width:580px;margin:30px auto;background:#0d1117;
               border:1px solid #21262d;border-radius:12px;overflow:hidden">

    <!-- Header -->
    <div style="background:#00e5a0;padding:24px 28px">
      <div style="font-size:11px;font-weight:700;letter-spacing:.1em;
                  color:#0d1117;margin-bottom:4px">ROAD-AI MONITOR</div>
      <div style="font-size:22px;font-weight:800;color:#0d1117">
        Road Damage Report
      </div>
      <div style="font-size:12px;color:#0d5530;margin-top:4px">
        {getattr(data, 'location', 'Road Survey')} &nbsp;·&nbsp;
        {getattr(data, 'surveyed_date', '')}
      </div>
    </div>

    <!-- Score banner -->
    <div style="background:#13161e;padding:20px 28px;
                border-bottom:1px solid #21262d;text-align:center">
      <div style="font-size:11px;color:#7c8299;letter-spacing:.08em;
                  margin-bottom:6px">OVERALL HEALTH SCORE</div>
      <div style="font-size:52px;font-weight:800;color:{color};line-height:1">
        {score:.0f}
      </div>
      <div style="font-size:13px;font-weight:600;color:{color};margin-top:4px">
        {band.upper()}
      </div>
    </div>

    <!-- Stats table -->
    <table style="width:100%;border-collapse:collapse">
      {rows_html}
    </table>

    <!-- Class breakdown -->
    {"<div style='padding:16px 28px 4px;border-top:1px solid #21262d'><div style='font-size:10px;color:#7c8299;letter-spacing:.08em;margin-bottom:8px'>DAMAGE CLASS BREAKDOWN</div><table style='width:100%;border-collapse:collapse'>" + class_rows + "</table></div>" if class_rows else ""}

    <!-- Alert box -->
    {"<div style='margin:16px 20px;padding:12px 16px;background:#1a0a0a;border:1px solid #e74c3c;border-radius:8px;color:#e74c3c;font-size:12px'><b>⚠ Urgent Action Required</b><br>Road condition is critical. Please arrange immediate inspection and repair.</div>" if score < 40 else ""}

    <!-- Footer -->
    <div style="padding:16px 28px;background:#090c10;
                border-top:1px solid #21262d;text-align:center">
      <div style="font-size:10px;color:#3a3f52">
        Generated automatically by Road-AI · Full PDF report attached
      </div>
      <div style="font-size:9px;color:#3a3f52;margin-top:4px">
        For official use only. Cost estimates are approximate.
      </div>
    </div>

  </div>
</body>
</html>"""


def _build_plain(data) -> str:
    score = getattr(data, "avg_health_score", 0)
    return f"""Road Damage Assessment Report
==============================
Location  : {getattr(data, 'location', '—')}
Date      : {getattr(data, 'surveyed_date', '—')}

OVERALL HEALTH SCORE: {score:.1f}/100

Total Detections  : {getattr(data, 'total_detections', 0)}
Est. Repair Cost  : INR {getattr(data, 'total_cost_inr', 0):,.0f}
Budget Tier       : {getattr(data, 'budget_tier', '—')}
Urgency           : Within {getattr(data, 'urgency_days', 30)} day(s)
Trend             : {getattr(data, 'score_trend', 'stable').title()}

Please find the full PDF report attached.

-- Road-AI Monitor (automated message)
"""


# ── Sender ────────────────────────────────────────────────────────────────────

@dataclass
class EmailReporter:
    """
    Sends road damage reports via email.

    Parameters
    ----------
    config : EmailConfig
        SMTP credentials and server settings.

    Example
    -------
    reporter = EmailReporter(EmailConfig(
        sender_email="road.ai@gmail.com",
        sender_password="xxxx xxxx xxxx xxxx",  # Gmail App Password
    ))
    reporter.send(
        to=["officer@municipality.gov.in"],
        report_data=data,
        pdf_path="output/report.pdf",
    )
    """

    config: EmailConfig

    def send(
        self,
        to:            list[str],
        report_data,                          # ReportData instance
        pdf_path:      Optional[str]  = None,
        cc:            Optional[list[str]] = None,
        subject:       Optional[str]  = None,
        extra_files:   Optional[list[str]] = None,  # e.g. ["map.html", "data.csv"]
    ) -> bool:
        """
        Send report email with optional PDF attachment.

        Returns True on success, False on failure.
        """
        self.config.validate()

        if not to:
            log.error("No recipients specified.")
            return False

        location = getattr(report_data, "location", "Road Survey")
        date     = getattr(report_data, "surveyed_date",
                           datetime.now().strftime("%d %b %Y"))
        subject  = subject or f"Road Damage Report — {location} — {date}"

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = (f"{self.config.sender_name} "
                          f"<{self.config.sender_email}>")
        msg["To"]      = ", ".join(to)
        if cc:
            msg["Cc"]  = ", ".join(cc)

        # Plain + HTML body
        msg.attach(MIMEText(_build_plain(report_data), "plain"))
        msg.attach(MIMEText(_build_html(report_data),  "html"))

        # Attach PDF
        if pdf_path and Path(pdf_path).exists():
            self._attach_file(msg, pdf_path)
            log.info(f"Attaching PDF: {pdf_path}")

        # Attach extras
        for fp in (extra_files or []):
            if Path(fp).exists():
                self._attach_file(msg, fp)

        # Send
        all_recipients = to + (cc or [])
        return self._send_smtp(msg, all_recipients)

    def send_alert(
        self,
        to:    list[str],
        score: float,
        location: str = "",
        frame: int = 0,
    ) -> bool:
        """
        Send a lightweight instant alert email (no PDF) for critical frames.
        """
        self.config.validate()
        subject = f"🚨 CRITICAL Road Damage Alert — {location or 'Unknown'}"
        html = f"""<div style="font-family:sans-serif;padding:20px">
        <h2 style="color:#e74c3c">⚠ Critical Road Damage Detected</h2>
        <p>Location: <b>{location}</b></p>
        <p>Frame: <b>#{frame}</b></p>
        <p>Health Score: <b style="color:#e74c3c">{score:.1f}/100</b></p>
        <p>Immediate inspection recommended.</p>
        </div>"""

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"{self.config.sender_name} <{self.config.sender_email}>"
        msg["To"]      = ", ".join(to)
        msg.attach(MIMEText(html, "html"))
        return self._send_smtp(msg, to)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _attach_file(self, msg: MIMEMultipart, filepath: str) -> None:
        path = Path(filepath)
        with open(path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition",
                        f'attachment; filename="{path.name}"')
        msg.attach(part)

    def _send_smtp(self, msg: MIMEMultipart, recipients: list[str]) -> bool:
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(self.config.smtp_host,
                              self.config.smtp_port,
                              timeout=self.config.timeout_sec) as server:
                if self.config.use_tls:
                    server.ehlo()
                    server.starttls(context=context)
                    server.ehlo()
                server.login(self.config.sender_email,
                             self.config.sender_password)
                server.sendmail(self.config.sender_email,
                                recipients, msg.as_string())
            log.info(f"Email sent to: {', '.join(recipients)}")
            return True
        except smtplib.SMTPAuthenticationError:
            log.error("SMTP auth failed. Check email/password. "
                      "For Gmail use an App Password.")
            return False
        except smtplib.SMTPException as e:
            log.error(f"SMTP error: {e}")
            return False
        except Exception as e:
            log.error(f"Email send failed: {e}")
            return False

"""Builds the submission PDF from reports/report.md (title page + figures)."""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak,
                                Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

from common import ROOT

REPORTS = ROOT / "reports"
OUT = REPORTS / "Project29_T_Manoj_Paddy_Disease_CAE_Report.pdf"
STUDENT = [
    ("Student", "T Manoj"),
    ("Student ID", "2023004133"),
    ("Project no.", "29"),
    ("Domain", "Agriculture"),
    ("Course mapping", "Unit 2 / Unit 3 · CO2, CO3"),
    ("Architecture (assigned)", "Convolutional Autoencoder"),
    ("Training method (assigned)", "Mini-Batch Gradient Descent"),
    ("Difficulty", "Intermediate"),
    ("Dataset", "Paddy Doctor: Paddy Disease Classification (Kaggle)"),
    ("Date", "October 2026"),
]
FOOTER = "Project 29 · T Manoj (2023004133) · Paddy Disease Representation using a CAE"

F = r"C:\Windows\Fonts"
pdfmetrics.registerFont(TTFont("Body", f"{F}\\arial.ttf"))
pdfmetrics.registerFont(TTFont("Body-B", f"{F}\\arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Body-I", f"{F}\\ariali.ttf"))
pdfmetrics.registerFont(TTFont("Body-BI", f"{F}\\arialbi.ttf"))
pdfmetrics.registerFont(TTFont("Code", f"{F}\\consola.ttf"))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-B", italic="Body-I", boldItalic="Body-BI")

INK, MUTED, RULE, ACCENT = (colors.HexColor(c) for c in ("#1a1a1a", "#555555", "#c8c8c8", "#1f4e79"))
S = {
    "body": ParagraphStyle("body", fontName="Body", fontSize=10, leading=14.2, textColor=INK, spaceAfter=6),
    "h1": ParagraphStyle("h1", fontName="Body-B", fontSize=14, leading=18, textColor=ACCENT,
                         spaceBefore=14, spaceAfter=6, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="Body-B", fontSize=11.5, leading=15, textColor=INK,
                         spaceBefore=10, spaceAfter=4, keepWithNext=1),
    "lead": ParagraphStyle("lead", fontName="Body", fontSize=10, leading=14.2, textColor=INK,
                           spaceAfter=4, keepWithNext=1),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.6, leading=11, textColor=INK),
    "cap": ParagraphStyle("cap", fontName="Body-I", fontSize=8.8, leading=12, textColor=MUTED,
                          alignment=TA_CENTER, spaceAfter=10),
    "title": ParagraphStyle("title", fontName="Body-B", fontSize=22, leading=28, textColor=ACCENT,
                            alignment=TA_CENTER),
    "sub": ParagraphStyle("sub", fontName="Body", fontSize=12.5, leading=17, textColor=MUTED,
                          alignment=TA_CENTER),
}
WIDTH = A4[0] - 4 * cm


def inline(s):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![*\w])\*(?![\s*])(.+?)(?<![\s*])\*(?![*\w])", r"<i>\1</i>", s)
    s = re.sub(r"`(.+?)`", r'<font name="Code">\1</font>', s)
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)


def figure(path, max_h=10 * cm):
    w, h = PILImage.open(path).size
    scale = min(WIDTH / w, max_h / h)
    return Image(str(path), w * scale, h * scale)


def equation():
    path = REPORTS / "figs" / "_sgd_equation.png"
    fig = plt.figure(figsize=(6, 0.6))
    fig.text(0.5, 0.5, r"$\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{m} \sum_{i \in B_t}"
             r" \nabla_\theta\, L(x_i;\, \theta_t)$", ha="center", va="center", fontsize=15)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return figure(path, max_h=1.4 * cm)


def table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [cells[0]] + cells[2:]  # drop the |---| separator
    ncol = len(cells[0])
    lens = [max(len(re.sub(r"[*`]", "", r[j])) for r in cells) for j in range(ncol)]
    weights = [min(max(n, 12), 45) for n in lens]
    widths = [WIDTH * w / sum(weights) for w in weights]
    data = [[Paragraph(f"<b>{inline(c)}</b>" if i == 0 and c else inline(c), S["cell"]) for c in r]
            for i, r in enumerate(cells)]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return [t, Spacer(1, 8)]


def bullets(items, numbered):
    flow = [ListItem(Paragraph(inline(i), S["body"]), leftIndent=14) for i in items]
    kw = {"bulletType": "1", "bulletFormat": "%s."} if numbered else {"bulletType": "bullet", "start": "•"}
    return [ListFlowable(flow, leftIndent=14, bulletFontName="Body", bulletFontSize=9.5, **kw)]


def body_story(md):
    lines = md.splitlines()
    lines = lines[next(i for i, l in enumerate(lines) if l.startswith("## ")):]  # title page replaces header
    story, para, i = [], [], 0

    def flush():
        if para:
            text = " ".join(para)
            figs = re.findall(r"\]\((figs/[^)]+\.png)\)", text)
            text = re.sub(r"\s*\(\[figure\]\([^)]+\)\)", "", text)  # figure is embedded below
            if re.fullmatch(r"\*\*[^*]+\*\*", text):
                style = S["lead"]  # bold run-in heading: keep with what follows
            elif text.startswith("*") and not text.startswith("**"):
                style = S["cap"]
            else:
                style = S["body"]
            story.append(Paragraph(inline(text), style))
            for link in figs:
                story.extend([figure(REPORTS / link, 7 * cm), Spacer(1, 6)])
            para.clear()

    while i < len(lines):
        l = lines[i]
        if not l.strip():
            flush()
        elif l.startswith("### "):
            flush(); story.append(Paragraph(inline(l[4:]), S["h2"]))
        elif l.startswith("## "):
            flush(); story.append(Paragraph(inline(l[3:]), S["h1"]))
        elif l.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i]); i += 1
            story.extend(table(rows)); continue
        elif m := re.match(r"!\[[^\]]*\]\(([^)]+)\)", l):
            flush(); story.append(figure(REPORTS / m.group(1)))
            story.append(Spacer(1, 4))
        elif "<sub>" in l:
            flush(); story.extend([equation(), Spacer(1, 6)])
        elif re.match(r"(- |\d+\. )", l):
            flush()
            numbered = l[0].isdigit()
            items = []
            while i < len(lines) and re.match(r"(- |\d+\. )", lines[i]):
                items.append(re.sub(r"^(- |\d+\. )", "", lines[i])); i += 1
            story.extend(bullets(items, numbered)); continue
        else:
            para.append(l.strip())
        i += 1
    flush()
    return story


def title_page():
    rows = [[Paragraph(f"<b>{k}</b>", S["cell"]), Paragraph(v, S["cell"])] for k, v in STUDENT]
    t = Table(rows, colWidths=[5 * cm, 9 * cm])
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, RULE), ("LINEABOVE", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, ACCENT),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return [Spacer(1, 4.5 * cm),
            Paragraph("Paddy Disease Representation using a Convolutional Autoencoder", S["title"]),
            Spacer(1, 10), Paragraph("Project Report", S["sub"]), Spacer(1, 2.2 * cm),
            KeepTogether(t), PageBreak()]


def footer(canvas, doc):
    if doc.page == 1:
        return
    canvas.saveState()
    canvas.setFont("Body", 8); canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, FOOTER)
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(doc.page))
    canvas.restoreState()


def main():
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title="Paddy Disease Representation using a Convolutional Autoencoder",
                            author="T Manoj (2023004133)", subject="Project 29 report")
    story = title_page() + body_story((REPORTS / "report.md").read_text(encoding="utf-8"))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

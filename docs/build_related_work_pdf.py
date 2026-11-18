"""Render docs/related_work.md (+ .bib) to docs/related_work.pdf with reportlab.

Requires: pip install reportlab. Uses Windows fonts (Segoe UI, Segoe UI Symbol, Consolas).
Usage: python docs/build_related_work_pdf.py docs/related_work.md docs/related_work.bib docs/related_work.pdf
"""
import posixpath
import re
import sys
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, LongTable, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer, TableStyle)

MD, BIB, OUT = sys.argv[1:4]
REPO_URL = "https://github.com/AayushOjha001/ARGUS/blob/main/"
FONTS = "C:/Windows/Fonts/"

pdfmetrics.registerFont(TTFont("Body", FONTS + "segoeui.ttf"))
pdfmetrics.registerFont(TTFont("Body-Bold", FONTS + "segoeuib.ttf"))
pdfmetrics.registerFont(TTFont("Body-Italic", FONTS + "segoeuii.ttf"))
pdfmetrics.registerFont(TTFont("Body-BoldItalic", FONTS + "segoeuiz.ttf"))
pdfmetrics.registerFont(TTFont("Sym", FONTS + "seguisym.ttf"))
pdfmetrics.registerFont(TTFont("Mono", FONTS + "consola.ttf"))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-BoldItalic")

INK = colors.HexColor("#1f2933")
ACCENT = colors.HexColor("#1d4e89")
MUTED = colors.HexColor("#52606d")
RULE = colors.HexColor("#cbd2d9")
HEAD_BG = colors.HexColor("#e4ecf5")
ZEBRA = colors.HexColor("#f5f7fa")
ARGUS_BG = colors.HexColor("#fff4d6")

S = {
    "title": ParagraphStyle("title", fontName="Body-Bold", fontSize=19, leading=24, textColor=ACCENT, spaceAfter=4),
    "subtitle": ParagraphStyle("subtitle", fontName="Body", fontSize=9.5, leading=13, textColor=MUTED, spaceAfter=10),
    "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=14, leading=18, textColor=ACCENT, spaceBefore=12, spaceAfter=6),
    "h3": ParagraphStyle("h3", fontName="Body-Bold", fontSize=11.5, leading=15, textColor=INK, spaceBefore=9, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Body", fontSize=9.5, leading=13.5, textColor=INK, spaceAfter=6, alignment=TA_LEFT),
    "quote": ParagraphStyle("quote", fontName="Body", fontSize=9, leading=12.5, textColor=INK, backColor=colors.HexColor("#eef2f7"),
                            borderPadding=(6, 8, 6, 8), leftIndent=8, rightIndent=8, spaceBefore=4, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=7.6, leading=9.6, textColor=INK),
    "cellh": ParagraphStyle("cellh", fontName="Body-Bold", fontSize=7.8, leading=9.8, textColor=ACCENT),
    "ref": ParagraphStyle("ref", fontName="Body", fontSize=8, leading=10.6, textColor=INK, leftIndent=20, bulletIndent=0, spaceAfter=2.5, bulletFontName="Body", bulletFontSize=8),
}


# ------------------------------------------------------------------ bibliography
def parse_bib(text):
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", text, re.S):
        fields = {k.lower(): v.strip() for k, v in re.findall(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", m.group(3))}
        entries[m.group(2)] = fields
    return entries


def tex_clean(s):
    s = s.replace("{\\`e}", "è").replace("\\`e", "è").replace("--", "–")
    return re.sub(r"[{}]", "", s)


def format_authors(raw):
    names = [tex_clean(a.strip()) for a in raw.split(" and ")]
    out = []
    for n in names:
        if "," in n:
            last, first = [p.strip() for p in n.split(",", 1)]
            out.append(f"{first} {last}")
        else:
            out.append(n)
    if len(out) > 6:
        return ", ".join(out[:6]) + ", et al."
    return ", ".join(out)


BIBDB = parse_bib(open(BIB, encoding="utf-8").read())
CITE_ORDER = []


def cite_num(key):
    if key not in CITE_ORDER:
        CITE_ORDER.append(key)
    return CITE_ORDER.index(key) + 1


# ------------------------------------------------------------------ inline markdown
def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def resolve_link(url):
    if re.match(r"^[a-z]+://", url):
        return url
    path, _, frag = url.partition("#")
    path = posixpath.normpath(posixpath.join("docs", path)).lstrip("./")
    return REPO_URL + path + (("#" + frag) if frag else "")


def inline(text):
    stash = []

    def keep(html):
        stash.append(html)
        return f"\x00{len(stash) - 1}\x00"

    # inline code first, so nothing inside it is interpreted
    text = re.sub(r"`([^`]+)`", lambda m: keep(f'<font name="Mono">{esc(m.group(1))}</font>'), text)
    # links [label](url); label may itself contain stashed code
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                  lambda m: keep(f'<a href="{esc(resolve_link(m.group(2)))}" color="#1d4e89">{m.group(1)}</a>'), text)

    # citations [key] / [key1, key2]
    def cite(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        if not all(k in BIBDB for k in keys):
            return m.group(0)
        nums = ", ".join(str(cite_num(k)) for k in keys)
        return keep(f'<font color="#1d4e89">[{nums}]</font>')

    text = re.sub(r"\[([A-Za-z0-9_]+(?:\s*,\s*[A-Za-z0-9_]+)*)\]", cite, text)
    text = esc(text)
    text = re.sub(r"&lt;(/?)(sub|super|sup)&gt;", lambda m: f"<{m.group(1)}{'super' if m.group(2) != 'sub' else 'sub'}>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])", r"<i>\1</i>", text)
    text = text.replace("✓", '<font name="Sym">✓</font>')
    # restore stashed fragments (they may nest, so loop)
    while "\x00" in text:
        text = re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)
    return text


# ------------------------------------------------------------------ tables
def build_table(rows, width):
    header, body = rows[0], rows[1:]
    ncol = len(header)
    # weight columns by the longest text they hold
    lens = [max(len(r[i]) if i < len(r) else 0 for r in rows) for i in range(ncol)]
    weights = [min(max(l, 11), 90) ** 0.75 for l in lens]
    total = sum(weights)
    widths = [width * w / total for w in weights]
    data = [[Paragraph(inline(c), S["cellh"]) for c in header]]
    for r in body:
        r = r + [""] * (ncol - len(r))
        data.append([Paragraph(inline(c), S["cell"]) for c in r])
    t = LongTable(data, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for i, r in enumerate(body, start=1):
        first = r[0].replace("*", "").strip() if r else ""
        if first == "ARGUS" or ("ARGUS" in first and "lu2026argus" in first):
            style.append(("BACKGROUND", (0, i), (-1, i), ARGUS_BG))
        elif i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(style))
    return t


def split_row(line):
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return cells


# ------------------------------------------------------------------ document
PORTRAIT_W = A4[0] - 36 * mm
LAND_W = landscape(A4)[0] - 30 * mm


def on_page(canvas, doc):
    canvas.saveState()
    w, h = canvas._pagesize
    canvas.setFont("Body", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(15 * mm, 9 * mm, "ARGUS — Related Work and Comparison")
    canvas.drawRightString(w - 15 * mm, 9 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(15 * mm, 12.5 * mm, w - 15 * mm, 12.5 * mm)
    canvas.restoreState()


doc = BaseDocTemplate(OUT, pagesize=A4, title="ARGUS — Related Work and Comparison",
                      author="ARGUS embodied fork", subject="Multimodal prompt-injection defenses and attacks")
doc.addPageTemplates([
    PageTemplate(id="portrait", pagesize=A4, onPage=on_page,
                 frames=[Frame(18 * mm, 18 * mm, PORTRAIT_W, A4[1] - 36 * mm, id="p")]),
    PageTemplate(id="landscape", pagesize=landscape(A4), onPage=on_page,
                 frames=[Frame(15 * mm, 17 * mm, LAND_W, landscape(A4)[1] - 32 * mm, id="l")]),
])

lines = open(MD, encoding="utf-8").read().splitlines()
story = []
para, quote, table, lst = [], [], [], []
title_done = False
pending_heading = None  # heading kept together with a following table
last_heading_idx = None  # story index of the most recent heading


def flush_para():
    if para:
        story.append(Paragraph(inline(" ".join(para)), S["body"]))
        para.clear()


def flush_quote():
    if quote:
        story.append(Paragraph(inline(" ".join(quote)), S["quote"]))
        quote.clear()


def flush_list():
    if not lst:
        return
    for depth, marker, text in lst:
        indent = 12 + depth * 14
        style = ParagraphStyle("li", parent=S["body"], leftIndent=indent, firstLineIndent=0, spaceAfter=2.5,
                               bulletIndent=indent - 10, bulletFontName="Body")
        bullet = marker if marker != "-" else ("•" if depth == 0 else "–")
        story.append(Paragraph(inline(text), style, bulletText=bullet))
    story.append(Spacer(1, 4))
    lst.clear()


def flush_table():
    global pending_heading, last_heading_idx
    if not table:
        return
    rows = [split_row(l) for l in table if not re.match(r"^\s*\|?\s*:?-{2,}", l)]
    # landscape only when the table is genuinely wide (many columns of long text)
    avg_row_chars = sum(len(" ".join(r)) for r in rows) / len(rows)
    wide = len(rows[0]) >= 5 and avg_row_chars > 120
    flow = build_table(rows, LAND_W if wide else PORTRAIT_W)
    if wide:
        # move the section heading and its short intro onto the landscape page
        carried = []
        if pending_heading is None and last_heading_idx is not None and len(story) - last_heading_idx <= 4:
            carried = story[last_heading_idx:]
            del story[last_heading_idx:]
        story.append(NextPageTemplate("landscape"))
        story.append(PageBreak())
        if pending_heading is not None:
            story.append(pending_heading)
        story.extend(carried)
        story.append(flow)
        story.append(NextPageTemplate("portrait"))
        story.append(PageBreak())
    else:
        if pending_heading is not None:
            story.append(KeepTogether([pending_heading, Spacer(1, 2)]))
        story.append(flow)
        story.append(Spacer(1, 8))
    pending_heading = None
    last_heading_idx = None
    table.clear()


def flush_all():
    flush_para(); flush_quote(); flush_list(); flush_table()


i = 0
while i < len(lines):
    line = lines[i]
    s = line.strip()
    if s.startswith("|"):
        flush_para(); flush_quote(); flush_list()
        table.append(s)
    elif table:
        flush_table()
        continue
    elif not s:
        flush_para(); flush_quote(); flush_list()
    elif s == "---":
        flush_all()
        story.append(Spacer(1, 4))
    elif s.startswith("# ") and not title_done:
        flush_all()
        story.append(Paragraph(inline(s[2:]), S["title"]))
        story.append(Paragraph(f"Compiled {date.today():%d %B %Y} · {len(BIBDB)} papers · numbered references at the end",
                               S["subtitle"]))
        title_done = True
    elif s.startswith("## ") or s.startswith("### "):
        flush_all()
        level = "h2" if s.startswith("## ") else "h3"
        heading = Paragraph(inline(s.lstrip("#").strip()), S[level])
        # look ahead: is the next non-empty line a table?
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines) and lines[j].strip().startswith("|"):
            pending_heading = heading
        else:
            last_heading_idx = len(story)
            story.append(heading)
    elif s.startswith(">"):
        flush_para(); flush_list()
        quote.append(s.lstrip(">").strip())
    elif re.match(r"^(\s*)(-|\d+\.)\s+", line):
        flush_para(); flush_quote()
        m = re.match(r"^(\s*)(-|\d+\.)\s+(.*)", line)
        lst.append([len(m.group(1)) // 2, m.group(2), m.group(3)])
    elif lst and line.startswith(" "):
        lst[-1][2] += " " + s  # continuation of a list item
    else:
        flush_quote(); flush_list()
        para.append(s)
    i += 1
flush_all()

# references
story.append(Paragraph("References", S["h2"]))
for n, key in enumerate(CITE_ORDER, start=1):
    e = BIBDB[key]
    parts = [format_authors(e.get("author", "")).rstrip(".") + "."] if e.get("author") else []
    t = esc(tex_clean(e.get('title', key)))
    parts.append(f"<i>{t}</i>" + ("" if t[-1] in "?!." else "."))
    venue = e.get("booktitle") or e.get("journal")
    if venue:
        parts.append(esc(tex_clean(venue)) + ",")
    parts.append(e.get("year", "") + ".")
    if e.get("eprint"):
        url = f"https://arxiv.org/abs/{e['eprint']}"
        parts.append(f'<a href="{url}" color="#1d4e89">arXiv:{e["eprint"]}</a>.')
    elif e.get("url"):
        parts.append(f'<a href="{esc(e["url"])}" color="#1d4e89">{esc(e["url"])}</a>.')
    if e.get("note"):
        parts.append(esc(tex_clean(e["note"])) + ".")
    story.append(Paragraph(" ".join(parts), S["ref"], bulletText=f"[{n}]"))

uncited = [k for k in BIBDB if k not in CITE_ORDER]
if uncited:
    print("bib entries not cited in the text:", uncited)

doc.build(story)
print("wrote", OUT, "with", len(CITE_ORDER), "references")

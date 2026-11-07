import re, copy
from docx import Document
from docx.shared import Pt, Cm, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from omml import omml

FONT = "Times New Roman"
TOKEN = re.compile(r"(\*\*.+?\*\*|\*.+?\*|\$.+?\$|\[\[.+?\]\])", re.S)


def new_doc():
    d = Document()
    s = d.sections[0]
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.top_margin = s.bottom_margin = Cm(2.0)
    s.left_margin, s.right_margin = Cm(2.26), Cm(2.08)
    st = d.styles["Normal"]
    st.font.name = FONT; st.font.size = Pt(12)
    rpr = st.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), FONT)
    pf = st.paragraph_format
    pf.space_before = pf.space_after = Pt(0)
    pf.line_spacing = 1.25
    return d


def _run(p, text, size=None, bold=None, italic=None, highlight=False):
    r = p.add_run(text)
    r.font.name = FONT
    r._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    if size: r.font.size = Pt(size)
    if bold is not None: r.bold = bold
    if italic is not None: r.italic = italic
    if highlight: r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    return r


def rich(p, text, size=None, bold=None):
    """Mini-markup: **bold**, *italic*, $latex$ (editable equation), [[placeholder]] (yellow highlight)."""
    for tok in TOKEN.split(text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            _run(p, tok[2:-2], size, True)
        elif tok.startswith("$") and tok.endswith("$") and len(tok) > 2:
            p._p.append(omml(tok[1:-1]))
        elif tok.startswith("[[") and tok.endswith("]]"):
            _run(p, "[" + tok[2:-2] + "]", size, bold, highlight=True)
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            _run(p, tok[1:-1], size, bold, True)
        else:
            _run(p, tok, size, bold)
    return p


def title(d, text):
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(12)
    _run(p, text, 16, True)


def author_line(d, text):
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(6)
    rich(p, text)


def heading(d, text, level=1):
    p = d.add_paragraph()
    p.paragraph_format.space_before = Pt(10 if level == 1 else 6)
    p.paragraph_format.keep_with_next = True
    _run(p, text, 12, True)
    return p


def body(d, text, indent=True):
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if indent: p.paragraph_format.first_line_indent = Pt(12)
    rich(p, text)
    return p


def label_para(d, label, text):
    """'Keywords: ...' / 'Abstract' style paragraphs."""
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(6)
    _run(p, label, None, True)
    rich(p, text)
    return p


def equation(d, latex, num):
    p = d.add_paragraph()
    p.paragraph_format.space_before = Pt(4); p.paragraph_format.space_after = Pt(4)
    width = d.sections[0].page_width - d.sections[0].left_margin - d.sections[0].right_margin
    ts = p.paragraph_format.tab_stops
    ts.add_tab_stop(Emu(int(width / 2)), WD_TAB_ALIGNMENT.CENTER)
    ts.add_tab_stop(Emu(int(width)), WD_TAB_ALIGNMENT.RIGHT)
    _run(p, "\t")
    p._p.append(omml(latex))
    _run(p, f"\t({num})")
    return p


def caption(d, text, above=True):
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6 if above else 2)
    p.paragraph_format.space_after = Pt(3 if above else 8)
    p.paragraph_format.keep_with_next = above
    rich(p, text, 10.5, True)
    return p


def note(d, text, size=9.5):
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.1
    rich(p, text, size)
    return p


def _borders(cell, sz=4):
    tcPr = cell._tc.get_or_add_tcPr()
    b = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(sz)); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
        b.append(e)
    tcPr.append(b)


def table(d, rows, widths_cm, size=9, header_rows=1, bold_first_col=False, merge_runs=None, align_first="left", keep=False):
    t = d.add_table(rows=len(rows), cols=len(rows[0]))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
    for i, row in enumerate(rows):
        tr = t.rows[i]
        trPr = tr._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit"); trPr.append(cs)
        if i < header_rows:
            th = OxmlElement("w:tblHeader"); trPr.append(th)
        for j, val in enumerate(row):
            c = tr.cells[j]
            c.width = Cm(widths_cm[j])
            _borders(c)
            c.vertical_alignment = 1
            p = c.paragraphs[0]
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_before = Pt(1); p.paragraph_format.space_after = Pt(1)
            if keep and i < len(rows) - 1: p.paragraph_format.keep_with_next = True
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if (j == 0 and align_first == "left") else WD_ALIGN_PARAGRAPH.CENTER
            rich(p, str(val), size, bold=True if (i < header_rows or (bold_first_col and j == 0)) else None)
    # merges: list of (row, col_start, col_end)
    for (r, c0, c1) in (merge_runs or []):
        a = t.cell(r, c0); b = t.cell(r, c1)
        txt = a.text
        m = a.merge(b)
        for extra in m.paragraphs[1:]:
            extra._element.getparent().remove(extra._element)
        m.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    return t


def figure(d, png, width_cm):
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6); p.paragraph_format.keep_with_next = True
    p.paragraph_format.line_spacing = 1.0
    p.add_run().add_picture(png, width=Cm(width_cm))
    return p


def reference(d, text):
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Pt(24); p.paragraph_format.first_line_indent = Pt(-24)
    rich(p, text)
    return p

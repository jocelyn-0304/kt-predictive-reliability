import subprocess, tempfile, zipfile, re, os
from lxml import etree
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_cache = {}

def omml(latex):
    """LaTeX -> editable OMML element (m:oMath) via pandoc."""
    if latex in _cache:
        return etree.fromstring(_cache[latex])
    with tempfile.TemporaryDirectory() as d:
        md = os.path.join(d, "e.md"); out = os.path.join(d, "e.docx")
        open(md, "w", encoding="utf8").write(f"${latex}$\n")
        subprocess.run(["pandoc", md, "-o", out], check=True)
        xml = zipfile.ZipFile(out).read("word/document.xml")
    root = etree.fromstring(xml)
    el = root.find(".//{%s}oMath" % M)
    _cache[latex] = etree.tostring(el)
    return el

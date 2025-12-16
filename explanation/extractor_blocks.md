# Text Extraction and Preprocessing Module — Block-by-Block

Module: `modules/extractor.py` — Text Extraction and Preprocessing Module

This document separates `modules/extractor.py` into logical blocks. For each
block we give a plain-language explanation, state the type of algorithm or
tool used, and show the relevant code snippet.

---

BLOCK: TXT Handler

What this block does:
- Reads uploaded `.txt` files and decodes them into a Unicode string.
- Simple, deterministic decoding with a UTF-8 primary attempt and a
  Latin-1 fallback for legacy encodings.

Model / algorithm used:
- None (straight text decoding); rule-based I/O logic.

Code:
```python
if name.endswith('.txt'):
    try:
        return data.decode('utf-8')
    except Exception:
        return data.decode('latin-1', errors='ignore')
```

PROCESS 1: Read binary bytes from uploaded file.
PROCESS 2: Try UTF-8 decode, otherwise fallback to Latin-1.

---

BLOCK: PDF primary conversion (pdf2docx → python-docx)

What this block does:
- Converts incoming PDF into a DOCX using `pdf2docx.Converter`, then
  parses the resulting DOCX with `python-docx` to extract paragraphs and
  table cell contents in document order.
- This produces structured text that is easier to normalize downstream.

Model / algorithm used:
- External conversion tool (`pdf2docx`) + rule-based DOM traversal using
  `python-docx`. No learned model here — a deterministic conversion + parse.

Code (excerpt):
```python
from pdf2docx import Converter
from docx import Document

with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
    tmp_pdf.write(data)
    pdf_path = tmp_pdf.name

with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_docx:
    docx_path = tmp_docx.name

cv = Converter(pdf_path)
cv.convert(docx_path, start=0, end=None)
cv.close()

doc = Document(docx_path)
text_parts = []
for element in doc.element.body:
    if isinstance(element, CT_P):
        para = Paragraph(element, doc)
        para_text = para.text.strip()
        if para_text:
            text_parts.append(para_text.strip())
    elif isinstance(element, CT_Tbl):
        table = Table(element, doc)
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    cell_text = para.text.strip()
                    if cell_text:
                        text_parts.append(cell_text.strip())

result = '\n'.join(text_parts)
```

PROCESS 1: Write PDF to a temp file and run `pdf2docx` conversion.
PROCESS 2: Load the converted DOCX with `python-docx` and extract text
           elements in document order (paragraphs and table cells).

---

BLOCK: PDF fallback — Docling (structure-preserving conversion)

What this block does:
- If the primary PDF→DOCX conversion fails, attempt an AI/structure-aware
  conversion using `docling.document_converter.DocumentConverter`, then
  export the result to Markdown and perform light cleaning.

Model / algorithm used:
- `docling` is a third-party converter that applies layout-aware heuristics
  (and optionally ML components) to preserve structure — treated here as a
  black-box converter. The code that follows is rule-based cleaning.

Code (excerpt):
```python
from docling.document_converter import DocumentConverter

with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
    tmp.write(data)
    tmp_path = tmp.name

converter = DocumentConverter()
result = converter.convert(tmp_path)
md_text = result.document.export_to_markdown()

# Clean markdown formatting but preserve structure
md_text = re.sub(r'^#+\s+', '', md_text, flags=re.MULTILINE)
md_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', md_text)
md_text = re.sub(r'\*([^*]+)\*', r'\1', md_text)
md_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', md_text)
md_text = re.sub(r'\n{3,}', '\n\n', md_text)
```

PROCESS 1: Create temp PDF file and pass to Docling converter.
PROCESS 2: Export to Markdown and run deterministic regex cleaning.

---

BLOCK: PDF final fallback — pdfminer

What this block does:
- If both converters fail, use `pdfminer.six` to extract raw text from the
  PDF stream as a last-resort fallback.

Model / algorithm used:
- `pdfminer` provides rule-based text extraction (layout heuristics), not a
  learned semantic model. This is a low-level, robust fallback.

Code (excerpt):
```python
from pdfminer.high_level import extract_text as extract_pdf_text
return extract_pdf_text(io.BytesIO(data))
```

PROCESS 1: Feed the PDF bytes to pdfminer's high-level extractor.
PROCESS 2: Return raw text for downstream normalization.

---

BLOCK: DOCX primary extraction (`python-docx`)

What this block does:
- For `.docx` files, parse the document element body in order and extract
  paragraphs and table cell text. Join parts while preserving readable
  spacing structure.

Model / algorithm used:
- Deterministic DOM traversal via `python-docx` (rule-based extraction).

Code (excerpt):
```python
from docx import Document

with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
    tmp.write(data)
    tmp_path = tmp.name
doc = Document(tmp_path)
text_parts = []
for element in doc.element.body:
    if isinstance(element, CT_P):
        para = Paragraph(element, doc)
        para_text = para.text
        if para_text.strip():
            text_parts.append(para_text)
    elif isinstance(element, CT_Tbl):
        table = Table(element, doc)
        for row in table.rows:
            for cell in row.cells:
                cell_text = cell.text.strip()
                if cell_text:
                    text_parts.append(cell_text)

result = '\n\n'.join(text_parts)
```

PROCESS 1: Write DOCX to temp and open with `python-docx`.
PROCESS 2: Iterate elements and collect paragraph and table text in order.

---

BLOCK: DOCX fallback (`docx2txt`)

What this block does:
- If `python-docx` parsing fails, fall back to `docx2txt.process()` which is
  a simpler extraction tool that often recovers text in more cases.

Model / algorithm used:
- Rule-based extraction via `docx2txt` (file-to-text utility).

Code (excerpt):
```python
import docx2txt
with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
    tmp.write(data)
    tmp_path = tmp.name
return docx2txt.process(tmp_path)
```

PROCESS 1: Save to temp file and call `docx2txt.process()`.
PROCESS 2: Return the processed plain text.

---

BLOCK: Normalization pipeline (`normalize_resume_text`)

What this block does (plain language):
- Clean and normalize messy extracted text so downstream NLP gets stable
  inputs: remove artifacts, fix hyphenation, normalize bullets, merge
  broken sentence lines, merge company/title/location blocks, and group
  short skill headers with their lists.

Model / algorithm used:
- Purely rule-based text processing using regular expressions and heuristic
  grouping logic. No learned model here — deterministic normalization.

Key code excerpts and processes:
```python
# Remove font change markers and normalize line endings
text = raw.replace("[FONT_CHANGE]", "")
text = text.replace("\r\n", "\n").replace("\r", "\n")

# Fix hyphenated words split across lines
text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

# Normalize bullets to a single marker
text = re.sub(r'^[\-●○▪▫◦⦿⦾]\s+', '• ', text, flags=re.MULTILINE)
text = re.sub(r'^\d+[\.)]\s+', '• ', text, flags=re.MULTILINE)

# Merge continuation lines where previous line does not end with punctuation
if (merged_continuation_lines and
    not re.search(r'[.!?:]$', merged_continuation_lines[-1].strip()) and
    not re.match(r'^[•]\s', line_stripped)):
    merged_continuation_lines[-1] = merged_continuation_lines[-1].rstrip() + ' ' + line_stripped

# Group section headers with their content (skill lists)
def is_section_header(line):
    if line.endswith(':'):
        return True
    if len(words) >= 2 and len(words) <= 5 and line.isupper():
        return True
    # Short Title Case heuristic
    ...
```

PROCESS 1: Clean artifact characters and whitespace.
PROCESS 2: Repair line breaks and hyphenation artifacts.
PROCESS 3: Normalize bullets and numbered lists.
PROCESS 4: Merge related lines (percentages, company blocks, skill lists).

---

BLOCK: Temp-file and error handling hygiene

What this block does:
- Uses `tempfile.NamedTemporaryFile` for intermediate files and ensures
  cleanup with `try/finally` and `os.remove()` calls so the system does not
  leak temporary files.

Model / algorithm used:
- Defensive I/O programming / resource cleanup patterns.

Code (excerpt):
```python
with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
    tmp_pdf.write(data)
    pdf_path = tmp_pdf.name
try:
    # conversion and parsing
    ...
finally:
    try:
        os.remove(pdf_path)
        os.remove(docx_path)
    except Exception:
        pass
```

PROCESS 1: Create temp files for converters.
PROCESS 2: Ensure removal in finally block regardless of success.

---

---

## Complexity & Performance

Definitions:
- $f$ = input file size (bytes)
- $c$ ≈ number of characters after extraction (proportional to $f$)
- $p$ = number of pages (for PDFs)
- $m$ = number of deterministic normalization passes (constant, e.g. ~5)

Time complexity (dominant costs: I/O + PDF→DOCX conversion + parsing + regex passes):
- Conversion / I/O: $T_{\text{io/conv}} = \Theta(f)$
- Parsing DOCX / iterating elements: $T_{\text{parse}} = \Theta(c)$
- Normalization (m regex/merge passes): $T_{\text{norm}} = m\cdot\Theta(c) = \Theta(c)$

Overall (combine linear terms):
\[ T_{\text{total}} = \Theta(f) + \Theta(c) = \Theta(f) \]
(since $c$ is proportional to $f$, the whole pipeline is linear in file size)

Space complexity:
- Peak in-memory text: $S_{\text{text}} = \Theta(c) = \Theta(f)$
- Temporary disk for converters: $S_{\text{temp}} = \Theta(f)$
- Auxiliary overhead (buffers, regex state): $O(1)$

Practical notes (short):
- The real bottleneck is external conversion (PDF→DOCX) and disk I/O; both are IO-bound and dominated by $\Theta(f)$.
- Each normalization pass is linear; keeping $m$ constant preserves linear time overall.
- For very large files, avoid full in-memory joins or use streaming extraction to reduce peak memory and latency.


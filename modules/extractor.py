"""Text extraction from various file formats"""
import io
import tempfile
import os
import re
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import Table
from docx.text.paragraph import Paragraph


def normalize_resume_text(raw: str) -> str:
    """Smart normalization that groups related résumé sections.
    - Normalizes all bullet types to consistent format
    - Merges section headers with their content (e.g., "Skills:" + skill list)
    - Keeps comma-separated lists together (do NOT split on commas)
    - Merges standalone bullets with their text
    - Merges percentages/grades with their course/item names
    - Merges company/position/location blocks into single lines
    - Preserves overall structure with proper section boundaries
    - Aggressive whitespace normalization for PDF consistency
    """
    if not raw:
        return ""
    
    # Remove font change markers from PDF extraction
    text = raw.replace("[FONT_CHANGE]", "")
    
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Aggressive whitespace normalization for PDF artifacts
    # Remove multiple spaces within lines
    text = re.sub(r'[ \t]{2,}', ' ', text)
    
    # Remove spaces before punctuation (common PDF artifact)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    # Fix hyphenated words split across lines (common in PDFs)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # FIRST: Normalize all bullet types to a single consistent format
    # Replace all bullet variants with standard bullet •
    text = re.sub(r'^[\-●○▪▫◦⦿⦾]\s+', '• ', text, flags=re.MULTILINE)
    # Also handle numbered lists (1. 2. etc.)
    text = re.sub(r'^\d+[\.)]\s+', '• ', text, flags=re.MULTILINE)
    
    # Preserve bullets by spacing them if stuck to text
    text = re.sub(r"([\-•●○▪▫◦⦿⦾])(?=\S)", r"\1 ", text)
    
    # First pass: merge continuation lines (lines that are part of same sentence)
    # This fixes awkward splits like "developed through degree,\nalongside the ability..."
    lines = text.split('\n')
    merged_continuation_lines = []
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        if not line_stripped:
            merged_continuation_lines.append(line)
            continue
        
        # Check if this should be merged with previous line
        # Merge if: previous line exists, previous doesn't end with sentence punctuation,
        # current line doesn't start with bullet, and previous line isn't a header (ends with :)
        # Docling handles structure well, so we don't need font change markers
        if (merged_continuation_lines and 
            merged_continuation_lines[-1].strip() and
            not re.search(r'[.!?:]$', merged_continuation_lines[-1].strip()) and
            not re.match(r'^[•]\s', line_stripped)):
            # This is a continuation line - merge with previous
            merged_continuation_lines[-1] = merged_continuation_lines[-1].rstrip() + ' ' + line_stripped
        else:
            # New line
            merged_continuation_lines.append(line)
    
    # Second pass: merge percentages with previous lines
    temp_lines = []
    
    for i, line in enumerate(merged_continuation_lines):
        line_stripped = line.strip()
        
        # If this line is just a percentage, merge it with the previous non-empty line
        if re.match(r'^\d{1,3}%$', line_stripped) and temp_lines:
            # Find last non-empty line and append percentage
            for j in range(len(temp_lines) - 1, -1, -1):
                if temp_lines[j].strip():
                    temp_lines[j] = f"{temp_lines[j]}\t{line_stripped}"
                    break
        else:
            temp_lines.append(line)
    
    # Third pass: merge standalone bullets with following text
    merged_lines = []
    i = 0
    
    while i < len(temp_lines):
        line = temp_lines[i].strip()
        
        # Skip empty lines
        if not line:
            merged_lines.append(temp_lines[i])
            i += 1
            continue
        
        # Check if this is a standalone bullet (bullet only, no text)
        if re.match(r'^[-•●○▪▫◦⦿⦾]$', line) and i + 1 < len(temp_lines):
            next_line = temp_lines[i + 1].strip()
            if next_line:
                merged_lines.append(f"{line} {next_line}")
                i += 2
                continue
        
        # Normal line - just add it
        merged_lines.append(temp_lines[i])
        i += 1
    
    # Fourth pass: merge company/position/location blocks
    job_titles = {'engineer', 'developer', 'manager', 'analyst', 'scientist', 'designer',
                  'consultant', 'director', 'coordinator', 'specialist', 'administrator',
                  'architect', 'lead', 'senior', 'junior', 'intern', 'assistant'}
    
    final_lines = []
    i = 0
    
    while i < len(merged_lines):
        line = merged_lines[i].strip()
        
        if not line:
            final_lines.append(merged_lines[i])
            i += 1
            continue
        
        # Check if this looks like start of company block
        if (not re.match(r'^[-•●○▪▫◦⦿⦾]', line) and 
            not line.endswith(':') and
            len(line.split()) >= 2 and
            i + 1 < len(merged_lines)):
            
            next_line = merged_lines[i + 1].strip()
            if next_line and any(title in next_line.lower() for title in job_titles):
                block_parts = [line, next_line]
                j = i + 2
                
                if j < len(merged_lines):
                    third_line = merged_lines[j].strip()
                    if (third_line and 
                        len(third_line.split()) <= 4 and
                        not re.match(r'^[-•●○▪▫◦⦿⦾]', third_line) and
                        not third_line.endswith(':')):
                        block_parts.append(third_line)
                        j += 1
                
                final_lines.append(' — '.join(block_parts))
                i = j
                continue
        
        final_lines.append(merged_lines[i])
        i += 1
    
    # Fifth pass: GROUP SECTION HEADERS WITH THEIR CONTENT
    # This keeps skill lists together instead of splitting them
    grouped_lines = []
    i = 0
    
    def is_section_header(line):
        """Detect section headers: ends with colon, ALL CAPS, or Title Case without commas"""
        if not line:
            return False
        # Ends with colon - strong indicator
        if line.endswith(':'):
            return True
        words = line.split()
        # ALL CAPS (at least 2 words, not too long)
        if len(words) >= 2 and len(words) <= 5 and line.isupper():
            return True
        # Short Title Case without commas or bullets
        if (',' not in line and 
            len(words) <= 4 and 
            len(words) >= 2 and
            not re.match(r'^[-•●○▪▫◦⦿⦾]', line)):
            # Check if it's title case
            title_case = all(w[0].isupper() if w and w[0].isalpha() else True for w in words)
            # Must not look like a sentence (no ending punctuation)
            no_sentence_end = not re.search(r'[.!?]$', line)
            if title_case and no_sentence_end:
                return True
        return False
    
    def is_bullet_line(line):
        """Check if line starts with bullet marker"""
        return bool(re.match(r'^[-•●○▪▫◦⦿⦾]\s', line))
    
    def is_skill_content(line):
        """Check if line looks like comma-separated skills or technical content"""
        # Has multiple commas (likely a list)
        if line.count(',') >= 2:
            return True
        # Has technical terms or version numbers
        if re.search(r'\.NET|[A-Z]{2,}|v?\d+\.\d+|Windows|Linux|SQL|API', line):
            return True
        return False
    
    while i < len(final_lines):
        line = final_lines[i].strip()
        
        # Empty lines are boundaries
        if not line:
            grouped_lines.append(final_lines[i])
            i += 1
            continue
        
        # Bullets should NEVER be grouped - each is completely separate
        # This is critical: prevents multiple bullets from merging
        if is_bullet_line(line):
            grouped_lines.append(final_lines[i])
            i += 1
            continue
        
        # Check if this is a section header
        if is_section_header(line):
            # Collect header + skill content lines only
            block = [line]
            j = i + 1
            lines_collected = 0
            
            while j < len(final_lines) and lines_collected < 3:
                next_line = final_lines[j].strip()
                
                # Stop at boundaries
                if not next_line:  # Empty line
                    break
                if is_section_header(next_line):  # New header
                    break
                if is_bullet_line(next_line):  # Bullet starts new item - NEVER group with header
                    break
                
                # Only group if it's skill content (comma-separated lists), not narrative text
                if is_skill_content(next_line):
                    block.append(next_line)
                    j += 1
                    lines_collected += 1
                else:
                    # This is narrative content or another resume item, don't group
                    break
            
            # If we only collected the header (no skill content), just add it alone
            if len(block) == 1:
                grouped_lines.append(final_lines[i])
                i += 1
            else:
                # Merge the skill block into one line
                grouped_lines.append('\n'.join(block))
                i = j
            continue
        
        # Regular line (not a header, not a bullet) - keep separate
        grouped_lines.append(final_lines[i])
        i += 1
    
    text = '\n'.join(grouped_lines)
    
    # Collapse excessive spaces but keep newlines and tabs
    text = re.sub(r"[ ]{2,}", " ", text)
    
    # Remove trailing spaces on lines (but keep tabs)
    text = re.sub(r"[ ]+\n", "\n", text)
    
    return text


def extract_from_file(uploaded_file) -> str:
    """Extract text from uploaded PDF, DOCX, or TXT file with structure preservation"""
    try:
        data = uploaded_file.read()
        name = uploaded_file.name.lower()
        
        if name.endswith('.txt'):
            try:
                return data.decode('utf-8')
            except Exception:
                return data.decode('latin-1', errors='ignore')
        
        elif name.endswith('.pdf'):
            # Convert PDF to DOCX first for consistent extraction
            try:
                from pdf2docx import Converter
                from docx import Document
                
                # Create temporary files for PDF and DOCX
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                    tmp_pdf.write(data)
                    pdf_path = tmp_pdf.name
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_docx:
                    docx_path = tmp_docx.name
                
                try:
                    # Convert PDF to DOCX
                    cv = Converter(pdf_path)
                    cv.convert(docx_path, start=0, end=None)
                    cv.close()
                    
                    # Now extract from the DOCX (same as DOCX extraction below)
                    doc = Document(docx_path)
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
                    return result
                finally:
                    # Clean up temp files
                    try:
                        os.remove(pdf_path)
                        os.remove(docx_path)
                    except Exception:
                        pass
                        
            except Exception:
                # Fallback to original PDF extraction if conversion fails
                pass
        
        # Original PDF extraction fallback
        if name.endswith('.pdf'):
            try:
                # Try Docling first - AI-powered structure preservation
                from docling.document_converter import DocumentConverter
                import re
                
                # Create temporary file for Docling
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                
                try:
                    converter = DocumentConverter()
                    result = converter.convert(tmp_path)
                    
                    # Export as markdown which preserves structure naturally
                    md_text = result.document.export_to_markdown()
                    
                    # Clean up markdown formatting but preserve structure and spacing
                    # Remove markdown headers (# ## ###) but keep the text
                    md_text = re.sub(r'^#+\s+', '', md_text, flags=re.MULTILINE)
                    # Remove markdown bold (**text**)
                    md_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', md_text)
                    # Remove markdown italic (*text*)
                    md_text = re.sub(r'\*([^*]+)\*', r'\1', md_text)
                    # Remove markdown links [text](url)
                    md_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', md_text)
                    # Clean up excessive blank lines (more than 2 consecutive)
                    md_text = re.sub(r'\n{3,}', '\n\n', md_text)
                    
                    return md_text.strip()
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                        
            except Exception:
                # Fallback to PyMuPDF with dict mode
                try:
                    import fitz
                    import re
                    doc = fitz.open(stream=data, filetype="pdf")
                    lines = []
                    prev_font_size = None
                    
                    for page in doc:
                        text_dict = page.get_text("dict")
                        for block in text_dict.get("blocks", []):
                            if block.get("type") == 0:  # text block
                                for line in block.get("lines", []):
                                    line_text = ""
                                    font_size = None
                                    
                                    for span in line.get("spans", []):
                                        line_text += span.get("text", "")
                                        if font_size is None:
                                            font_size = span.get("size", 0)
                                    
                                    line_text = line_text.strip()
                                    if line_text:
                                        # Add blank line for visual spacing in certain cases
                                        if lines:
                                            is_bullet = re.match(r'^[-•●○▪▫◦⦿⦾]\s', line_text)
                                            # Add spacing before bullets or when font size changes significantly
                                            if is_bullet or (prev_font_size and font_size and abs(font_size - prev_font_size) > 1.5):
                                                lines.append("")
                                                # Add special marker to prevent continuation merging across font changes
                                                if prev_font_size and font_size and abs(font_size - prev_font_size) > 1.5:
                                                    line_text = "[FONT_CHANGE]" + line_text
                                        
                                        lines.append(line_text)
                                        prev_font_size = font_size
                    
                    doc.close()
                    return '\n'.join(lines)
                except Exception:
                    # Final fallback to pdfminer
                    try:
                        from pdfminer.high_level import extract_text as extract_pdf_text
                        return extract_pdf_text(io.BytesIO(data))
                    except Exception as e:
                        raise Exception(f"PDF extraction failed: {e}")
        
        elif name.endswith('.docx'):
            try:
                # Use python-docx but mimic docx2txt's extraction behavior
                from docx import Document
                from docx.oxml.text.paragraph import CT_P
                from docx.oxml.table import CT_Tbl
                from docx.table import Table
                from docx.text.paragraph import Paragraph
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                try:
                    doc = Document(tmp_path)
                    text_parts = []
                    
                    # Extract text from all elements in document order (paragraphs and tables)
                    # This matches docx2txt behavior of capturing everything
                    for element in doc.element.body:
                        if isinstance(element, CT_P):
                            # Paragraph
                            para = Paragraph(element, doc)
                            para_text = para.text
                            if para_text.strip():
                                text_parts.append(para_text)
                        elif isinstance(element, CT_Tbl):
                            # Table - extract all cells
                            table = Table(element, doc)
                            for row in table.rows:
                                for cell in row.cells:
                                    cell_text = cell.text.strip()
                                    if cell_text:
                                        text_parts.append(cell_text)
                    
                    # Join with double newlines like docx2txt does
                    result = '\n\n'.join(text_parts)
                    return result
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
            except Exception:
                # Fallback to docx2txt
                try:
                    import docx2txt
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                        tmp.write(data)
                        tmp_path = tmp.name
                    try:
                        return docx2txt.process(tmp_path)
                    finally:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                except Exception as e:
                    raise Exception(f"DOCX extraction failed: {e}")
        
        else:
            raise Exception(f"Unsupported file type: {name}")
    
    except Exception as e:
        raise Exception(f"Error extracting text: {e}")

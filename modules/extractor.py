"""Text extraction from various file formats"""
import io
import tempfile
import os
import re


def normalize_resume_text(raw: str) -> str:
    """Light normalization that preserves bullets and line breaks.
    - Collapse runs of spaces
    - Normalize Windows/Mac line endings
    - Keep bullets and common list markers intact
    """
    if not raw:
        return ""
    # Normalize line endings
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    
    # Preserve bullets by spacing them if stuck to text
    text = re.sub(r"([\-•●○▪▫◦⦿⦾])(?=\S)", r"\1 ", text)
    
    # Collapse excessive spaces but keep newlines
    text = re.sub(r"[ \t]{2,}", " ", text)
    
    # Remove trailing spaces on lines
    text = re.sub(r"[ \t]+\n", "\n", text)
    
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
            try:
                # Try PyMuPDF with "blocks" to preserve line structure
                import fitz
                doc = fitz.open(stream=data, filetype="pdf")
                text = ""
                for page in doc:
                    # Get text blocks which preserve layout better
                    blocks = page.get_text("blocks")
                    for block in blocks:
                        if len(block) >= 5:
                            block_text = block[4].strip()
                            if block_text:
                                # Each block is a complete unit (bullet point, paragraph, etc.)
                                text += block_text + "\n\n"
                doc.close()
                return text
            except Exception:
                # Fallback to pdfminer
                try:
                    from pdfminer.high_level import extract_text as extract_pdf_text
                    return extract_pdf_text(io.BytesIO(data))
                except Exception as e:
                    raise Exception(f"PDF extraction failed: {e}")
        
        elif name.endswith('.docx'):
            try:
                # Use python-docx for better paragraph structure
                from docx import Document
                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                try:
                    doc = Document(tmp_path)
                    paragraphs = []
                    for para in doc.paragraphs:
                        text = para.text.strip()
                        if text:
                            paragraphs.append(text)
                    return '\n\n'.join(paragraphs)  # Use double newline for paragraph separation
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

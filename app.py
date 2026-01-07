"""
Text Summarizer Pro - Professional Edition
Developed by Mohammed Moota

Architecture:
- OCR: EasyOCR (fast, accurate, offline)
- Summarization: Sumy LSA (semantic analysis, not TF-IDF)
- PDF: pdfplumber + OCR fallback for scanned documents
- Sentiment: TextBlob
"""

import os
import re
import io
import tempfile
from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename
from textblob import TextBlob

# PDF Processing
import pdfplumber

# OCR Engine - Tesseract (Memory Efficient & Reliable)
import pytesseract

# Semantic Summarization (LSA - Latent Semantic Analysis)
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Image Processing
from PIL import Image

# Initialize Flask
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default-dev-key-change-in-prod')

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp', 'gif'}
IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp', 'gif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max

# Configure Tesseract OCR
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    print(f"[OCR] Tesseract configured: {TESSERACT_PATH}", flush=True)
else:
    print("[OCR] Warning: Tesseract not found at default path", flush=True)

# Initialize Summarizer
LANGUAGE = "english"
stemmer = Stemmer(LANGUAGE)
# Initialize Summarizers
# 1. LSA (Latent Semantic Analysis) - Better for structured text (PDFs, Articles)
lsa_summarizer = LsaSummarizer(stemmer)
lsa_summarizer.stop_words = get_stop_words(LANGUAGE)

# 2. TextRank (Graph-based) - Better for noisy text (OCR, fragments)
text_rank_summarizer = TextRankSummarizer(stemmer)
text_rank_summarizer.stop_words = get_stop_words(LANGUAGE)

# ============== HELPER FUNCTIONS ==============

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in IMAGE_EXTENSIONS

def clean_text(text):
    """Clean and normalize text."""
    if not text:
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix broken lines (hyphenation at line end)
    # e.g. "uni- versity" -> "university"
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    
    # Fix missing spaces after punctuation (e.g. "word.Next" -> "word. Next")
    text = re.sub(r'([.,!?;:])([A-Z])', r'\1 \2', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
    return text.strip()

def get_word_count(text):
    """Count words in text."""
    if not text:
        return 0
    return len(text.split())

# ============== OCR FUNCTIONS ==============

def preprocess_image_for_ocr(img):
    """
    Conservative Image Preprocessing for OCR.
    
    Key insight: 
    - Small/thumbnail images need aggressive upscaling
    - Large/clear images should be preserved as-is
    """
    from PIL import ImageOps, ImageEnhance
    
    print(f"[PREPROCESS] Original: size={img.size}, mode={img.mode}", flush=True)
    
    # Step 1: Convert to RGB first (handle RGBA, P, etc.)
    if img.mode == 'RGBA':
        # Create white background for transparent images
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    
    # Step 2: Smart sizing for OCR
    width, height = img.size
    max_dimension = max(width, height)
    
    # OCR needs at least 10-12 pixels per character
    # For a document with small text, we need minimum ~1000px width
    
    if max_dimension < 600:
        # Very small image (thumbnail) - scale up 4x
        scale_factor = 4.0
        new_size = (int(width * scale_factor), int(height * scale_factor))
        img = img.resize(new_size, Image.LANCZOS)
        print(f"[PREPROCESS] SMALL IMAGE: Scaled 4x to {img.size}", flush=True)
    elif max_dimension < 1000:
        # Small image - scale up 2x
        scale_factor = 2.0
        new_size = (int(width * scale_factor), int(height * scale_factor))
        img = img.resize(new_size, Image.LANCZOS)
        print(f"[PREPROCESS] Scaled 2x to {img.size}", flush=True)
    elif max_dimension > 4000:
        # Too large - scale down to prevent memory issues
        scale_factor = 4000 / max_dimension
        new_size = (int(width * scale_factor), int(height * scale_factor))
        img = img.resize(new_size, Image.LANCZOS)
        print(f"[PREPROCESS] Scaled DOWN to {img.size}", flush=True)
    
    # Step 3: Convert to Grayscale
    img_gray = img.convert('L')
    
    # Step 4: Light contrast enhancement (helps with faded text)
    enhancer = ImageEnhance.Contrast(img_gray)
    img_contrast = enhancer.enhance(1.2)  # Gentle 20% boost
    
    # Step 5: Autocontrast to normalize levels
    img_final = ImageOps.autocontrast(img_contrast, cutoff=0.5)
    
    print(f"[PREPROCESS] Final: size={img_final.size}, mode={img_final.mode}", flush=True)
    
    return img_final


def extract_text_from_image(image_path):
    """
    Extract text from image using Tesseract OCR with professional preprocessing.
    Handles any image size/dimension through intelligent preprocessing.
    """
    print(f"[OCR] Processing image: {image_path}", flush=True)
    
    try:
        # Open image
        img = Image.open(image_path)
        original_size = img.size
        print(f"[OCR] Original image: {img.size}, mode: {img.mode}", flush=True)
        
        # Check if image is too small for reliable OCR
        max_dim = max(original_size)
        is_low_quality = max_dim < 500
        if is_low_quality:
            print(f"[OCR] WARNING: Image is very small ({original_size}). OCR may be unreliable.", flush=True)
        
        # Apply preprocessing
        img_processed = preprocess_image_for_ocr(img)
        
        # Run OCR with multiple PSM modes for best results
        results = []
        
        # Try PSM 6 first (single block) - often works better for documents
        config_block = r'--oem 3 --psm 6'
        text_block = pytesseract.image_to_string(img_processed, lang='eng', config=config_block)
        if text_block.strip():
            results.append(text_block.strip())
            print(f"[OCR] PSM 6 extracted: {len(text_block)} chars", flush=True)
        
        # Try automatic mode
        config_auto = r'--oem 3 --psm 3'
        text_auto = pytesseract.image_to_string(img_processed, lang='eng', config=config_auto)
        if text_auto.strip():
            # Use whichever got more coherent text
            if not results or len(text_auto.strip()) > len(results[0]):
                results = [text_auto.strip()]
                print(f"[OCR] PSM 3 extracted: {len(text_auto)} chars", flush=True)
        
        img.close()
        
        if results:
            final_text = results[0]
            
            # Clean up OCR artifacts (common issues)
            import re
            # Remove lines that are mostly gibberish (>50% non-word characters)
            lines = final_text.split('\n')
            clean_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Count word characters vs total
                word_chars = len(re.findall(r'[a-zA-Z0-9]', line))
                if len(line) > 0 and word_chars / len(line) > 0.4:
                    clean_lines.append(line)
            
            final_text = ' '.join(clean_lines)
            
            # Add warning for low quality images
            if is_low_quality and len(final_text) > 20:
                final_text = "[Note: Image resolution was low, text may have errors] " + final_text
            
            print(f"[OCR] Success! Extracted {len(final_text)} characters.", flush=True)
            return final_text
        else:
            print("[OCR] No text detected in image.", flush=True)
            return ""
            
    except Exception as e:
        print(f"[OCR] Error: {e}", flush=True)
        return f"OCR Error: {str(e)}"

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF.
    - First tries native text extraction (for text PDFs)
    - Falls back to OCR (for scanned PDFs)
    """
    print(f"[PDF] Processing: {pdf_path}", flush=True)
    text_content = []
    used_ocr = False
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Try native text extraction first
                page_text = page.extract_text()
                
                if page_text and len(page_text.strip()) > 10:
                    text_content.append(page_text)
                    print(f"[PDF] Page {i+1}: Extracted {len(page_text)} chars (native).", flush=True)
                else:
                    # Page has no text - it's likely scanned, use OCR
                    print(f"[PDF] Page {i+1}: No native text, using OCR...", flush=True)
                    used_ocr = True
                    
                    # Convert page to image for OCR
                    img = page.to_image(resolution=200)
                    
                    # Save to temp file for OCR
                    # Save to temp file for OCR
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        img.save(tmp.name)
                        tmp_path = tmp.name
                    
                    # File is now closed, so 'extract_text_from_image' can open it
                    try:
                        ocr_text = extract_text_from_image(tmp_path)
                        
                        if ocr_text and not ocr_text.startswith("OCR Error"):
                            text_content.append(ocr_text)
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
        
        full_text = '\n'.join(text_content)
        print(f"[PDF] Total extracted: {len(full_text)} chars, OCR used: {used_ocr}", flush=True)
        return full_text.strip(), used_ocr
        
    except Exception as e:
        print(f"[PDF] Error: {e}", flush=True)
        return f"PDF Error: {str(e)}", False

# ============== SUMMARIZATION (SEMANTIC) ==============

def generate_summary(text, source_type='text', num_sentences=3):
    """
    Generate summary using hybrid strategy:
    - Text/PDF: Use LSA (previous method, better for structured content)
    - Image: Use TextRank (better for noisy/OCR content)
    """
    if not text or len(text.strip()) < 20:
        return text
    
    try:
        # Parse text
        parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
        
        # Select summarizer based on source
        if source_type in ['pdf', 'text']:
            # Use LSA for clean/structured text (User Request)
            print(f"[SUMMARY] Using LSA algorithm for {source_type}", flush=True)
            summary_sentences = lsa_summarizer(parser.document, num_sentences)
        else:
            # Use TextRank for Images (OCR text) which often lacks structure
            print(f"[SUMMARY] Using TextRank algorithm for {source_type}", flush=True)
            summary_sentences = text_rank_summarizer(parser.document, num_sentences)
        
        # Join sentences
        summary = ' '.join(str(sentence) for sentence in summary_sentences)
        
        if not summary.strip():
            # Fallback: return first few sentences
            sentences = text.split('.')
            summary = '. '.join(sentences[:num_sentences]) + '.'
        
        return summary.strip()
        
    except Exception as e:
        print(f"[SUMMARY] Error: {e}", flush=True)
        # Fallback to simple extraction
        sentences = text.split('.')
        return '. '.join(sentences[:num_sentences]) + '.'

# ============== SENTIMENT ANALYSIS ==============

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob."""
    if not text:
        return "Neutral", 0.0
    
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            sentiment = "Positive"
        elif polarity < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        return sentiment, polarity
        
    except Exception as e:
        return "Neutral", 0.0

# ============== ROUTES ==============

@app.route('/')
def index():
    return render_template('index.html', summary=None)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = None
    source = "text"
    used_ocr = False
    
    input_type = request.form.get('input_type', 'text')
    print(f"\n[REQUEST] Input type: {input_type}", flush=True)
    
    # ===== TEXT INPUT =====
    if input_type == 'text':
        text = request.form.get('input_text', '').strip()
        source = "text"
        
        if not text:
            flash('Please enter some text to summarize.')
            return render_template('index.html', summary=None)
    
    # ===== PDF INPUT =====
    elif input_type == 'pdf':
        # Find the actual file from multiple inputs with same name
        file = None
        if 'pdf_file' in request.files:
            for f in request.files.getlist('pdf_file'):
                if f and f.filename:
                    file = f
                    break
        
        if not file:
            flash('Please upload a PDF file.')
            return render_template('index.html', summary=None)
            
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload a PDF.')
            return render_template('index.html', summary=None)
        
        print(f"[PDF] Received file: {file.filename}", flush=True)
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(file_path)
            result = extract_text_from_pdf(file_path)
            
            if isinstance(result, tuple):
                text, used_ocr = result
            else:
                text = result
                
            source = "pdf"
            
        except Exception as e:
            flash(f'Error processing PDF: {str(e)}')
            return render_template('index.html', summary=None)
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    # ===== IMAGE INPUT =====
    elif input_type == 'image':
        # Find the actual file from multiple inputs with same name
        file = None
        if 'pdf_file' in request.files:
            for f in request.files.getlist('pdf_file'):
                if f and f.filename:
                    file = f
                    break
        
        if not file:
            flash('Please upload an image file.')
            return render_template('index.html', summary=None)
        
        print(f"[IMAGE] Received file: {file.filename}", flush=True)
            
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(file_path)
            print(f"[IMAGE] File saved to: {file_path}", flush=True)
            text = extract_text_from_image(file_path)
            used_ocr = True
            source = "image"
            
        except Exception as e:
            print(f"[IMAGE] Error: {e}", flush=True)
            flash(f'Error processing image: {str(e)}')
            return render_template('index.html', summary=None)
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    else:
        flash('Invalid input type.')
        return render_template('index.html', summary=None)
    
    # ===== VALIDATION =====
    if not text:
        flash('Could not extract any text. Please try again.')
        return render_template('index.html', summary=None)
    
    if text.startswith("OCR Error") or text.startswith("PDF Error"):
        flash(text)
        return render_template('index.html', summary=None)
    
    # Clean text
    text = clean_text(text)
    text_word_count = get_word_count(text)
    
    if text_word_count < 5:
        flash('Text is too short. Please provide more content.')
        return render_template('index.html', summary=None, original_text=text)
    
    # ===== GET PARAMETERS =====
    try:
        num_sentences = int(request.form.get('num_sentences', 3))
        num_sentences = max(1, min(10, num_sentences))
    except ValueError:
        num_sentences = 3
    
    # ===== GENERATE SUMMARY (HYBRID) =====
    summary = generate_summary(text, source, num_sentences)
    summary_word_count = get_word_count(summary)
    
    # ===== SENTIMENT ANALYSIS =====
    original_sentiment, original_score = analyze_sentiment(text)
    summary_sentiment, summary_score = analyze_sentiment(summary)
    
    # ===== STATISTICS =====
    compression = round((1 - summary_word_count / text_word_count) * 100, 1) if text_word_count > 0 else 0
    
    print(f"[RESULT] Summary: {summary_word_count} words, Compression: {compression}%", flush=True)
    
    return render_template('index.html',
                           summary=summary,
                           original_text=text,
                           original_sentiment=original_sentiment,
                           original_sentiment_score=round(original_score, 2),
                           summary_sentiment=summary_sentiment,
                           summary_sentiment_score=round(summary_score, 2),
                           text_word_count=text_word_count,
                           summary_word_count=summary_word_count,
                           compression_ratio=compression,
                           num_sentences=num_sentences,
                           source=source,
                           used_ocr=used_ocr)

if __name__ == '__main__':
    # Pre-warm OCR engine (optional, for faster first request)
    print("[STARTUP] Text Summarizer Pro v2 starting...", flush=True)
    print("[STARTUP] Architecture: EasyOCR + Sumy LSA (Semantic Summarization)", flush=True)
    app.run(debug=True)

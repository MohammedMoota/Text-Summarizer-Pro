# Text Summarizer Pro

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![Status](https://img.shields.io/badge/Status-Active-success)

**Text Summarizer Pro** is a powerful, locally-hosted web application that uses advanced NLP and OCR technologies to summarize text from various sources including raw text, PDF documents, and images.

##  Key Features

*   **Hybrid Summarization Engine**:
    *   **LSA (Latent Semantic Analysis)** for structured text and PDFs.
    *   **TextRank** for noisy text (OCR output) to maintain coherence.
*   **Smart OCR Integration**:
    *   Powered by **Tesseract OCR**.
    *   Automatic image upscaling (4x) for small images/thumbnails.
    *   Intelligent text cleaning to remove gibberish.
*   **Professional UI**:
    *   Modern **Dark Mode** interface.
    *   Responsive design with real-time word count and sentiment analysis.
    *   Drag-and-drop file support.
*   **Privacy-First**: All processing happens locally on your machine. No data is sent to external clouds.

##  Tech Stack

*   **Backend**: Flask (Python)
*   **NLP**: Sumy, NLTK, TextBlob
*   **OCR**: Tesseract, Pillow (PIL), pdfplumber
*   **Frontend**: HTML5, CSS3 (Custom Dark Theme), Vanilla JS

##  Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/YourUsername/Text-Summarizer-Pro.git
    cd Text-Summarizer-Pro
    ```

2.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Tesseract OCR**:
    *   Download and install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).
    *   Ensure `tesseract.exe` is in your system PATH or at default location.

4.  **Download NLTK Data** (First run only):
    ```python
    import nltk
    nltk.download('punkt')
    ```

##   Usage

1.  **Start the Application**:
    ```bash
    python app.py
    ```

2.  **Open Browser**:
    Navigate to `http://127.0.0.1:5000`

3.  **Summarize**:
    *   **Text**: Paste text directly.
    *   **PDF**: Upload a digital PDF document.
    *   **Image**: Upload a screenshot or scanned image (OCR will activate).

##  Security Note
This application is designed for local use. The `SECRET_KEY` in `app.py` should be changed via environment variables if deploying to a public server.

##  License
This project is open source and available under the [MIT License](LICENSE).

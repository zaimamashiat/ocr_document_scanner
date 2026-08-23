# OCR Document Scanner

An AI-assisted **OCR Document Scanner** built with Python and Streamlit. The application extracts text and structured information from scanned documents and images, with preprocessing tools to improve OCR readability.

## Features

* Upload scanned documents and images
* Image preprocessing with OpenCV
* Image enhancement for better OCR accuracy
* Extract text from documents
* Process handwritten and printed content
* Display extracted results in Streamlit
* Convert extracted information into structured data
* Export results to Excel
* Generate formatted Excel workbooks
* Local processing support
* Integration-ready architecture for OCR and LLM models

## Tech Stack

* **Python**
* **Streamlit** – Web interface
* **OpenCV** – Image preprocessing
* **NumPy** – Image/data processing
* **Pandas** – Structured data handling
* **Pillow** – Image enhancement
* **OpenPyXL** – Excel generation
* **Requests** – API communication

## Project Structure

```text
sj_ocr/
│
├── app.py
├── requirements.txt
├── README.md
└── ...
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/zaimamashiat/ocr_document_scanner.git
cd ocr_document_scanner
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv v11
.\v11\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Requirements

The main Python packages are:

```txt
streamlit
opencv-python-headless
numpy
pandas
requests
openpyxl
Pillow
```

Python standard-library modules such as `base64`, `io`, `json`, `re`, `shutil`, `subprocess`, `tempfile`, and `pathlib` do not need to be installed separately.

## Running the Application

Start the Streamlit application:

```powershell
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Basic Workflow

```text
Document / Image
       ↓
Image Preprocessing
       ↓
Image Enhancement
       ↓
OCR Processing
       ↓
Text Extraction
       ↓
Data Structuring
       ↓
Streamlit Preview
       ↓
Excel Export
```

## Image Processing

The application uses OpenCV and Pillow for preprocessing scanned documents. Depending on the document, preprocessing can include:

* Grayscale conversion
* Contrast enhancement
* Sharpening
* Noise reduction
* Thresholding
* Document cleanup

These operations can make difficult scans and handwriting easier for the OCR engine to interpret.

## Excel Export

Extracted information can be converted into Excel workbooks using `openpyxl`.

The application supports formatting options such as:

* Headers
* Font styling
* Cell alignment
* Borders
* Background fills
* Automatic column sizing

## Supported Input

Depending on the current application configuration, the scanner can process common document/image formats such as:

```text
PNG
JPG / JPEG
WEBP
```

Additional document formats can be supported by adding the appropriate conversion or extraction pipeline.

## GitHub

Repository:

```text
https://github.com/zaimamashiat/ocr_document_scanner
```

## Development

Local project directory:

```text
C:\PROJECTS\sj_ocr
```

To push updates to GitHub:

```powershell
git add .
git commit -m "Update OCR document scanner"
git push origin main
```

## Notes

OCR accuracy depends heavily on the quality of the original document. Blurry images, unusual handwriting, shadows, low contrast, skewed pages, and very small text can reduce extraction accuracy.

Image preprocessing and AI-assisted correction can be used to improve results for difficult documents.


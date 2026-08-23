import base64
import io
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from PIL import Image, ImageEnhance, ImageFilter

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
DEFAULT_MODEL = "qwen2.5vl:7b"

# ----------------------------------------------------------------------------
# Preprocessing (thresholding) — improves both docTR and vision-model input
# ----------------------------------------------------------------------------
def preprocess_image(pil_img: Image.Image, block_size: int = 25, c_value: int = 15) -> np.ndarray:
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=block_size, C=c_value,
    )
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


def np_to_png_bytes(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("Failed to encode preprocessed image")
    return buf.tobytes()


def prepare_vision_image_bytes(image_bytes: bytes, target_width: int = 2400) -> bytes:
    """Upscale and lightly enhance the sheet for the vision model.

    Wide attendance sheets contain very small day cells. Upscaling before the
    vision request makes the tick/cross marks easier to distinguish without
    changing the original file shown to the user.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if img.width < target_width:
        scale = target_width / img.width
        img = img.resize((target_width, round(img.height * scale)), Image.Resampling.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.10)
    img = ImageEnhance.Sharpness(img).enhance(1.20)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=110, threshold=3))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# docTR OCR — a raw-text pass used as a cross-check hint for the vision model.
# NOTE: docTR's built-in recognizers are trained on Latin script. On Arabic
# handwriting it will often misread or drop characters — that's expected.
# It's included here as a second, independent signal (word/number shapes,
# grid structure) that the vision model can use to catch its own mistakes,
# not as a standalone Arabic reader.
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading docTR model (first run only)...")
def load_doctr_model():
    from doctr.models import ocr_predictor
    return ocr_predictor(det_arch="db_resnet50", reco_arch="parseq", pretrained=True)


def run_doctr(model, preprocessed: np.ndarray) -> str:
    rgb = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
    result = model([rgb])
    exported = result.export()
    lines = []
    for page in exported["pages"]:
        for block in page["blocks"]:
            for line in block["lines"]:
                lines.append(" ".join(w["value"] for w in line["words"]))
    return "\n".join(lines).strip()


EXTRACTION_PROMPT = """You are reading a handwritten Arabic school attendance sheet \
(monthly attendance for a reinforcement-class activity, from an NGO education programme).

Look at the image carefully and extract its content EXACTLY AS WRITTEN, in ARABIC — \
do NOT translate anything. Names, school name, teacher name, directorate, governorate, \
grade, and notes should all stay in Arabic script exactly as they appear.

Follow these rules:

1. Transcribe Arabic text as-is (no translation, no transliteration).
2. For the attendance grid (day columns, typically 1-31): read each cell's mark and
   classify it as one of:
     "P"  - a check/tick mark (present)
     "A"  - an X / cross mark (absent)
     "H"  - the whole column is crossed out with one long diagonal line across ALL rows
            (a non-instructional day / holiday / school break — not a per-student mark)
     ""   - empty / no mark
     "?"  - a mark is present but genuinely ambiguous (overlapping gridlines, unclear
            whether tick or cross) — use this rather than guessing
3. A blank day cell means you have visually checked that exact cell and found no handwritten
    attendance mark. Never output a run of blank days just because the grid is difficult to read.
    Zoom mentally into the grid and inspect each column separately.
4. Do NOT invent students, marks, or metadata that aren't visible. If a metadata field
   (governorate, directorate, month, school, teacher) is illegible, put your best guess
   and prefix it with "غير واضح: " (Arabic for "unclear:").
5. Preserve the original row order and any ID numbers shown.
6. Gender should be "ذكر" or "أنثى" exactly as marked, kept in Arabic.

Return ONLY valid JSON (no markdown fences, no commentary). The JSON *keys* below must stay \
exactly as shown (in English) — only the *values* should be in Arabic (except day marks, \
which stay P/A/H/?/""):

{{
  "metadata": {{
    "governorate": "...", "directorate": "...", "month": "...",
    "school_name": "...", "teacher_name": "..."
  }},
  "day_range": [1, 31],
  "students": [
    {{
      "id": "115", "name": "...", "gender": "...", "age": "...", "grade": "...",
      "days": {{"1": "P", "2": "A", "17": "H", "...": "..."}},
      "days_present_written": "16", "days_absent_written": "", "notes": ""
    }}
  ]
}}

For extra reference, here is a raw OCR pass over the (Latin-script-only) OCR engine docTR run on \
this image. It was NOT trained on Arabic, so most of it will be garbled or wrong — treat it only \
as a weak hint about word/number positions and column structure, and trust your own reading of \
the image over it wherever they disagree:
---
{ocr_text}
---
"""


ATTENDANCE_VERIFICATION_PROMPT = """You are doing a SECOND, attendance-only verification pass on an Arabic monthly attendance sheet.

Ignore metadata and handwriting names except for using row order / visible student IDs to align rows.
Read the attendance grid cell-by-cell. The sheet is RIGHT-TO-LEFT: day 1 is the rightmost day
column and day numbers increase toward the LEFT.

IMPORTANT: inspect days 4-14 especially carefully. They may contain faint red tick marks or red X
marks partly hidden by blue/red vertical grid lines. Do not treat these days as blank unless you
have checked each cell.

Use only these marks:
  P = tick/check (present)
  A = X/cross (absent)
  H = whole day column crossed out across rows (holiday/no session)
  ? = visible but genuinely ambiguous
  "" = truly empty cell

Return ONLY valid JSON in this exact shape:
{
  "students": [
    {"id": "visible id or empty", "row": 1, "days": {"1": "P", "2": "", "3": "A"}}
  ]
}

Include every visible student row in top-to-bottom order and include keys for all days 1-31.
"""


def merge_attendance_fill_blanks(data: dict, verified: dict) -> dict:
    """Use the focused second pass to fill cells missed by the first pass.

    Existing non-empty marks are preserved. The verifier only fills blanks, which
    prevents a second noisy pass from overwriting already-detected attendance.
    """
    students = data.get("students", [])
    verified_students = verified.get("students", [])
    for idx, s in enumerate(students):
        if idx >= len(verified_students):
            break
        v = verified_students[idx]
        # If both IDs are readable and disagree, do not merge that row.
        sid = str(s.get("id", "")).strip()
        vid = str(v.get("id", "")).strip()
        if sid and vid and sid != vid:
            continue
        days = s.setdefault("days", {})
        for d in range(1, 32):
            key = str(d)
            current = str(days.get(key, "") or "").strip()
            candidate = str(v.get("days", {}).get(key, "") or "").strip()
            if not current and candidate in {"P", "A", "H", "?"}:
                days[key] = candidate
    return data


# ----------------------------------------------------------------------------
# Ollama helpers
# ----------------------------------------------------------------------------
def list_ollama_models() -> list[str]:
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def extract_sheet_data(image_bytes: bytes, model: str, timeout: int, ocr_text: str = "") -> dict:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt = EXTRACTION_PROMPT.format(ocr_text=ocr_text or "(no OCR text available)")
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192,      # default 2048 is too small once the image + a
                                   # multi-row JSON response are both in context —
                                   # this was silently truncating output.
            "num_predict": 4096,  # allow a long enough completion for a full sheet
        },
    }
    resp = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=timeout)

    if resp.status_code == 400:
        # Ollama's 400 body usually names the real problem (e.g. "model does not
        # support images", "model not found") — surface it instead of a generic
        # HTTPError, since that's almost always more useful than the status code.
        try:
            detail = resp.json().get("error", resp.text)
        except ValueError:
            detail = resp.text
        raise ValueError(
            f"Ollama rejected the request (400): {detail}\n"
            "Most common cause: the selected model isn't multimodal (can't accept images). "
            "Vision-capable models include qwen2.5vl, llama3.2-vision, minicpm-v, llava — "
            "check `ollama list` and pull one of these if needed."
        )
    resp.raise_for_status()
    raw = resp.json().get("response", "").strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Model did not return valid JSON: {e}\n--- raw ---\n{raw[:2000]}")


def verify_attendance(image_bytes: bytes, model: str, timeout: int) -> dict:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": model,
        "prompt": ATTENDANCE_VERIFICATION_PROMPT,
        "images": [b64],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 8192,
            "num_predict": 4096,
        },
    }
    resp = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    raw = resp.json().get("response", "").strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("Attendance verification pass did not return valid JSON.")


# ----------------------------------------------------------------------------
# JSON <-> flat table (editable in Streamlit)
# ----------------------------------------------------------------------------
def data_to_dataframe(data: dict) -> pd.DataFrame:
    day_lo, day_hi = data.get("day_range", [1, 31])
    rows = []
    for s in data.get("students", []):
        row = {
            "id": s.get("id", ""),
            "name": s.get("name", ""),
            "gender": s.get("gender", ""),
            "age": s.get("age", ""),
            "grade": s.get("grade", ""),
        }
        days = s.get("days", {})
        for d in range(day_lo, day_hi + 1):
            row[f"d{d}"] = days.get(str(d), "")
        row["present_written"] = s.get("days_present_written", "")
        row["absent_written"] = s.get("days_absent_written", "")
        row["notes"] = s.get("notes", "")
        rows.append(row)
    return pd.DataFrame(rows)


def dataframe_to_data(df: pd.DataFrame, metadata: dict, day_lo: int, day_hi: int) -> dict:
    students = []
    for _, row in df.iterrows():
        days = {str(d): str(row.get(f"d{d}", "") or "") for d in range(day_lo, day_hi + 1)}
        students.append({
            "id": str(row.get("id", "")),
            "name": str(row.get("name", "")),
            "gender": str(row.get("gender", "")),
            "age": str(row.get("age", "")),
            "grade": str(row.get("grade", "")),
            "days": days,
            "days_present_written": str(row.get("present_written", "")),
            "days_absent_written": str(row.get("absent_written", "")),
            "notes": str(row.get("notes", "")),
        })
    return {"metadata": metadata, "day_range": [day_lo, day_hi], "students": students}


# ----------------------------------------------------------------------------
# Excel builder
# ----------------------------------------------------------------------------
def build_excel(data: dict) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "كشف الحضور"
    ws.sheet_view.rightToLeft = True  # RTL layout for Arabic content

    thin = Side(style="thin", color="000000")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    header_fill = PatternFill("solid", fgColor="D9E1F2")
    holiday_fill = PatternFill("solid", fgColor="F2F2F2")
    flag_fill = PatternFill("solid", fgColor="FFF2CC")
    font_normal = Font(name="Arial", size=10)
    font_bold = Font(name="Arial", size=10, bold=True)
    font_title = Font(name="Arial", size=13, bold=True)
    font_present = Font(name="Arial", size=10, color="1F7A1F")
    font_absent = Font(name="Arial", size=10, color="C00000", bold=True)
    font_flag = Font(name="Arial", size=10, color="9C6500", bold=True)
    align_right = Alignment(horizontal="right", vertical="center")

    day_lo, day_hi = data.get("day_range", [1, 31])
    n_days = day_hi - day_lo + 1
    day_col_start = 6
    total_cols = 5 + n_days + 3

    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=total_cols)
    ws["A1"] = "كشف التحضير الشهري"
    ws["A1"].font = font_title
    ws["A1"].alignment = Alignment(horizontal="center")

    meta = data.get("metadata", {})
    meta_rows = [
        ("المحافظة:", meta.get("governorate", "")),
        ("المديرية:", meta.get("directorate", "")),
        ("الشهر:", meta.get("month", "")),
        ("اسم المدرسة:", meta.get("school_name", "")),
        ("اسم المعلم:", meta.get("teacher_name", "")),
    ]
    row = 3
    for label, val in meta_rows:
        c1 = ws.cell(row=row, column=1, value=label)
        c1.font = font_bold
        c1.alignment = align_right
        ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=5)
        c = ws.cell(row=row, column=2, value=val)
        c.font = font_normal
        c.alignment = align_right
        if str(val).startswith("غير واضح"):
            c.fill = flag_fill
        row += 1

    start_row = row + 1
    headers = ["الرقم", "الاسم", "الجنس", "العمر", "الصف"] + \
              [str(d) for d in range(day_lo, day_hi + 1)] + \
              ["أيام الحضور", "أيام الغياب", "ملاحظات"]
    for col, h in enumerate(headers, start=1):
        c = ws.cell(row=start_row, column=col, value=h)
        c.font = font_bold
        c.fill = header_fill
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = border

    mark_style = {
        "P": ("\u2713", font_present),
        "A": ("X", font_absent),
        "?": ("?", font_flag),
        "H": ("\u2014", None),
        "": ("", font_normal),
    }

    r = start_row + 1
    for s in data.get("students", []):
        ws.cell(row=r, column=1, value=s.get("id", "")).font = font_normal
        c = ws.cell(row=r, column=2, value=s.get("name", ""))
        c.font = font_normal
        c.alignment = align_right
        ws.cell(row=r, column=3, value=s.get("gender", "")).font = font_normal
        ws.cell(row=r, column=4, value=s.get("age", "")).font = font_normal
        ws.cell(row=r, column=5, value=s.get("grade", "")).font = font_normal

        days = s.get("days", {})
        for i, d in enumerate(range(day_lo, day_hi + 1)):
            col = day_col_start + i
            cell = ws.cell(row=r, column=col)
            cell.alignment = Alignment(horizontal="center")
            cell.border = border
            mark = days.get(str(d), "")
            symbol, font = mark_style.get(mark, ("", font_normal))
            cell.value = symbol
            if mark == "H":
                cell.fill = holiday_fill
            elif font:
                cell.font = font

        present_col = day_col_start + n_days
        absent_col = present_col + 1
        notes_col = absent_col + 1
        ws.cell(row=r, column=present_col, value=s.get("days_present_written", "")).font = font_bold
        ws.cell(row=r, column=absent_col, value=s.get("days_absent_written", "")).font = font_normal
        notes = s.get("notes", "")
        if "?" in days.values():
            notes = (notes + " | يحتاج تأكيد").strip(" |")
        ncell = ws.cell(row=r, column=notes_col, value=notes)
        ncell.font = font_normal
        ncell.alignment = align_right
        if notes:
            ncell.fill = flag_fill

        for col in range(1, notes_col + 1):
            ws.cell(row=r, column=col).border = border
        r += 1

    r += 2
    ws.cell(row=r, column=1, value="دليل الرموز").font = font_bold
    r += 1
    for line in [
        "✓ = حاضر    X = غائب    — = لا يوجد نشاط (عطلة)    ? = غير واضح، يحتاج تأكيد",
        "تم الاستخراج تلقائيًا بواسطة نموذج ذكاء اصطناعي من صورة مكتوبة بخط اليد — يُرجى مراجعة الخلايا المُعلّمة.",
    ]:
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=total_cols)
        c = ws.cell(row=r, column=1, value=line)
        c.font = Font(name="Arial", size=9, italic=True)
        c.alignment = align_right
        r += 1

    ws.column_dimensions["A"].width = 6
    ws.column_dimensions["B"].width = 30
    ws.column_dimensions["C"].width = 7
    ws.column_dimensions["D"].width = 6
    ws.column_dimensions["E"].width = 8
    for i in range(n_days):
        ws.column_dimensions[get_column_letter(day_col_start + i)].width = 3.5
    ws.column_dimensions[get_column_letter(day_col_start + n_days)].width = 9
    ws.column_dimensions[get_column_letter(day_col_start + n_days + 1)].width = 9
    ws.column_dimensions[get_column_letter(day_col_start + n_days + 2)].width = 24
    ws.freeze_panes = ws.cell(row=start_row + 1, column=day_col_start)

    # One-page PDF/print layout: A4 landscape, all 31 dates on a single page.
    ws.page_setup.orientation = ws.ORIENTATION_LANDSCAPE
    ws.page_setup.paperSize = ws.PAPERSIZE_A4
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 1
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.print_options.horizontalCentered = True
    ws.page_margins.left = 0.20
    ws.page_margins.right = 0.20
    ws.page_margins.top = 0.30
    ws.page_margins.bottom = 0.30
    ws.page_margins.header = 0.10
    ws.page_margins.footer = 0.10
    ws.print_area = f"A1:{get_column_letter(total_cols)}{r - 1}"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# PDF export — converts the exact same formatted workbook (title box, RTL
# headers, grid, legend) so the PDF matches the Excel template precisely
# instead of being a second layout to keep in sync. Requires LibreOffice
# ("soffice") to be installed and on PATH.
# ----------------------------------------------------------------------------
def excel_to_pdf(excel_bytes: bytes) -> bytes:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        # shutil.which() often misses LibreOffice on Windows because the
        # installer doesn't add it to PATH by default — check common install
        # locations directly before giving up.
        windows_candidates = [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        ]
        for candidate in windows_candidates:
            if Path(candidate).exists():
                soffice = candidate
                break
    if not soffice:
        raise RuntimeError(
            "LibreOffice ('soffice') was not found. Install LibreOffice to enable PDF export: "
            "Linux: `sudo apt install libreoffice` | macOS: `brew install --cask libreoffice` | "
            "Windows: download from libreoffice.org — if it's installed but still not found, "
            "add 'C:\\Program Files\\LibreOffice\\program' to your PATH and restart the terminal."
        )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        xlsx_path = tmp_path / "sheet.xlsx"
        xlsx_path.write_bytes(excel_bytes)
        result = subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(tmp_path), str(xlsx_path)],
            capture_output=True, text=True, timeout=120,
        )
        pdf_path = tmp_path / "sheet.pdf"
        if not pdf_path.exists():
            raise RuntimeError(f"LibreOffice conversion failed:\n{result.stdout}\n{result.stderr}")
        return pdf_path.read_bytes()


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Attendance Sheet Translator", layout="wide")
    st.title("📋 Arabic Attendance Sheet Extractor")
    st.caption("Vision-LLM extraction (Ollama) with an editable review step — keeps everything in Arabic.")

    if "extracted" not in st.session_state:
        st.session_state.extracted = None

    with st.sidebar:
        st.header("Settings")
        available_models = list_ollama_models()
        if available_models:
            model_name = st.selectbox("Ollama vision model", available_models, index=0)
        else:
            model_name = st.text_input("Ollama vision model (Ollama not detected)", DEFAULT_MODEL)
            st.caption("Couldn't reach Ollama at localhost:11434 — make sure `ollama serve` is running.")
        timeout = st.slider("Extraction timeout (seconds)", 60, 1800, 600, step=60)
        st.caption("Use a model like `qwen2.5vl`, `llama3.2-vision`, or `minicpm-v` — plain `llama3.1` can't see images.")

        st.divider()
        use_ocr = st.checkbox("Run docTR OCR as a cross-check hint", value=True)
        st.caption(
            "docTR is Latin-script only, so on Arabic handwriting it's noisy — it's fed to the "
            "vision model only as a secondary hint, not trusted directly."
        )
        verify_grid = st.checkbox("Second-pass attendance verification", value=True)
        st.caption("Recommended for wide sheets. Re-reads the day grid and fills marks missed on the first pass, especially days 4-14.")
        block_size = st.slider("Threshold block size (odd)", 5, 51, 25, step=2, disabled=not use_ocr)
        c_value = st.slider("Threshold C (offset)", 0, 30, 15, disabled=not use_ocr)

    uploaded_file = st.file_uploader("Upload the attendance sheet (image)", type=["png", "jpg", "jpeg"])

    if uploaded_file is None:
        st.info("Upload a photo or scan to get started.")
        return

    image_bytes = uploaded_file.getvalue()
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    ocr_text = ""
    if use_ocr:
        preprocessed = preprocess_image(pil_img, block_size, c_value)
        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_img, caption="Uploaded sheet", use_container_width=True)
        with col2:
            st.image(preprocessed, caption="Thresholded (fed to docTR)", use_container_width=True)
    else:
        st.image(pil_img, caption="Uploaded sheet", use_container_width=True)

    if st.button("Extract & Translate", type="primary"):
        if use_ocr:
            with st.spinner("Running docTR OCR pass..."):
                doctr_model = load_doctr_model()
                ocr_text = run_doctr(doctr_model, preprocessed)
            with st.expander("Raw docTR OCR output (noisy — Latin-script only)"):
                st.text(ocr_text or "(no text detected)")

        with st.spinner(f"Reading the sheet with {model_name}... (can take a few minutes)"):
            try:
                vision_bytes = prepare_vision_image_bytes(image_bytes)
                data = extract_sheet_data(vision_bytes, model_name, timeout, ocr_text)
                if verify_grid:
                    with st.spinner("Verifying the attendance grid cell-by-cell (especially days 4-14)..."):
                        verified = verify_attendance(vision_bytes, model_name, timeout)
                        data = merge_attendance_fill_blanks(data, verified)
                n_students = len(data.get("students", []))
                blank_meta = sum(
                    1 for v in data.get("metadata", {}).values()
                    if not v or v.strip().rstrip(":") in ("غير واضح", "UNCLEAR")
                )
                if n_students <= 1 or blank_meta >= 3:
                    st.warning(
                        f"Extraction looks incomplete (only {n_students} student row(s), "
                        f"{blank_meta} empty metadata field(s)). This usually means the model's "
                        "response got cut off. Try: a smaller/clearer image, a stronger model, "
                        "or re-running (context/output limits were just raised — see notes below)."
                    )
                st.session_state.extracted = data
                st.success("Extraction complete — review and edit the table below.")
            except requests.exceptions.Timeout:
                st.error(f"Timed out after {timeout}s. Try a smaller model or raise the timeout in the sidebar.")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach Ollama: {e}")
            except ValueError as e:
                st.error(str(e))

    data = st.session_state.extracted
    if not data:
        return

    st.divider()
    st.subheader("Metadata")
    meta = data.get("metadata", {})
    cols = st.columns(5)
    labels = ["governorate", "directorate", "month", "school_name", "teacher_name"]
    new_meta = {}
    for col, label in zip(cols, labels):
        with col:
            new_meta[label] = st.text_input(label.replace("_", " ").title(), meta.get(label, ""))

    st.subheader("Attendance table (editable)")
    st.caption("P = present, A = absent, H = holiday/no session, ? = unclear — edit any cell directly.")
    day_lo, day_hi = data.get("day_range", [1, 31])
    df = data_to_dataframe(data)
    edited_df = st.data_editor(df, use_container_width=True, num_rows="dynamic", height=400)

    final_data = dataframe_to_data(edited_df, new_meta, day_lo, day_hi)

    st.divider()
    excel_bytes = build_excel(final_data)
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "⬇️ تحميل ملف Excel",
            data=excel_bytes,
            file_name="attendance_sheet_arabic.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with dl_col2:
        if st.button("🖨️ تجهيز ملف PDF"):
            try:
                pdf_bytes = excel_to_pdf(excel_bytes)
                st.download_button(
                    "⬇️ تحميل ملف PDF",
                    data=pdf_bytes,
                    file_name="attendance_sheet_arabic.pdf",
                    mime="application/pdf",
                )
            except RuntimeError as e:
                st.error(str(e))

    with st.expander("Raw extracted JSON"):
        st.json(final_data)


if __name__ == "__main__":
    main()
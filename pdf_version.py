import io
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pypdfium2 as pdfium
import requests
import streamlit as st

from PIL import Image
from paddleocr import PaddleOCR

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, A4, landscape
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
)


# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_MODEL = "gemma3:12b"

PDF_RENDER_SCALE = 3.0
MAX_IMAGE_WIDTH = 3500

MIN_LINE_LENGTH_RATIO = 0.12
ATTENDANCE_MARK_BLACK_RATIO = 0.012


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Attendance Sheet Digitizer",
    page_icon="📋",
    layout="wide",
)

st.title("📋 Attendance Sheet Digitizer")

st.caption(
    "Scan attendance sheets → detect structure → extract names & attendance → "
    "review → export Excel / CSV / recreated printable PDF"
)


# ============================================================
# SESSION STATE
# ============================================================

if "pages" not in st.session_state:
    st.session_state.pages = []

if "results" not in st.session_state:
    st.session_state.results = []


# ============================================================
# OCR
# ============================================================

@st.cache_resource
def load_ocr():
    try:
        return PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )
    except Exception:
        try:
            return PaddleOCR(
                lang="en",
                use_angle_cls=True,
            )
        except Exception:
            return PaddleOCR(lang="en")


def polygon_to_box(poly):
    p = np.asarray(poly)

    xs = p[:, 0]
    ys = p[:, 1]

    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    ]


def normalize_paddle_result(result):
    words = []

    try:
        result_dict = result.res

        texts = result_dict.get("rec_texts", [])
        scores = result_dict.get("rec_scores", [])

        polygons = (
            result_dict.get("rec_polys")
            or result_dict.get("dt_polys")
            or []
        )

        for text, score, poly in zip(
            texts,
            scores,
            polygons,
        ):
            text = str(text).strip()

            if not text:
                continue

            words.append(
                {
                    "text": text,
                    "confidence": float(score),
                    "box": polygon_to_box(poly),
                }
            )

    except Exception:
        pass

    return words


def run_ocr(ocr, image):
    if len(image.shape) == 2:
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB,
        )
    else:
        rgb = image.copy()

    # PaddleOCR 3.x
    try:
        results = ocr.predict(rgb)

        words = []

        for result in results:
            words.extend(
                normalize_paddle_result(result)
            )

        if words:
            return words

    except Exception as exc:
        print("PaddleOCR predict fallback:", exc)

    # Older PaddleOCR fallback
    try:
        results = ocr.ocr(rgb)

        words = []

        if not results:
            return words

        for page in results:
            if not page:
                continue

            for line in page:
                box = line[0]
                text = str(line[1][0]).strip()
                score = float(line[1][1])

                if not text:
                    continue

                words.append(
                    {
                        "text": text,
                        "confidence": score,
                        "box": polygon_to_box(box),
                    }
                )

        return words

    except Exception as exc:
        raise RuntimeError(
            f"PaddleOCR failed: {exc}"
        )


def ocr_crop(
    ocr,
    image,
    upscale=3.0,
):
    if image.size == 0:
        return ""

    crop = image.copy()

    h, w = crop.shape[:2]

    if h < 5 or w < 5:
        return ""

    crop = cv2.resize(
        crop,
        None,
        fx=upscale,
        fy=upscale,
        interpolation=cv2.INTER_CUBIC,
    )

    if len(crop.shape) == 3:
        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_RGB2GRAY,
        )
    else:
        gray = crop

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    gray = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    try:
        words = run_ocr(
            ocr,
            gray,
        )
    except Exception:
        return ""

    words = sorted(
        words,
        key=lambda x: (
            x["box"][1],
            x["box"][0],
        ),
    )

    return " ".join(
        item["text"]
        for item in words
    ).strip()


# ============================================================
# INPUT FILES
# ============================================================

def pdf_to_images(data):
    pdf = pdfium.PdfDocument(data)

    pages = []

    for i in range(len(pdf)):
        page = pdf[i]

        rendered = page.render(
            scale=PDF_RENDER_SCALE,
        )

        pil_image = (
            rendered
            .to_pil()
            .convert("RGB")
        )

        pages.append(
            np.array(pil_image)
        )

    return pages


def uploaded_file_to_images(
    uploaded_file,
):
    data = uploaded_file.getvalue()

    suffix = (
        Path(uploaded_file.name)
        .suffix
        .lower()
    )

    if suffix == ".pdf":
        return pdf_to_images(data)

    image = Image.open(
        io.BytesIO(data)
    ).convert("RGB")

    return [
        np.array(image)
    ]


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def resize_image(
    image,
    max_width=MAX_IMAGE_WIDTH,
):
    h, w = image.shape[:2]

    if w <= max_width:
        return image

    scale = (
        max_width
        / float(w)
    )

    return cv2.resize(
        image,
        (
            int(w * scale),
            int(h * scale),
        ),
        interpolation=cv2.INTER_AREA,
    )


def deskew_image(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY,
    )

    edges = cv2.Canny(
        gray,
        50,
        150,
        apertureSize=3,
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=max(
            100,
            image.shape[1] // 4,
        ),
        maxLineGap=20,
    )

    if lines is None:
        return image

    angles = []

    for line in lines[:, 0]:
        x1, y1, x2, y2 = line

        angle = math.degrees(
            math.atan2(
                y2 - y1,
                x2 - x1,
            )
        )

        if abs(angle) < 12:
            angles.append(angle)

    if not angles:
        return image

    angle = float(
        np.median(angles)
    )

    if abs(angle) < 0.1:
        return image

    h, w = image.shape[:2]

    center = (
        w // 2,
        h // 2,
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0,
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return rotated


def preprocess(image):
    image = resize_image(image)
    image = deskew_image(image)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    gray = clahe.apply(gray)

    return image, gray


# ============================================================
# GRID DETECTION
# ============================================================

def group_positions(
    values,
    tolerance=5,
):
    if not values:
        return []

    values = sorted(values)

    groups = [
        [values[0]]
    ]

    for value in values[1:]:
        mean = np.mean(
            groups[-1]
        )

        if abs(
            value - mean
        ) <= tolerance:
            groups[-1].append(
                value
            )
        else:
            groups.append(
                [value]
            )

    return [
        int(
            round(
                np.mean(group)
            )
        )
        for group in groups
    ]


def detect_lines(gray):
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        13,
    )

    h, w = gray.shape

    # Horizontal
    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(20, w // 40),
            1,
        ),
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        h_kernel,
    )

    # Vertical
    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            1,
            max(20, h // 40),
        ),
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        v_kernel,
    )

    h_proj = np.sum(
        horizontal > 0,
        axis=1,
    )

    v_proj = np.sum(
        vertical > 0,
        axis=0,
    )

    y_threshold = max(
        25,
        int(
            w
            * MIN_LINE_LENGTH_RATIO
        ),
    )

    x_threshold = max(
        25,
        int(
            h
            * MIN_LINE_LENGTH_RATIO
        ),
    )

    y_candidates = np.where(
        h_proj > y_threshold
    )[0].tolist()

    x_candidates = np.where(
        v_proj > x_threshold
    )[0].tolist()

    ys = group_positions(
        y_candidates,
        tolerance=4,
    )

    xs = group_positions(
        x_candidates,
        tolerance=4,
    )

    return (
        xs,
        ys,
        horizontal,
        vertical,
        binary,
    )


# ============================================================
# OCR HELPERS
# ============================================================

def word_center(box):
    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2,
    )


def find_text_candidates(
    words,
    keyword,
):
    keyword = keyword.lower()

    matches = []

    for word in words:
        text = (
            word["text"]
            .lower()
            .strip()
        )

        if keyword in text:
            matches.append(
                word
            )

    return matches


# ============================================================
# ATTENDANCE TABLE REGION
# ============================================================

def detect_attendance_anchor(
    words,
    image_height,
):
    keywords = [
        "student full name",
        "student name",
        "sex",
        "total absent",
        "absent days",
    ]

    y_values = []

    for keyword in keywords:
        matches = find_text_candidates(
            words,
            keyword,
        )

        for match in matches:
            _, cy = word_center(
                match["box"]
            )

            y_values.append(
                cy
            )

    if y_values:
        return int(
            np.median(y_values)
        )

    return int(
        image_height * 0.45
    )


def estimate_attendance_region(
    image,
    words,
    xs,
    ys,
):
    h, w = image.shape[:2]

    anchor_y = detect_attendance_anchor(
        words,
        h,
    )

    nearby_top = [
        y
        for y in ys
        if (
            anchor_y - 100
            <= y
            <= anchor_y + 60
        )
    ]

    if nearby_top:
        table_top = min(
            nearby_top
        )
    else:
        table_top = int(
            h * 0.45
        )

    lower_lines = [
        y
        for y in ys
        if y > table_top
    ]

    if lower_lines:
        table_bottom = min(
            int(h * 0.93),
            max(lower_lines),
        )
    else:
        table_bottom = int(
            h * 0.9
        )

    valid_x = [
        x
        for x in xs
        if (
            w * 0.005
            < x
            < w * 0.995
        )
    ]

    if len(valid_x) >= 2:
        table_left = min(
            valid_x
        )
        table_right = max(
            valid_x
        )
    else:
        table_left = int(
            w * 0.01
        )
        table_right = int(
            w * 0.99
        )

    return (
        table_left,
        table_top,
        table_right,
        table_bottom,
    )


# ============================================================
# FILTER TABLE LINES
# ============================================================

def line_strength_vertical(
    vertical_mask,
    x,
    y1,
    y2,
):
    h, w = vertical_mask.shape

    crop = vertical_mask[
        max(0, y1):
        min(h, y2),
        max(0, x - 2):
        min(w, x + 3),
    ]

    if crop.size == 0:
        return 0

    return float(
        np.mean(crop > 0)
    )


def line_strength_horizontal(
    horizontal_mask,
    y,
    x1,
    x2,
):
    h, w = horizontal_mask.shape

    crop = horizontal_mask[
        max(0, y - 2):
        min(h, y + 3),
        max(0, x1):
        min(w, x2),
    ]

    if crop.size == 0:
        return 0

    return float(
        np.mean(crop > 0)
    )


def detect_table_verticals(
    xs,
    vertical_mask,
    table_top,
    table_bottom,
):
    result = []

    for x in xs:
        strength = (
            line_strength_vertical(
                vertical_mask,
                x,
                table_top,
                table_bottom,
            )
        )

        if strength > 0.18:
            result.append(x)

    return group_positions(
        result,
        tolerance=5,
    )


def detect_table_horizontals(
    ys,
    horizontal_mask,
    table_left,
    table_right,
    table_top,
    table_bottom,
):
    result = []

    for y in ys:
        if not (
            table_top - 10
            <= y
            <= table_bottom + 10
        ):
            continue

        strength = (
            line_strength_horizontal(
                horizontal_mask,
                y,
                table_left,
                table_right,
            )
        )

        if strength > 0.18:
            result.append(y)

    return group_positions(
        result,
        tolerance=5,
    )


# ============================================================
# DETECT 1-31 DAY COLUMNS
# ============================================================

def infer_attendance_columns(
    verticals,
):
    if len(verticals) < 15:
        return None

    gaps = np.diff(
        verticals
    )

    positive_gaps = [
        g
        for g in gaps
        if g > 3
    ]

    if not positive_gaps:
        return None

    sorted_gaps = sorted(
        positive_gaps
    )

    take = max(
        5,
        len(sorted_gaps) // 2,
    )

    day_width = float(
        np.median(
            sorted_gaps[:take]
        )
    )

    candidates = []

    for start in range(
        len(verticals) - 10
    ):
        run = [
            verticals[start]
        ]

        for j in range(
            start,
            len(verticals) - 1
        ):
            gap = (
                verticals[j + 1]
                - verticals[j]
            )

            if (
                day_width * 0.5
                <= gap
                <= day_width * 1.75
            ):
                run.append(
                    verticals[j + 1]
                )
            else:
                break

        if len(run) >= 15:
            candidates.append(
                run
            )

    if not candidates:
        return None

    day_lines = max(
        candidates,
        key=len,
    )

    if len(day_lines) >= 32:
        day_lines = (
            day_lines[:32]
        )

    day_start_index = (
        verticals.index(
            day_lines[0]
        )
    )

    if day_start_index < 3:
        return None

    sex_left = (
        verticals[
            day_start_index - 1
        ]
    )

    name_left = (
        verticals[
            day_start_index - 2
        ]
    )

    serial_left = (
        verticals[
            day_start_index - 3
        ]
    )

    if day_start_index >= 4:
        no_left = (
            verticals[
                day_start_index - 4
            ]
        )
    else:
        no_left = (
            verticals[0]
        )

    return {
        "day_lines": day_lines,
        "day_start_index":
            day_start_index,
        "no_left":
            no_left,
        "serial_left":
            serial_left,
        "name_left":
            name_left,
        "sex_left":
            sex_left,
        "day_left":
            day_lines[0],
        "day_right":
            day_lines[-1],
    }


# ============================================================
# STUDENT ROW DETECTION
# ============================================================

def identify_student_rows(
    horizontal_lines,
):
    if len(
        horizontal_lines
    ) < 3:
        return []

    gaps = np.diff(
        horizontal_lines
    )

    good_gaps = [
        gap
        for gap in gaps
        if 8 <= gap <= 120
    ]

    if not good_gaps:
        return []

    typical = float(
        np.median(
            good_gaps
        )
    )

    row_pairs = []

    for i in range(
        len(horizontal_lines) - 1
    ):
        y1 = horizontal_lines[i]
        y2 = horizontal_lines[
            i + 1
        ]

        gap = y2 - y1

        if (
            typical * 0.50
            <= gap
            <= typical * 1.7
        ):
            row_pairs.append(
                (y1, y2)
            )

    best_run = []
    current = []
    previous_y2 = None

    for pair in row_pairs:
        y1, y2 = pair

        if (
            previous_y2 is None
            or abs(
                y1 - previous_y2
            ) <= 8
        ):
            current.append(
                pair
            )
        else:
            if (
                len(current)
                > len(best_run)
            ):
                best_run = current

            current = [
                pair
            ]

        previous_y2 = y2

    if (
        len(current)
        > len(best_run)
    ):
        best_run = current

    return best_run


# ============================================================
# CROP CELL
# ============================================================

def crop_cell(
    image,
    x1,
    y1,
    x2,
    y2,
    pad=2,
):
    h, w = image.shape[:2]

    x1 = max(
        0,
        int(x1 + pad),
    )

    y1 = max(
        0,
        int(y1 + pad),
    )

    x2 = min(
        w,
        int(x2 - pad),
    )

    y2 = min(
        h,
        int(y2 - pad),
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return np.empty(
            (0, 0),
            dtype=np.uint8,
        )

    return image[
        y1:y2,
        x1:x2,
    ]


# ============================================================
# ATTENDANCE MARK DETECTION
# ============================================================

def remove_grid_lines(
    gray_crop,
):
    if gray_crop.size == 0:
        return gray_crop

    binary = cv2.threshold(
        gray_crop,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU,
    )[1]

    h, w = binary.shape

    horizontal_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(
                    3,
                    w // 2,
                ),
                1,
            ),
        )
    )

    vertical_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                1,
                max(
                    3,
                    h // 2,
                ),
            ),
        )
    )

    horizontal_lines = (
        cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            horizontal_kernel,
        )
    )

    vertical_lines = (
        cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            vertical_kernel,
        )
    )

    cleaned = cv2.subtract(
        binary,
        horizontal_lines,
    )

    cleaned = cv2.subtract(
        cleaned,
        vertical_lines,
    )

    return cleaned


def has_attendance_ink(
    image_crop,
):
    if image_crop.size == 0:
        return False

    if len(
        image_crop.shape
    ) == 3:
        gray = cv2.cvtColor(
            image_crop,
            cv2.COLOR_RGB2GRAY,
        )
    else:
        gray = image_crop

    cleaned = remove_grid_lines(
        gray
    )

    if cleaned.size == 0:
        return False

    h, w = cleaned.shape

    mx = max(
        1,
        int(w * 0.10),
    )

    my = max(
        1,
        int(h * 0.10),
    )

    inner = cleaned[
        my:
        max(
            my + 1,
            h - my,
        ),
        mx:
        max(
            mx + 1,
            w - mx,
        ),
    ]

    if inner.size == 0:
        return False

    ink_ratio = float(
        np.mean(
            inner > 0
        )
    )

    return (
        ink_ratio
        >= ATTENDANCE_MARK_BLACK_RATIO
    )


def read_attendance_mark(
    ocr,
    crop,
):
    if not has_attendance_ink(
        crop
    ):
        return ""

    if len(
        crop.shape
    ) == 3:
        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_RGB2GRAY,
        )
    else:
        gray = crop

    enlarged = cv2.resize(
        gray,
        None,
        fx=5,
        fy=5,
        interpolation=cv2.INTER_CUBIC,
    )

    try:
        text = ocr_crop(
            ocr,
            enlarged,
            upscale=1.0,
        )
    except Exception:
        text = ""

    text = (
        text
        .strip()
        .replace(" ", "")
    )

    # Only trust tiny OCR outputs
    if (
        text
        and len(text) <= 3
    ):
        return text

    # Preserve mark presence
    return "•"


# ============================================================
# ATTENDANCE EXTRACTION
# ============================================================

def extract_attendance_sheet(
    image,
    ocr,
):
    processed, gray = (
        preprocess(image)
    )

    (
        xs,
        ys,
        horizontal_mask,
        vertical_mask,
        binary,
    ) = detect_lines(
        gray
    )

    raw_words = run_ocr(
        ocr,
        processed,
    )

    (
        table_left,
        table_top,
        table_right,
        table_bottom,
    ) = estimate_attendance_region(
        processed,
        raw_words,
        xs,
        ys,
    )

    verticals = (
        detect_table_verticals(
            xs,
            vertical_mask,
            table_top,
            table_bottom,
        )
    )

    horizontals = (
        detect_table_horizontals(
            ys,
            horizontal_mask,
            table_left,
            table_right,
            table_top,
            table_bottom,
        )
    )

    columns = (
        infer_attendance_columns(
            verticals
        )
    )

    if not columns:
        raise RuntimeError(
            "Could not detect the 1–31 attendance columns."
        )

    student_rows = (
        identify_student_rows(
            horizontals
        )
    )

    if not student_rows:
        raise RuntimeError(
            "Could not detect student rows."
        )

    day_lines = (
        columns["day_lines"]
    )

    day_count = (
        len(day_lines) - 1
    )

    filtered_rows = []

    for y1, y2 in student_rows:
        name_crop = crop_cell(
            processed,
            columns["name_left"],
            y1,
            columns["sex_left"],
            y2,
            pad=3,
        )

        name = ocr_crop(
            ocr,
            name_crop,
            upscale=3,
        )

        lower_name = (
            name.lower()
        )

        if (
            "student"
            in lower_name
            or "full name"
            in lower_name
            or "first name"
            in lower_name
        ):
            continue

        filtered_rows.append(
            (
                y1,
                y2,
                name,
            )
        )

    student_rows = (
        filtered_rows
    )

    records = []

    for row_index, (
        y1,
        y2,
        preliminary_name,
    ) in enumerate(
        student_rows,
        start=1,
    ):

        # No
        no_crop = crop_cell(
            processed,
            columns["no_left"],
            y1,
            columns["serial_left"],
            y2,
            pad=3,
        )

        no_text = ocr_crop(
            ocr,
            no_crop,
            upscale=3,
        )

        # Serial
        serial_crop = crop_cell(
            processed,
            columns["serial_left"],
            y1,
            columns["name_left"],
            y2,
            pad=3,
        )

        serial_text = ocr_crop(
            ocr,
            serial_crop,
            upscale=3,
        )

        # Name
        name_crop = crop_cell(
            processed,
            columns["name_left"],
            y1,
            columns["sex_left"],
            y2,
            pad=3,
        )

        name = (
            preliminary_name
            or ocr_crop(
                ocr,
                name_crop,
                upscale=3,
            )
        )

        # Sex
        sex_crop = crop_cell(
            processed,
            columns["sex_left"],
            y1,
            columns["day_left"],
            y2,
            pad=2,
        )

        sex = ocr_crop(
            ocr,
            sex_crop,
            upscale=4,
        )

        sex = (
            sex.strip()
            .upper()
        )

        if len(sex) > 3:
            if "F" in sex:
                sex = "F"
            elif "M" in sex:
                sex = "M"
            else:
                sex = sex[:3]

        record = {
            "No": (
                no_text
                if no_text
                else str(row_index)
            ),
            "Serial No":
                serial_text,
            "Student Full Name":
                name,
            "Sex":
                sex,
        }

        for day_index in range(
            day_count
        ):
            x1 = (
                day_lines[
                    day_index
                ]
            )

            x2 = (
                day_lines[
                    day_index + 1
                ]
            )

            attendance_crop = (
                crop_cell(
                    processed,
                    x1,
                    y1,
                    x2,
                    y2,
                    pad=1,
                )
            )

            mark = (
                read_attendance_mark(
                    ocr,
                    attendance_crop,
                )
            )

            record[
                str(
                    day_index + 1
                )
            ] = mark

        records.append(
            record
        )

    df = pd.DataFrame(
        records
    )

    # Make sure 1-31 exist
    for day in range(
        1,
        32,
    ):
        column = str(day)

        if column not in df.columns:
            df[column] = ""

    expected_columns = (
        [
            "No",
            "Serial No",
            "Student Full Name",
            "Sex",
        ]
        + [
            str(day)
            for day
            in range(
                1,
                32,
            )
        ]
    )

    df = df[
        expected_columns
    ]

    # Debug preview
    preview = (
        processed.copy()
    )

    cv2.rectangle(
        preview,
        (
            table_left,
            table_top,
        ),
        (
            table_right,
            table_bottom,
        ),
        (
            255,
            0,
            0,
        ),
        3,
    )

    for x in day_lines:
        cv2.line(
            preview,
            (
                x,
                table_top,
            ),
            (
                x,
                table_bottom,
            ),
            (
                0,
                255,
                0,
            ),
            1,
        )

    for (
        y1,
        y2,
        *_,
    ) in student_rows:
        cv2.line(
            preview,
            (
                table_left,
                y1,
            ),
            (
                table_right,
                y1,
            ),
            (
                255,
                255,
                0,
            ),
            1,
        )

    return {
        "dataframe":
            df,
        "raw_words":
            raw_words,
        "preview":
            preview,
        "day_count":
            day_count,
        "verticals":
            verticals,
        "horizontals":
            horizontals,
    }


# ============================================================
# GENERIC GRID FALLBACK
# ============================================================

def find_interval(
    value,
    lines,
):
    for i in range(
        len(lines) - 1
    ):
        if (
            lines[i]
            <= value
            <= lines[i + 1]
        ):
            return i

    return None


def reconstruct_generic_grid(
    image,
    ocr,
):
    processed, gray = (
        preprocess(image)
    )

    (
        xs,
        ys,
        horizontal,
        vertical,
        binary,
    ) = detect_lines(
        gray
    )

    words = run_ocr(
        ocr,
        processed,
    )

    if (
        len(xs) < 2
        or len(ys) < 2
    ):
        raise RuntimeError(
            "No usable grid detected."
        )

    rows = len(ys) - 1
    cols = len(xs) - 1

    cells = {
        (r, c): []
        for r in range(rows)
        for c in range(cols)
    }

    for word in words:
        cx, cy = (
            word_center(
                word["box"]
            )
        )

        row = find_interval(
            cy,
            ys,
        )

        col = find_interval(
            cx,
            xs,
        )

        if (
            row is not None
            and col is not None
        ):
            cells[
                (
                    row,
                    col,
                )
            ].append(
                word
            )

    data = []

    for r in range(rows):
        row_data = []

        for c in range(cols):
            items = sorted(
                cells[
                    (
                        r,
                        c,
                    )
                ],
                key=lambda x: (
                    x["box"][1],
                    x["box"][0],
                ),
            )

            text = " ".join(
                item["text"]
                for item in items
            )

            row_data.append(
                text
            )

        data.append(
            row_data
        )

    df = pd.DataFrame(
        data
    )

    df = (
        df.replace(
            r"^\s*$",
            np.nan,
            regex=True,
        )
        .dropna(
            axis=0,
            how="all",
        )
        .dropna(
            axis=1,
            how="all",
        )
        .fillna("")
        .reset_index(
            drop=True
        )
    )

    preview = (
        processed.copy()
    )

    for x in xs:
        cv2.line(
            preview,
            (x, 0),
            (
                x,
                preview.shape[0],
            ),
            (
                0,
                255,
                0,
            ),
            1,
        )

    for y in ys:
        cv2.line(
            preview,
            (0, y),
            (
                preview.shape[1],
                y,
            ),
            (
                255,
                0,
                0,
            ),
            1,
        )

    return {
        "dataframe":
            df,
        "raw_words":
            words,
        "preview":
            preview,
    }


# ============================================================
# OLLAMA
# ============================================================

def ollama_json(
    prompt,
    model,
    timeout=1800,
):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [
                {
                    "role":
                        "system",
                    "content": (
                        "You classify OCR documents. "
                        "Never invent values. "
                        "Return valid JSON only."
                    ),
                },
                {
                    "role":
                        "user",
                    "content":
                        prompt,
                },
            ],
            "stream":
                False,
            "format":
                "json",
            "options": {
                "temperature": 0,
            },
        },
        timeout=timeout,
    )

    response.raise_for_status()

    content = (
        response
        .json()
        .get(
            "message",
            {},
        )
        .get(
            "content",
            "",
        )
    )

    return json.loads(
        content
    )


def detect_document_type(
    words,
    model,
):
    text = " ".join(
        item["text"]
        for item
        in words[:300]
    )

    prompt = f"""
Classify this scanned document.

Possible types:
- attendance_sheet
- financial_ledger
- inventory
- timesheet
- register
- generic_table

OCR TEXT:
{text}

Return JSON:

{{
    "document_type": "attendance_sheet",
    "confidence": 0.95
}}
"""

    try:
        return ollama_json(
            prompt,
            model,
        )
    except Exception:
        lower = text.lower()

        if (
            "attendance"
            in lower
            or "absent"
            in lower
            or "student full name"
            in lower
        ):
            return {
                "document_type":
                    "attendance_sheet",
                "confidence":
                    0.75,
            }

        return {
            "document_type":
                "generic_table",
            "confidence":
                0.2,
        }


# ============================================================
# EXCEL EXPORT
# ============================================================

def export_excel(df):
    output = io.BytesIO()

    wb = Workbook()

    ws = wb.active
    ws.title = (
        "Attendance"
    )

    thin = Side(
        style="thin",
        color="999999",
    )

    border = Border(
        left=thin,
        right=thin,
        top=thin,
        bottom=thin,
    )

    header_fill = (
        PatternFill(
            "solid",
            fgColor="D9EAF7",
        )
    )

    for col_index, column in enumerate(
        df.columns,
        start=1,
    ):
        cell = ws.cell(
            row=1,
            column=col_index,
            value=str(column),
        )

        cell.font = Font(
            bold=True
        )

        cell.fill = (
            header_fill
        )

        cell.border = (
            border
        )

        cell.alignment = (
            Alignment(
                horizontal=
                    "center",
                vertical=
                    "center",
                wrap_text=True,
            )
        )

    for row_index, row in enumerate(
        df.itertuples(
            index=False
        ),
        start=2,
    ):
        for col_index, value in enumerate(
            row,
            start=1,
        ):
            cell = ws.cell(
                row=row_index,
                column=col_index,
                value=str(value),
            )

            cell.border = (
                border
            )

            cell.alignment = (
                Alignment(
                    horizontal=
                        "center",
                    vertical=
                        "center",
                    wrap_text=True,
                )
            )

    for index, column in enumerate(
        df.columns,
        start=1,
    ):
        name = str(
            column
        )

        if (
            "Student Full Name"
            in name
        ):
            width = 30
        elif (
            name
            == "Serial No"
        ):
            width = 15
        elif (
            name
            == "No"
        ):
            width = 6
        elif (
            name
            == "Sex"
        ):
            width = 7
        elif name.isdigit():
            width = 5
        else:
            width = 12

        ws.column_dimensions[
            get_column_letter(
                index
            )
        ].width = width

    ws.freeze_panes = (
        "A2"
    )

    wb.save(
        output
    )

    output.seek(0)

    return output.getvalue()


# ============================================================
# FULL ATTENDANCE PDF
# ============================================================

def export_attendance_pdf(
    df,
    title="ABSENCE RECORDING SHEET",
    school_name="",
    grade="",
    class_name="",
    zone="",
    district="",
    year="",
    teacher_name="",
    month="",
    working_days="",
):
    output = io.BytesIO()

    page_size = (
        landscape(A3)
    )

    doc = SimpleDocTemplate(
        output,
        pagesize=page_size,
        leftMargin=5 * mm,
        rightMargin=5 * mm,
        topMargin=5 * mm,
        bottomMargin=5 * mm,
    )

    styles = (
        getSampleStyleSheet()
    )

    story = []

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    title_style = (
        styles["Heading1"]
    )

    title_style.alignment = 1
    title_style.fontSize = 14
    title_style.leading = 16

    story.append(
        Paragraph(
            title,
            title_style,
        )
    )

    story.append(
        Spacer(
            1,
            2 * mm,
        )
    )

    # --------------------------------------------------------
    # Instructions
    # --------------------------------------------------------

    instruction_text = (
        "Use for each month of the Academic Year. "
        "Mark attendance daily during school days. "
        "Use the appropriate absence code where relevant."
    )

    story.append(
        Paragraph(
            instruction_text,
            styles["Normal"],
        )
    )

    story.append(
        Spacer(
            1,
            2 * mm,
        )
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = [
        [
            f"School Name: {school_name}",
            f"Zone: {zone}",
            f"Teacher's Name: {teacher_name}",
        ],
        [
            f"Grade: {grade}",
            f"District: {district}",
            f"Month: {month}",
        ],
        [
            f"Class: {class_name}",
            f"Year: {year}",
            f"No. of working days: {working_days}",
        ],
    ]

    metadata_table = (
        Table(
            metadata,
            colWidths=[
                125 * mm,
                105 * mm,
                150 * mm,
            ],
        )
    )

    metadata_table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.black,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, -1),
                    "Helvetica",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),
            ]
        )
    )

    story.append(
        metadata_table
    )

    story.append(
        Spacer(
            1,
            2 * mm,
        )
    )

    # --------------------------------------------------------
    # Ensure required columns
    # --------------------------------------------------------

    temp_df = df.copy()

    required = [
        "No",
        "Serial No",
        "Student Full Name",
        "Sex",
    ]

    for column in required:
        if column not in temp_df.columns:
            temp_df[
                column
            ] = ""

    for day in range(
        1,
        32,
    ):
        column = str(day)

        if column not in temp_df.columns:
            temp_df[
                column
            ] = ""

    # --------------------------------------------------------
    # Total absent
    # --------------------------------------------------------

    def absent_total(
        row,
    ):
        total = 0

        for day in range(
            1,
            32,
        ):
            value = str(
                row.get(
                    str(day),
                    "",
                )
            ).strip().upper()

            if value in {
                "A",
                "ABSENT",
            }:
                total += 1

        return total

    temp_df[
        "Total Absent Days"
    ] = temp_df.apply(
        absent_total,
        axis=1,
    )

    columns = (
        [
            "No",
            "Serial No",
            "Student Full Name",
            "Sex",
        ]
        + [
            str(day)
            for day
            in range(
                1,
                32,
            )
        ]
        + [
            "Total Absent Days"
        ]
    )

    temp_df = (
        temp_df[
            columns
        ]
    )

    # --------------------------------------------------------
    # Main table header
    # --------------------------------------------------------

    header = [
        "No",
        "Serial No\n(Same as in\nenrollment form)",
        "Student Full Name\n(First Name, Father Name,\nGrandfather Name)",
        "Sex\n(F/M)",
    ]

    header.extend(
        [
            str(day)
            for day
            in range(
                1,
                32,
            )
        ]
    )

    header.append(
        "Total\nAbsent\nDays"
    )

    table_data = [
        header
    ]

    for _, row in (
        temp_df.iterrows()
    ):
        table_data.append(
            [
                str(
                    row[column]
                )
                if not pd.isna(
                    row[column]
                )
                else ""
                for column
                in columns
            ]
        )

    # --------------------------------------------------------
    # Add blank printable rows
    # --------------------------------------------------------

    minimum_rows = 20

    while (
        len(table_data) - 1
        < minimum_rows
    ):
        row_number = (
            len(table_data)
        )

        blank = [
            ""
            for _
            in columns
        ]

        blank[0] = str(
            row_number
        )

        table_data.append(
            blank
        )

    # --------------------------------------------------------
    # Widths
    # --------------------------------------------------------

    column_widths = [
        8 * mm,
        23 * mm,
        62 * mm,
        10 * mm,
    ]

    column_widths.extend(
        [
            7.0 * mm
            for _
            in range(31)
        ]
    )

    column_widths.append(
        17 * mm
    )

    # --------------------------------------------------------
    # Main attendance table
    # --------------------------------------------------------

    attendance_table = Table(
        table_data,
        colWidths=column_widths,
        repeatRows=1,
        rowHeights=[
            14 * mm
        ]
        + [
            7.4 * mm
            for _
            in range(
                len(table_data)
                - 1
            )
        ],
    )

    attendance_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#E5E5E5"
                    ),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (-1, -1),
                    "Helvetica",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, 0),
                    5.2,
                ),
                (
                    "FONTSIZE",
                    (0, 1),
                    (-1, -1),
                    6,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.35,
                    colors.black,
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "ALIGN",
                    (2, 1),
                    (2, -1),
                    "LEFT",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    1,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    1,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    1,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    1,
                ),
            ]
        )
    )

    story.append(
        attendance_table
    )

    # --------------------------------------------------------
    # Daily absent totals
    # --------------------------------------------------------

    totals = [
        "Total Absent Students for the Day",
        "",
        "",
        "",
    ]

    for day in range(
        1,
        32,
    ):
        column = str(day)

        count = 0

        for value in (
            temp_df[column]
        ):
            value = str(
                value
            ).strip().upper()

            if value in {
                "A",
                "ABSENT",
            }:
                count += 1

        totals.append(
            str(count)
            if count
            else ""
        )

    totals.append("")

    totals_table = Table(
        [
            totals
        ],
        colWidths=column_widths,
        rowHeights=[
            7 * mm
        ],
    )

    totals_table.setStyle(
        TableStyle(
            [
                (
                    "SPAN",
                    (0, 0),
                    (3, 0),
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.35,
                    colors.black,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, -1),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    5.5,
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
            ]
        )
    )

    story.append(
        totals_table
    )

    story.append(
        Spacer(
            1,
            2 * mm,
        )
    )

    # --------------------------------------------------------
    # Absence codes legend
    # --------------------------------------------------------

    legend = [
        [
            "Reason",
            "CODE",
            "Reason",
            "CODE",
            "Reason",
            "CODE",
            "Reason",
            "CODE",
        ],
        [
            "Unknown Absence",
            "U",
            "Chores",
            "C",
            "Not Interested",
            "NI",
            "Local / Public Holiday",
            "L",
        ],
        [
            "Illness",
            "H",
            "No School Materials",
            "NSM",
            "Other",
            "O",
            "Weekend",
            "X",
        ],
    ]

    legend_table = Table(
        legend,
        colWidths=[
            39 * mm,
            12 * mm,
            39 * mm,
            12 * mm,
            39 * mm,
            12 * mm,
            39 * mm,
            12 * mm,
        ],
    )

    legend_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#E5E5E5"
                    ),
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.35,
                    colors.black,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
            ]
        )
    )

    story.append(
        legend_table
    )

    doc.build(
        story
    )

    output.seek(0)

    return output.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header(
        "Settings"
    )

    ollama_model = st.text_input(
        "Ollama model",
        value=DEFAULT_OLLAMA_MODEL,
    )

    force_attendance = st.checkbox(
        "Force Attendance Mode",
        value=True,
    )

    use_ollama = st.checkbox(
        "Use Ollama document classification",
        value=True,
    )

    show_grid = st.checkbox(
        "Show detected structure",
        value=True,
    )

    st.info(
        "For your current attendance sheets, "
        "keep Force Attendance Mode enabled."
    )


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = (
    st.file_uploader(
        "Upload attendance sheet / PDF",
        type=[
            "pdf",
            "png",
            "jpg",
            "jpeg",
            "webp",
        ],
    )
)


if uploaded_file:
    pages = (
        uploaded_file_to_images(
            uploaded_file
        )
    )

    st.session_state.pages = (
        pages
    )

    st.success(
        f"Loaded {len(pages)} page(s)."
    )

    if st.button(
        "🚀 Extract Attendance",
        type="primary",
        width="stretch",
    ):
        try:
            ocr = load_ocr()
        except Exception as exc:
            st.error(
                f"Could not start PaddleOCR: {exc}"
            )
            st.stop()

        results = []

        progress = (
            st.progress(0)
        )

        status = st.empty()

        for page_index, page in enumerate(
            pages
        ):
            status.write(
                f"Processing page "
                f"{page_index + 1}/"
                f"{len(pages)}..."
            )

            try:
                prepared, _ = (
                    preprocess(page)
                )

                first_words = (
                    run_ocr(
                        ocr,
                        prepared,
                    )
                )

                if force_attendance:
                    document_type = (
                        "attendance_sheet"
                    )

                elif use_ollama:
                    detection = (
                        detect_document_type(
                            first_words,
                            ollama_model,
                        )
                    )

                    document_type = (
                        detection.get(
                            "document_type",
                            "generic_table",
                        )
                    )

                else:
                    text = " ".join(
                        item["text"].lower()
                        for item
                        in first_words
                    )

                    if (
                        "attendance"
                        in text
                        or "absent"
                        in text
                    ):
                        document_type = (
                            "attendance_sheet"
                        )
                    else:
                        document_type = (
                            "generic_table"
                        )

                if (
                    document_type
                    == "attendance_sheet"
                ):
                    result = (
                        extract_attendance_sheet(
                            page,
                            ocr,
                        )
                    )
                else:
                    result = (
                        reconstruct_generic_grid(
                            page,
                            ocr,
                        )
                    )

                result[
                    "document_type"
                ] = document_type

                results.append(
                    result
                )

            except Exception as exc:
                st.error(
                    f"Page "
                    f"{page_index + 1} "
                    f"failed: {exc}"
                )

                results.append(
                    {
                        "dataframe":
                            pd.DataFrame(),
                        "raw_words":
                            [],
                        "preview":
                            page,
                        "document_type":
                            "failed",
                    }
                )

            progress.progress(
                (
                    page_index + 1
                )
                / len(pages)
            )

        st.session_state.results = (
            results
        )

        status.success(
            "Extraction complete."
        )


# ============================================================
# RESULTS
# ============================================================

if st.session_state.results:
    st.divider()

    valid_results = [
        result
        for result
        in st.session_state.results
        if not result[
            "dataframe"
        ].empty
    ]

    if not valid_results:
        st.error(
            "No usable attendance data was extracted."
        )
        st.stop()

    st.header(
        "Document Preview"
    )

    selected_page = (
        st.selectbox(
            "Page",
            range(
                1,
                len(
                    st.session_state.results
                )
                + 1
            ),
        )
    )

    index = (
        selected_page - 1
    )

    selected_result = (
        st.session_state.results[
            index
        ]
    )

    left, right = (
        st.columns(2)
    )

    with left:
        st.markdown(
            "### Original"
        )

        st.image(
            st.session_state.pages[
                index
            ],
            width="stretch",
        )

    with right:
        st.markdown(
            "### Detected Structure"
        )

        if show_grid:
            st.image(
                selected_result[
                    "preview"
                ],
                width="stretch",
            )
        else:
            st.info(
                "Structure preview disabled."
            )

    st.markdown(
        f"**Document type:** "
        f"`{selected_result['document_type']}`"
    )

    if (
        "day_count"
        in selected_result
    ):
        st.markdown(
            f"**Detected day columns:** "
            f"{selected_result['day_count']}"
        )

    with st.expander(
        "Raw OCR"
    ):
        raw_df = pd.DataFrame(
            [
                {
                    "Text":
                        item["text"],
                    "Confidence":
                        item[
                            "confidence"
                        ],
                    "Box":
                        str(
                            item["box"]
                        ),
                }
                for item
                in selected_result[
                    "raw_words"
                ]
            ]
        )

        st.dataframe(
            raw_df,
            width="stretch",
        )

    # --------------------------------------------------------
    # Combine pages
    # --------------------------------------------------------

    dataframes = [
        result[
            "dataframe"
        ]
        for result
        in valid_results
    ]

    first_columns = list(
        dataframes[0].columns
    )

    if all(
        list(df.columns)
        == first_columns
        for df
        in dataframes
    ):
        combined_df = pd.concat(
            dataframes,
            ignore_index=True,
        )
    else:
        combined_df = (
            dataframes[0]
        )

    # --------------------------------------------------------
    # Editable table
    # --------------------------------------------------------

    st.header(
        "Extracted Attendance"
    )

    st.caption(
        "Review and correct OCR values before export."
    )

    edited_df = st.data_editor(
        combined_df,
        width="stretch",
        height=650,
        num_rows="dynamic",
        key="attendance_editor",
    )

    # --------------------------------------------------------
    # Attendance summary
    # --------------------------------------------------------

    day_columns = [
        column
        for column
        in edited_df.columns
        if str(column).isdigit()
    ]

    if day_columns:
        summary = []

        for row_index, row in (
            edited_df.iterrows()
        ):
            marks = [
                str(
                    row[column]
                ).strip()
                for column
                in day_columns
            ]

            marked = sum(
                bool(value)
                for value
                in marks
            )

            absent = sum(
                value.upper()
                in {
                    "A",
                    "ABSENT",
                }
                for value
                in marks
            )

            summary.append(
                {
                    "Student":
                        row.get(
                            "Student Full Name",
                            f"Row {row_index + 1}",
                        ),
                    "Marked Days":
                        marked,
                    "Absent Days":
                        absent,
                    "Blank Days":
                        len(day_columns)
                        - marked,
                }
            )

        st.subheader(
            "Attendance Validation"
        )

        st.dataframe(
            pd.DataFrame(
                summary
            ),
            width="stretch",
        )

    # ========================================================
    # PDF INFORMATION
    # ========================================================

    st.divider()

    st.header(
        "Printable PDF Information"
    )

    c1, c2, c3 = (
        st.columns(3)
    )

    with c1:
        school_name = (
            st.text_input(
                "School Name"
            )
        )

        grade = (
            st.text_input(
                "Grade"
            )
        )

        class_name = (
            st.text_input(
                "Class"
            )
        )

    with c2:
        zone = (
            st.text_input(
                "Zone"
            )
        )

        district = (
            st.text_input(
                "District"
            )
        )

        year = (
            st.text_input(
                "Year"
            )
        )

    with c3:
        teacher_name = (
            st.text_input(
                "Teacher Name"
            )
        )

        month = (
            st.text_input(
                "Month"
            )
        )

        working_days = (
            st.text_input(
                "No. of Working Days"
            )
        )

    pdf_title = (
        st.text_input(
            "PDF Title",
            value=
                "ABSENCE RECORDING SHEET",
        )
    )

    # ========================================================
    # EXPORTS
    # ========================================================

    st.divider()

    st.header(
        "Export"
    )

    excel_bytes = (
        export_excel(
            edited_df
        )
    )

    csv_bytes = (
        edited_df
        .to_csv(
            index=False
        )
        .encode(
            "utf-8-sig"
        )
    )

    pdf_bytes = (
        export_attendance_pdf(
            edited_df,
            title=pdf_title,
            school_name=
                school_name,
            grade=
                grade,
            class_name=
                class_name,
            zone=
                zone,
            district=
                district,
            year=
                year,
            teacher_name=
                teacher_name,
            month=
                month,
            working_days=
                working_days,
        )
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:
        st.download_button(
            "⬇️ Download Excel",
            data=excel_bytes,
            file_name=
                "digitized_attendance.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            width="stretch",
        )

    with col2:
        st.download_button(
            "⬇️ Download CSV",
            data=csv_bytes,
            file_name=
                "digitized_attendance.csv",
            mime="text/csv",
            width="stretch",
        )

    with col3:
        st.download_button(
            "⬇️ Download Printable PDF",
            data=pdf_bytes,
            file_name=
                "attendance_recording_sheet.pdf",
            mime="application/pdf",
            width="stretch",
        )
import os

# ============================================================
# IMPORTANT: MUST BE SET BEFORE importing Paddle/PaddleOCR
# ============================================================

# Disable OneDNN to avoid context errors on Windows
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# ============================================================
# IMPORTS
# ============================================================

import io
import json
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from paddleocr import PaddleOCR


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Document Digitizer",
    page_icon="📄",
    layout="wide",
)

st.title("Document Digitizer")
st.caption(
    "PaddleOCR document extraction with grid-aware table reconstruction"
)


# ============================================================
# OCR MODEL
# ============================================================

@st.cache_resource
def load_ocr():
    return PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )


try:
    ocr = load_ocr()
except Exception as exc:
    st.error(f"Could not initialize PaddleOCR: {exc}")
    st.code(traceback.format_exc())
    st.stop()


# ============================================================
# BASIC IMAGE HELPERS
# ============================================================

def pil_to_cv(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    arr = np.asarray(image)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def upscale_image(
    image: np.ndarray,
    minimum_long_side: int = 1800,
) -> np.ndarray:

    h, w = image.shape[:2]
    long_side = max(h, w)

    if long_side >= minimum_long_side:
        return image

    scale = minimum_long_side / long_side

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )


def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Mild enhancement intended to preserve table borders.
    """

    image = upscale_image(image)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.fastNlMeansDenoising(
        gray,
        None,
        5,
        7,
        21,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    gray = clahe.apply(gray)

    return cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR,
    )


# ============================================================
# PDF RENDERING
# ============================================================

def pdf_to_images(
    pdf_bytes: bytes,
    dpi: int = 220,
):

    try:
        import fitz
    except ImportError:
        raise RuntimeError(
            "PyMuPDF is required. Run: pip install pymupdf"
        )

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    pages = []

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page_index in range(len(document)):
        page = document.load_page(page_index)

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        pil_image = Image.frombytes(
            "RGB",
            (pixmap.width, pixmap.height),
            pixmap.samples,
        )

        pages.append(pil_image)

    document.close()

    return pages


# ============================================================
# PADDLEOCR OUTPUT PARSING
# ============================================================

def polygon_to_rect(polygon):
    points = np.asarray(
        polygon,
        dtype=np.float32,
    )

    if points.ndim != 2:
        raise ValueError("Unexpected polygon shape")

    if points.shape[1] < 2:
        raise ValueError("Polygon does not contain XY coordinates")

    xs = points[:, 0]
    ys = points[:, 1]

    x1 = float(np.min(xs))
    y1 = float(np.min(ys))
    x2 = float(np.max(xs))
    y2 = float(np.max(ys))

    return x1, y1, x2, y2


def paddle_result_to_dict(result_item):
    if isinstance(result_item, dict):
        return result_item

    if hasattr(result_item, "json"):
        try:
            value = result_item.json

            if callable(value):
                value = value()

            if isinstance(value, str):
                return json.loads(value)

            if isinstance(value, dict):
                return value

        except Exception:
            pass

    if hasattr(result_item, "res"):
        try:
            value = result_item.res

            if isinstance(value, dict):
                return value
        except Exception:
            pass

    try:
        return dict(result_item)
    except Exception:
        return {}


def parse_paddle_output(result):
    entries = []

    if result is None:
        return entries

    if not isinstance(result, (list, tuple)):
        result = [result]

    for result_item in result:

        data = paddle_result_to_dict(
            result_item
        )

        if not data:
            continue

        if (
            "res" in data
            and isinstance(data["res"], dict)
        ):
            data = data["res"]

        texts = (
            data.get("rec_texts")
            or data.get("texts")
            or []
        )

        scores = (
            data.get("rec_scores")
            or data.get("scores")
            or []
        )

        polygons = (
            data.get("rec_polys")
            or data.get("dt_polys")
            or data.get("boxes")
            or []
        )

        if isinstance(texts, str):
            texts = [texts]

        for index, text in enumerate(texts):

            if text is None:
                continue

            text = str(text).strip()

            if not text:
                continue

            confidence = 1.0

            if index < len(scores):
                try:
                    confidence = float(
                        scores[index]
                    )
                except Exception:
                    confidence = 1.0

            if index >= len(polygons):
                continue

            try:
                x1, y1, x2, y2 = polygon_to_rect(
                    polygons[index]
                )
            except Exception:
                continue

            entries.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": (x1 + x2) / 2,
                    "cy": (y1 + y2) / 2,
                    "width": max(x2 - x1, 1),
                    "height": max(y2 - y1, 1),
                }
            )

    return entries


def run_ocr(image: np.ndarray):
    """
    PaddleOCR 3.x.

    Use the __call__ method (call the object directly).
    
    Handles OneDNN context errors on Windows by returning empty results
    instead of crashing the application.
    """

    try:
        result = ocr(image)
    except RuntimeError as e:
        # Handle OneDNN context errors gracefully
        if "OneDnnContext" in str(e) or "does not have the input" in str(e):
            st.warning(
                "⚠️ OneDNN error occurred during OCR processing. "
                "This can happen on Windows with certain hardware configurations. "
                "The page will be processed with empty results - you can manually verify the data."
            )
            return []
        else:
            raise

    return parse_paddle_output(result)


# ============================================================
# GRID / CELL DETECTION
# ============================================================

def detect_table_mask(image: np.ndarray):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        10,
    )

    height, width = binary.shape

    horizontal_length = max(
        20,
        width // 30,
    )

    vertical_length = max(
        20,
        height // 30,
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_length, 1),
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, vertical_length),
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )

    combined = cv2.add(
        horizontal,
        vertical,
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3),
    )

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )

    return combined


def detect_cells(image: np.ndarray):
    table_mask = detect_table_mask(image)

    contours, _ = cv2.findContours(
        table_mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    image_height, image_width = image.shape[:2]

    cells = []

    for contour in contours:

        x, y, width, height = cv2.boundingRect(
            contour
        )

        if width < 20:
            continue

        if height < 12:
            continue

        # Whole-page container
        if (
            width >= image_width * 0.95
            and height >= image_height * 0.95
        ):
            continue

        # Huge regions that clearly aren't normal cells
        if width >= image_width * 0.95:
            continue

        if height >= image_height * 0.6:
            continue

        cells.append(
            {
                "x1": int(x),
                "y1": int(y),
                "x2": int(x + width),
                "y2": int(y + height),
                "cx": float(x + width / 2),
                "cy": float(y + height / 2),
                "width": int(width),
                "height": int(height),
            }
        )

    return cells, table_mask


# ============================================================
# CELL DEDUPLICATION
# ============================================================

def rectangle_iou(a, b):
    left = max(a["x1"], b["x1"])
    top = max(a["y1"], b["y1"])
    right = min(a["x2"], b["x2"])
    bottom = min(a["y2"], b["y2"])

    intersection_width = max(
        0,
        right - left,
    )

    intersection_height = max(
        0,
        bottom - top,
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    area_a = (
        (a["x2"] - a["x1"])
        * (a["y2"] - a["y1"])
    )

    area_b = (
        (b["x2"] - b["x1"])
        * (b["y2"] - b["y1"])
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0

    return intersection / union


def remove_duplicate_cells(cells):
    cells = sorted(
        cells,
        key=lambda cell: (
            cell["y1"],
            cell["x1"],
            cell["width"] * cell["height"],
        ),
    )

    kept = []

    for cell in cells:

        duplicate = False

        for existing in kept:

            iou = rectangle_iou(
                cell,
                existing,
            )

            nearly_same = (
                abs(cell["x1"] - existing["x1"]) <= 4
                and
                abs(cell["y1"] - existing["y1"]) <= 4
                and
                abs(cell["x2"] - existing["x2"]) <= 4
                and
                abs(cell["y2"] - existing["y2"]) <= 4
            )

            if iou >= 0.92 or nearly_same:
                duplicate = True
                break

        if not duplicate:
            kept.append(cell)

    return kept


# ============================================================
# ROW GROUPING
# ============================================================

def group_cells_into_rows(cells):
    if not cells:
        return []

    heights = [
        cell["height"]
        for cell in cells
        if cell["height"] > 0
    ]

    median_height = (
        float(np.median(heights))
        if heights
        else 20
    )

    row_threshold = max(
        8,
        median_height * 0.45,
    )

    cells = sorted(
        cells,
        key=lambda cell: (
            cell["cy"],
            cell["x1"],
        ),
    )

    rows = []

    for cell in cells:

        best_row_index = None
        best_distance = None

        for index, row in enumerate(rows):

            row_center = float(
                np.mean(
                    [
                        item["cy"]
                        for item in row
                    ]
                )
            )

            distance = abs(
                cell["cy"]
                - row_center
            )

            if distance <= row_threshold:

                if (
                    best_distance is None
                    or distance < best_distance
                ):
                    best_distance = distance
                    best_row_index = index

        if best_row_index is None:
            rows.append([cell])
        else:
            rows[
                best_row_index
            ].append(cell)

    for row in rows:
        row.sort(
            key=lambda cell:
            cell["x1"]
        )

    rows.sort(
        key=lambda row:
        np.mean(
            [
                cell["cy"]
                for cell in row
            ]
        )
    )

    return rows


# ============================================================
# OCR TEXT INSIDE CELLS
# ============================================================

def extract_cell_text(cell, entries):
    matching_entries = []

    for entry in entries:

        center_x = float(
            entry["cx"]
        )

        center_y = float(
            entry["cy"]
        )

        if (
            cell["x1"] <= center_x <= cell["x2"]
            and
            cell["y1"] <= center_y <= cell["y2"]
        ):
            matching_entries.append(
                entry
            )

    matching_entries.sort(
        key=lambda item: (
            item["cy"],
            item["x1"],
        )
    )

    return " ".join(
        entry["text"]
        for entry in matching_entries
    ).strip()


# ============================================================
# CONTOUR-BASED TABLE RECONSTRUCTION
# ============================================================

def reconstruct_using_cells(
    image,
    entries,
):

    cells, table_mask = detect_cells(
        image
    )

    if not cells:
        return None, table_mask, []

    cells = remove_duplicate_cells(
        cells
    )

    rows = group_cells_into_rows(
        cells
    )

    # Ignore clearly bogus single-cell "tables"
    valid_rows = [
        row
        for row in rows
        if len(row) >= 2
    ]

    if not valid_rows:
        return None, table_mask, cells

    max_columns = max(
        len(row)
        for row in valid_rows
    )

    matrix = []

    for row in valid_rows:

        values = []

        for cell in row:

            value = extract_cell_text(
                cell,
                entries,
            )

            values.append(
                value
            )

        while len(values) < max_columns:
            values.append("")

        matrix.append(values)

    dataframe = pd.DataFrame(
        matrix
    )

    dataframe = dataframe[
        dataframe.apply(
            lambda row:
            row.astype(str)
            .str.strip()
            .ne("")
            .any(),
            axis=1,
        )
    ]

    dataframe = dataframe.loc[
        :,
        dataframe.apply(
            lambda column:
            column.astype(str)
            .str.strip()
            .ne("")
            .any()
        )
    ]

    if dataframe.empty:
        return None, table_mask, cells

    return (
        dataframe.reset_index(
            drop=True
        ),
        table_mask,
        cells,
    )


# ============================================================
# POSITION-BASED FALLBACK
# ============================================================

def group_ocr_into_rows(entries):
    if not entries:
        return []

    heights = [
        entry["height"]
        for entry in entries
        if entry["height"] > 0
    ]

    median_height = (
        float(np.median(heights))
        if heights
        else 20
    )

    threshold = max(
        10,
        median_height * 0.65,
    )

    sorted_entries = sorted(
        entries,
        key=lambda entry: (
            entry["cy"],
            entry["x1"],
        ),
    )

    rows = []

    for entry in sorted_entries:

        best_index = None
        best_distance = None

        for index, row in enumerate(rows):

            row_center = float(
                np.mean(
                    [
                        item["cy"]
                        for item in row
                    ]
                )
            )

            distance = abs(
                entry["cy"]
                - row_center
            )

            if distance <= threshold:

                if (
                    best_distance is None
                    or distance < best_distance
                ):
                    best_distance = distance
                    best_index = index

        if best_index is None:
            rows.append(
                [entry]
            )
        else:
            rows[
                best_index
            ].append(entry)

    for row in rows:
        row.sort(
            key=lambda entry:
            entry["x1"]
        )

    rows.sort(
        key=lambda row:
        np.mean(
            [
                item["cy"]
                for item in row
            ]
        )
    )

    return rows


def infer_column_centers(entries):
    if not entries:
        return []

    sorted_entries = sorted(
        entries,
        key=lambda entry:
        entry["cx"]
    )

    widths = [
        entry["width"]
        for entry in entries
        if entry["width"] > 0
    ]

    median_width = (
        float(np.median(widths))
        if widths
        else 80
    )

    threshold = max(
        45,
        median_width * 0.55,
    )

    groups = []

    for entry in sorted_entries:

        x = float(
            entry["cx"]
        )

        if not groups:
            groups.append([x])
            continue

        current_center = float(
            np.mean(
                groups[-1]
            )
        )

        if abs(x - current_center) <= threshold:
            groups[-1].append(x)
        else:
            groups.append([x])

    return [
        float(
            np.mean(group)
        )
        for group in groups
    ]


def reconstruct_from_positions(entries):
    rows = group_ocr_into_rows(
        entries
    )

    if not rows:
        return None

    column_centers = infer_column_centers(
        entries
    )

    if not column_centers:
        return None

    matrix = []

    for row in rows:

        values = [
            ""
            for _ in column_centers
        ]

        for entry in row:

            distances = [
                abs(
                    float(entry["cx"])
                    - center
                )
                for center
                in column_centers
            ]

            column_index = int(
                np.argmin(
                    distances
                )
            )

            if values[column_index]:
                values[column_index] += (
                    " "
                    + entry["text"]
                )
            else:
                values[column_index] = (
                    entry["text"]
                )

        matrix.append(values)

    dataframe = pd.DataFrame(
        matrix
    )

    dataframe = dataframe[
        dataframe.apply(
            lambda row:
            row.astype(str)
            .str.strip()
            .ne("")
            .any(),
            axis=1,
        )
    ]

    dataframe = dataframe.loc[
        :,
        dataframe.apply(
            lambda column:
            column.astype(str)
            .str.strip()
            .ne("")
            .any()
        )
    ]

    if dataframe.empty:
        return None

    return dataframe.reset_index(
        drop=True
    )


# ============================================================
# MAIN TABLE RECONSTRUCTION
# ============================================================

def reconstruct_table(
    image,
    entries,
):

    debug = {
        "method": None,
        "mask": None,
        "cells": [],
        "grid_error": None,
        "fallback_error": None,
    }

    try:

        dataframe, mask, cells = (
            reconstruct_using_cells(
                image,
                entries,
            )
        )

        debug["mask"] = mask
        debug["cells"] = cells

        if (
            dataframe is not None
            and not dataframe.empty
        ):
            debug["method"] = (
                "Contour-based grid reconstruction"
            )

            return dataframe, debug

    except Exception as exc:

        debug["grid_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    try:

        dataframe = reconstruct_from_positions(
            entries
        )

        if (
            dataframe is not None
            and not dataframe.empty
        ):
            debug["method"] = (
                "OCR coordinate reconstruction"
            )

            return dataframe, debug

    except Exception as exc:

        debug["fallback_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    return None, debug


# ============================================================
# DEBUG DRAWING
# ============================================================

def draw_ocr_boxes(
    image,
    entries,
):

    result = image.copy()

    for entry in entries:

        x1 = int(entry["x1"])
        y1 = int(entry["y1"])
        x2 = int(entry["x2"])
        y2 = int(entry["y2"])

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            (0, 200, 0),
            2,
        )

        cv2.putText(
            result,
            entry["text"][:30],
            (
                x1,
                max(
                    15,
                    y1 - 5,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return result


def draw_cells(
    image,
    cells,
):

    result = image.copy()

    for index, cell in enumerate(
        cells,
        start=1,
    ):

        cv2.rectangle(
            result,
            (
                int(cell["x1"]),
                int(cell["y1"]),
            ),
            (
                int(cell["x2"]),
                int(cell["y2"]),
            ),
            (0, 0, 255),
            2,
        )

        cv2.putText(
            result,
            str(index),
            (
                int(cell["x1"]) + 3,
                int(cell["y1"]) + 15,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return result


# ============================================================
# EXPORT
# ============================================================

def dataframe_to_csv(dataframe):
    return dataframe.to_csv(
        index=False,
        header=False,
    ).encode("utf-8-sig")


def tables_to_excel(tables):
    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:

        for page_number, dataframe in tables:

            dataframe.to_excel(
                writer,
                sheet_name=(
                    f"Page_{page_number}"
                ),
                index=False,
                header=False,
            )

    output.seek(0)

    return output.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("OCR Settings")

    pdf_dpi = st.slider(
        "PDF DPI",
        min_value=150,
        max_value=350,
        value=220,
        step=10,
    )

    enhance_scan = st.checkbox(
        "Enhance image",
        value=True,
    )

    confidence_threshold = st.slider(
        "Minimum OCR confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.05,
    )

    show_ocr_debug = st.checkbox(
        "Show OCR boxes",
        value=True,
    )

    show_grid_debug = st.checkbox(
        "Show detected table cells",
        value=True,
    )

    show_table_mask = st.checkbox(
        "Show table-line mask",
        value=False,
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload document",
    type=[
        "pdf",
        "png",
        "jpg",
        "jpeg",
        "webp",
        "bmp",
        "tif",
        "tiff",
    ],
)


if uploaded_file is None:
    st.info(
        "Upload a PDF or image to begin."
    )
    st.stop()


file_bytes = uploaded_file.read()

extension = Path(
    uploaded_file.name
).suffix.lower()


# ============================================================
# DOCUMENT LOADING
# ============================================================

try:

    if extension == ".pdf":

        with st.spinner(
            "Rendering PDF pages..."
        ):
            pages = pdf_to_images(
                file_bytes,
                dpi=pdf_dpi,
            )

    else:

        image = Image.open(
            io.BytesIO(
                file_bytes
            )
        ).convert(
            "RGB"
        )

        pages = [image]

except Exception as exc:

    st.error(
        f"Could not read the document: {exc}"
    )

    st.code(
        traceback.format_exc()
    )

    st.stop()


st.success(
    f"Loaded {len(pages)} page(s)."
)


# ============================================================
# EXTRACTION
# ============================================================

all_tables = []


for page_number, pil_page in enumerate(
    pages,
    start=1,
):

    st.divider()

    st.subheader(
        f"Page {page_number}"
    )

    try:

        original_image = pil_to_cv(
            pil_page
        )

        if enhance_scan:
            processed_image = enhance_image(
                original_image
            )
        else:
            processed_image = (
                original_image
            )

        page_col, metric_col = st.columns(
            [3, 1]
        )

        with page_col:

            st.image(
                cv_to_rgb(
                    processed_image
                ),
                caption=(
                    f"Page {page_number}"
                ),
                width="stretch",
            )

        # ====================================================
        # OCR
        # ====================================================

        try:
            with st.spinner(
                f"Running OCR on page {page_number}..."
            ):

                entries = run_ocr(
                    processed_image
                )
        except Exception as e:
            st.error(f"OCR failed on page {page_number}: {e}")
            entries = []

        entries = [
            entry
            for entry in entries
            if (
                float(entry["confidence"])
                >= confidence_threshold
            )
        ]

        with metric_col:

            st.metric(
                "OCR regions",
                len(entries),
            )

            if entries:

                average_confidence = float(
                    np.mean(
                        [
                            entry["confidence"]
                            for entry in entries
                        ]
                    )
                )

                st.metric(
                    "Average confidence",
                    f"{average_confidence:.1%}",
                )

            else:

                st.metric(
                    "Average confidence",
                    "—",
                )

        # ====================================================
        # OCR PREVIEW
        # ====================================================

        if (
            show_ocr_debug
            and entries
        ):

            with st.expander(
                "OCR bounding boxes"
            ):

                boxed_image = draw_ocr_boxes(
                    processed_image,
                    entries,
                )

                st.image(
                    cv_to_rgb(
                        boxed_image
                    ),
                    width="stretch",
                )

        # ====================================================
        # RAW OCR OUTPUT
        # ====================================================

        with st.expander(
            "Raw OCR results"
        ):

            if entries:

                raw_dataframe = pd.DataFrame(
                    [
                        {
                            "Text":
                                entry["text"],

                            "Confidence":
                                round(
                                    float(
                                        entry[
                                            "confidence"
                                        ]
                                    ),
                                    4,
                                ),

                            "X1":
                                round(
                                    float(
                                        entry["x1"]
                                    ),
                                    1,
                                ),

                            "Y1":
                                round(
                                    float(
                                        entry["y1"]
                                    ),
                                    1,
                                ),

                            "X2":
                                round(
                                    float(
                                        entry["x2"]
                                    ),
                                    1,
                                ),

                            "Y2":
                                round(
                                    float(
                                        entry["y2"]
                                    ),
                                    1,
                                ),
                        }
                        for entry in entries
                    ]
                )

                st.dataframe(
                    raw_dataframe,
                    width="stretch",
                    hide_index=True,
                )

                raw_text = "\n".join(
                    entry["text"]
                    for entry in entries
                )

                st.text_area(
                    "Plain OCR text",
                    value=raw_text,
                    height=250,
                    key=(
                        f"ocr_text_"
                        f"{page_number}"
                    ),
                )

            else:

                st.warning(
                    "PaddleOCR did not detect any usable text."
                )

        # ====================================================
        # TABLE RECONSTRUCTION
        # ====================================================

        if entries:

            dataframe, debug = (
                reconstruct_table(
                    processed_image,
                    entries,
                )
            )

        else:

            dataframe = None

            debug = {
                "method": None,
                "mask": None,
                "cells": [],
                "grid_error": None,
                "fallback_error": None,
            }

        # ====================================================
        # TABLE DEBUGGING
        # ====================================================

        if (
            show_table_mask
            and debug["mask"] is not None
        ):

            with st.expander(
                "Detected table-line mask"
            ):

                st.image(
                    debug["mask"],
                    width="stretch",
                    clamp=True,
                )

        if (
            show_grid_debug
            and debug["cells"]
        ):

            with st.expander(
                "Detected table cells"
            ):

                cell_preview = draw_cells(
                    processed_image,
                    debug["cells"],
                )

                st.image(
                    cv_to_rgb(
                        cell_preview
                    ),
                    width="stretch",
                )

                st.caption(
                    f"Detected "
                    f"{len(debug['cells'])} "
                    f"candidate cells."
                )

        # ====================================================
        # ERROR INFO
        # ====================================================

        if debug["grid_error"]:

            st.warning(
                "Grid reconstruction failed, "
                "so the app attempted coordinate-based reconstruction."
            )

            with st.expander(
                "Grid error details"
            ):

                st.code(
                    debug["grid_error"]
                )

        if debug["fallback_error"]:

            with st.expander(
                "Coordinate reconstruction error"
            ):

                st.code(
                    debug[
                        "fallback_error"
                    ]
                )

        # ====================================================
        # DISPLAY TABLE
        # ====================================================

        if (
            dataframe is None
            or dataframe.empty
        ):

            st.warning(
                "OCR completed, but no reliable table "
                "structure could be reconstructed "
                f"from page {page_number}."
            )

            continue

        st.markdown(
            "### Extracted Table"
        )

        st.caption(
            "Reconstruction method: "
            f"{debug['method']}"
        )

        edited_dataframe = st.data_editor(
            dataframe,
            width="stretch",
            num_rows="dynamic",
            key=(
                f"table_editor_"
                f"{page_number}"
            ),
        )

        all_tables.append(
            (
                page_number,
                edited_dataframe,
            )
        )

        st.download_button(
            label=(
                f"Download Page "
                f"{page_number} as CSV"
            ),
            data=dataframe_to_csv(
                edited_dataframe
            ),
            file_name=(
                f"page_{page_number}_"
                f"table.csv"
            ),
            mime="text/csv",
            key=(
                f"download_csv_"
                f"{page_number}"
            ),
        )

    except Exception as exc:

        st.error(
            f"Page {page_number} failed: "
            f"{type(exc).__name__}: {exc}"
        )

        with st.expander(
            "Full technical traceback"
        ):

            st.code(
                traceback.format_exc()
            )


# ============================================================
# FINAL OUTPUT
# ============================================================

st.divider()

st.header(
    "Extraction Results"
)


if not all_tables:

    st.warning(
        "No usable table data was extracted."
    )

else:

    total_rows = sum(
        len(dataframe)
        for _, dataframe
        in all_tables
    )

    result_col_1, result_col_2 = (
        st.columns(2)
    )

    result_col_1.metric(
        "Tables extracted",
        len(all_tables),
    )

    result_col_2.metric(
        "Extracted rows",
        total_rows,
    )

    excel_data = tables_to_excel(
        all_tables
    )

    st.download_button(
        label="Download All Tables as Excel",
        data=excel_data,
        file_name=(
            "document_extraction.xlsx"
        ),
        mime=(
            "application/"
            "vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )
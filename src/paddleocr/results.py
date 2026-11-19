"""Structured result types and serialization utilities for PaddleOCR services."""

from __future__ import annotations

import html as html_lib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image, ImageDraw

from .io import BBox, Polygon, box_intersection_over_area


def _resolve_output_file(
    save_path: str | Path,
    *,
    input_path: str | None,
    page_index: int | None,
    default_prefix: str,
    suffix: str,
) -> Path:
    target = Path(save_path)
    if target.suffix.lower() == suffix.lower() or (
        suffix == ".png" and target.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    target.mkdir(parents=True, exist_ok=True)
    if input_path and input_path.startswith(("http://", "https://")):
        stem = Path(urlparse(input_path).path).stem or default_prefix
    else:
        stem = Path(input_path).stem if input_path else default_prefix
    if page_index is not None:
        stem = f"{stem}_page{page_index}"
    return target / f"{stem}{suffix}"


@dataclass(slots=True)
class TextBox:
    """A single detected and recognized text line."""

    box: BBox
    polygon: Polygon
    text: str
    score: float
    det_score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": [round(v, 2) for v in self.box],
            "polygon": [[round(pt[0], 2), round(pt[1], 2)] for pt in self.polygon],
            "text": self.text,
            "score": round(self.score, 4),
            "det_score": round(self.det_score, 4),
        }


@dataclass(slots=True)
class TableResult:
    """Parsed table structure and content."""

    box: BBox
    table_type: str
    html: str
    markdown: str
    score: float = 1.0
    cell_boxes: list[BBox] = field(default_factory=list[BBox])

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": [round(v, 2) for v in self.box],
            "table_type": self.table_type,
            "html": self.html,
            "markdown": self.markdown,
            "score": round(self.score, 4),
            "cell_boxes": [[round(v, 2) for v in b] for b in self.cell_boxes],
        }


@dataclass(slots=True)
class LayoutBlock:
    """A document layout region with its recognized content."""

    label: str
    box: BBox
    score: float
    content: str = ""
    html: str | None = None
    polygon: Polygon | None = None
    order_index: int | None = None
    ocr_boxes: list[TextBox] = field(default_factory=list[TextBox])

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "label": self.label,
            "box": [round(v, 2) for v in self.box],
            "score": round(self.score, 4),
            "content": self.content,
        }
        if self.html is not None:
            data["html"] = self.html
        if self.polygon is not None:
            data["polygon"] = [[round(pt[0], 2), round(pt[1], 2)] for pt in self.polygon]
        if self.order_index is not None:
            data["order_index"] = self.order_index
        if self.ocr_boxes:
            data["ocr_boxes"] = [b.to_dict() for b in self.ocr_boxes]
        return data


def format_blocks_to_markdown(blocks: Sequence[LayoutBlock]) -> str:
    """Render ordered ``LayoutBlock`` items into a clean Markdown document."""
    sections: list[str] = []
    for block in blocks:
        label = block.label.lower()
        text = block.content.strip()
        if label in {"header", "footer", "page_number", "number"} and not text:
            continue

        if label in {"doc_title", "title"}:
            if text:
                sections.append(f"# {text}")
        elif label in {"paragraph_title", "sub_title", "section_header"}:
            if text:
                sections.append(f"## {text}")
        elif label in {"formula", "equation", "display_formula"}:
            if text:
                clean_tex = text.strip("$").strip()
                sections.append(f"$$\n{clean_tex}\n$$")
        elif label == "table":
            table_repr = text or (block.html or "")
            if table_repr:
                sections.append(table_repr)
        elif label == "chart":
            if text:
                sections.append(f"### Chart Data\n\n{text}")
        elif label == "seal":
            if text:
                sections.append(f"**[Seal: {text}]**")
        elif text:
            sections.append(text)

    return "\n\n".join(sections)


@dataclass(slots=True)
class OCRResult:
    """Result returned by ``PaddleOCR.predict()`` for a single page/image."""

    boxes: list[TextBox] = field(default_factory=list[TextBox])
    input_path: str | None = None
    page_index: int | None = None
    doc_orientation_angle: int | None = None
    image: Image.Image | None = field(default=None, repr=False)

    @property
    def dt_polys(self) -> list[Polygon]:
        return [b.polygon for b in self.boxes]

    @property
    def rec_texts(self) -> list[str]:
        return [b.text for b in self.boxes]

    @property
    def rec_scores(self) -> list[float]:
        return [b.score for b in self.boxes]

    @property
    def rec_boxes(self) -> list[BBox]:
        return [b.box for b in self.boxes]

    @property
    def markdown(self) -> str:
        return "\n".join(b.text for b in self.boxes if b.text.strip())

    @property
    def json(self) -> dict[str, Any]:
        return self.to_dict()

    def __getitem__(self, key: str) -> Any:
        mapping = self.to_dict()
        if key in mapping:
            return mapping[key]
        raise KeyError(key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "page_index": self.page_index,
            "doc_orientation_angle": self.doc_orientation_angle,
            "dt_polys": [
                [[round(pt[0], 2), round(pt[1], 2)] for pt in b.polygon] for b in self.boxes
            ],
            "rec_texts": self.rec_texts,
            "rec_scores": [round(s, 4) for s in self.rec_scores],
            "rec_boxes": [[round(v, 2) for v in b.box] for b in self.boxes],
            "boxes": [b.to_dict() for b in self.boxes],
        }

    def print(self) -> None:
        print(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    def save_to_json(self, save_path: str | Path) -> Path:
        out_file = _resolve_output_file(
            save_path,
            input_path=self.input_path,
            page_index=self.page_index,
            default_prefix="ocr_result",
            suffix=".json",
        )
        out_file.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return out_file

    def save_to_markdown(self, save_path: str | Path) -> Path:
        out_file = _resolve_output_file(
            save_path,
            input_path=self.input_path,
            page_index=self.page_index,
            default_prefix="ocr_result",
            suffix=".md",
        )
        out_file.write_text(self.markdown, encoding="utf-8")
        return out_file

    def save_to_img(self, save_path: str | Path) -> Path:
        out_file = _resolve_output_file(
            save_path,
            input_path=self.input_path,
            page_index=self.page_index,
            default_prefix="ocr_result",
            suffix=".png",
        )
        canvas = (
            self.image.copy() if self.image is not None else Image.new("RGB", (800, 1000), "white")
        )
        draw = ImageDraw.Draw(canvas)
        for item in self.boxes:
            if len(item.polygon) >= 2:
                pts = [*item.polygon, item.polygon[0]]
                draw.line(pts, fill=(255, 0, 0), width=2)
            else:
                draw.rectangle(item.box, outline=(255, 0, 0), width=2)
            if item.text:
                draw.text(
                    (item.box[0], max(0.0, item.box[1] - 14.0)), item.text, fill=(0, 100, 255)
                )
        canvas.save(out_file)
        return out_file

    def save_all(self, save_path: str | Path) -> dict[str, Path]:
        """Save JSON, Markdown, and annotated image outputs into ``save_path``."""
        return {
            "json": self.save_to_json(save_path),
            "markdown": self.save_to_markdown(save_path),
            "img": self.save_to_img(save_path),
        }


@dataclass(slots=True)
class StructureResult:
    """Result returned by ``PPStructureV3.predict()`` for a single page/image."""

    blocks: list[LayoutBlock] = field(default_factory=list[LayoutBlock])
    tables: list[TableResult] = field(default_factory=list[TableResult])
    formulas: list[str] = field(default_factory=list[str])
    seals: list[str] = field(default_factory=list[str])
    charts: list[str] = field(default_factory=list[str])
    ocr_boxes: list[TextBox] = field(default_factory=list[TextBox])
    input_path: str | None = None
    page_index: int | None = None
    doc_orientation_angle: int | None = None
    image: Image.Image | None = field(default=None, repr=False)

    @property
    def markdown(self) -> str:
        if self.blocks:
            return format_blocks_to_markdown(self.blocks)
        return "\n".join(b.text for b in self.ocr_boxes if b.text.strip())

    @property
    def json(self) -> dict[str, Any]:
        return self.to_dict()

    def __getitem__(self, key: str) -> Any:
        mapping = self.to_dict()
        if key in mapping:
            return mapping[key]
        raise KeyError(key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "page_index": self.page_index,
            "doc_orientation_angle": self.doc_orientation_angle,
            "blocks": [b.to_dict() for b in self.blocks],
            "tables": [t.to_dict() for t in self.tables],
            "formulas": list(self.formulas),
            "seals": list(self.seals),
            "charts": list(self.charts),
            "ocr_boxes": [b.to_dict() for b in self.ocr_boxes],
            "markdown": self.markdown,
        }

    def print(self) -> None:
        print(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    def save_to_json(self, save_path: str | Path) -> Path:
        out_file = _resolve_output_file(
            save_path,
            input_path=self.input_path,
            page_index=self.page_index,
            default_prefix="structure_result",
            suffix=".json",
        )
        out_file.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return out_file

    def save_to_markdown(self, save_path: str | Path) -> Path:
        out_file = _resolve_output_file(
            save_path,
            input_path=self.input_path,
            page_index=self.page_index,
            default_prefix="structure_result",
            suffix=".md",
        )
        out_file.write_text(self.markdown, encoding="utf-8")
        return out_file

    def save_to_img(self, save_path: str | Path) -> Path:
        out_file = _resolve_output_file(
            save_path,
            input_path=self.input_path,
            page_index=self.page_index,
            default_prefix="structure_result",
            suffix=".png",
        )
        canvas = (
            self.image.copy() if self.image is not None else Image.new("RGB", (800, 1000), "white")
        )
        draw = ImageDraw.Draw(canvas)
        for block in self.blocks:
            draw.rectangle(block.box, outline=(0, 180, 0), width=2)
            draw.text((block.box[0], max(0.0, block.box[1] - 14.0)), block.label, fill=(0, 120, 0))
        canvas.save(out_file)
        return out_file

    def save_all(self, save_path: str | Path) -> dict[str, Path]:
        """Save JSON, Markdown, and annotated image outputs into ``save_path``."""
        return {
            "json": self.save_to_json(save_path),
            "markdown": self.save_to_markdown(save_path),
            "img": self.save_to_img(save_path),
        }


@dataclass(slots=True)
class VLResult:
    """Result returned by ``PaddleOCRVL.predict()`` for a single page/image."""

    blocks: list[LayoutBlock] = field(default_factory=list[LayoutBlock])
    markdown_text: str = ""
    input_path: str | None = None
    page_index: int | None = None
    doc_orientation_angle: int | None = None
    image: Image.Image | None = field(default=None, repr=False)

    @property
    def markdown(self) -> str:
        if self.markdown_text.strip():
            return self.markdown_text
        return format_blocks_to_markdown(self.blocks)

    @property
    def json(self) -> dict[str, Any]:
        return self.to_dict()

    def __getitem__(self, key: str) -> Any:
        mapping = self.to_dict()
        if key in mapping:
            return mapping[key]
        raise KeyError(key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "page_index": self.page_index,
            "doc_orientation_angle": self.doc_orientation_angle,
            "blocks": [b.to_dict() for b in self.blocks],
            "markdown": self.markdown,
        }

    def print(self) -> None:
        print(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    def save_to_json(self, save_path: str | Path) -> Path:
        out_file = _resolve_output_file(
            save_path,
            input_path=self.input_path,
            page_index=self.page_index,
            default_prefix="vl_result",
            suffix=".json",
        )
        out_file.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return out_file

    def save_to_markdown(self, save_path: str | Path) -> Path:
        out_file = _resolve_output_file(
            save_path,
            input_path=self.input_path,
            page_index=self.page_index,
            default_prefix="vl_result",
            suffix=".md",
        )
        out_file.write_text(self.markdown, encoding="utf-8")
        return out_file

    def save_to_img(self, save_path: str | Path) -> Path:
        out_file = _resolve_output_file(
            save_path,
            input_path=self.input_path,
            page_index=self.page_index,
            default_prefix="vl_result",
            suffix=".png",
        )
        canvas = (
            self.image.copy() if self.image is not None else Image.new("RGB", (800, 1000), "white")
        )
        draw = ImageDraw.Draw(canvas)
        for block in self.blocks:
            draw.rectangle(block.box, outline=(0, 100, 255), width=2)
            draw.text((block.box[0], max(0.0, block.box[1] - 14.0)), block.label, fill=(0, 80, 200))
        canvas.save(out_file)
        return out_file

    def save_all(self, save_path: str | Path) -> dict[str, Path]:
        """Save JSON, Markdown, and annotated image outputs into ``save_path``."""
        return {
            "json": self.save_to_json(save_path),
            "markdown": self.save_to_markdown(save_path),
            "img": self.save_to_img(save_path),
        }


class _SimpleHTMLTableParser(HTMLParser):
    """Lightweight HTML table parser to convert ``<table>`` into Markdown tables."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        if lower == "tr":
            self._current_row = []
        elif lower in {"td", "th"} and self._current_row is not None:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if (
            lower in {"td", "th"}
            and self._current_row is not None
            and self._current_cell is not None
        ):
            cell_text = " ".join("".join(self._current_cell).split())
            self._current_row.append(cell_text)
            self._current_cell = None
        elif lower == "tr" and self._current_row is not None:
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = None


def html_table_to_markdown(html: str) -> str:
    """Convert an HTML ``<table>`` string into a GitHub-flavored Markdown table."""
    parser = _SimpleHTMLTableParser()
    parser.feed(html)
    if not parser.rows:
        return html

    num_cols = max(len(r) for r in parser.rows)
    if num_cols == 0:
        return html

    normalized_rows = [r + [""] * (num_cols - len(r)) for r in parser.rows]
    header = "| " + " | ".join(normalized_rows[0]) + " |"
    separator = "| " + " | ".join(["---"] * num_cols) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in normalized_rows[1:]]
    return "\n".join([header, separator, *body_lines])


_OTSL_CELL_PATTERN = re.compile(r"<(fcel|ecel|lcel|ucel|xcel)>")


def otsl_to_html(otsl_text: str) -> str:
    """Convert an OTSL table sequence (e.g. from ``PaddleOCR-VL-1.6``) into HTML ``<table>``."""
    raw_rows = [row for row in otsl_text.split("<nl>") if row.strip()]
    grid: list[list[tuple[str, str]]] = []
    for row_str in raw_rows:
        matches = list(_OTSL_CELL_PATTERN.finditer(row_str))
        if not matches:
            continue
        cells: list[tuple[str, str]] = []
        for i, match in enumerate(matches):
            token = f"<{match.group(1)}>"
            end = matches[i + 1].start() if i + 1 < len(matches) else len(row_str)
            raw_cell = row_str[match.end() : end]
            cleaned = html_lib.escape(" ".join(raw_cell.split()))
            cells.append((token, cleaned))
        grid.append(cells)

    if not grid:
        return otsl_text

    num_cols = max(len(row) for row in grid)
    if num_cols == 0:
        return otsl_text

    num_rows = len(grid)
    for row in grid:
        while len(row) < num_cols:
            row.append(("<ecel>", ""))

    html_rows: list[str] = []
    for r in range(num_rows):
        row_html = ["<tr>"]
        for c in range(num_cols):
            token, text = grid[r][c]
            if token not in {"<fcel>", "<ecel>"}:
                continue

            colspan = 1
            while c + colspan < num_cols and grid[r][c + colspan][0] == "<lcel>":
                colspan += 1

            rowspan = 1
            nr = r + 1
            while nr < num_rows and grid[nr][c][0] == "<ucel>":
                if colspan > 1 and any(
                    grid[nr][cc][0] != "<xcel>" for cc in range(c + 1, c + colspan)
                ):
                    break
                rowspan += 1
                nr += 1

            attrs = ""
            if colspan > 1:
                attrs += f' colspan="{colspan}"'
            if rowspan > 1:
                attrs += f' rowspan="{rowspan}"'
            row_html.append(f"<td{attrs}>{text}</td>")
        row_html.append("</tr>")
        html_rows.append("".join(row_html))

    return f"<table>{''.join(html_rows)}</table>"


def assemble_table_html(
    structure_tokens: Sequence[str],
    cell_boxes: Sequence[BBox],
    ocr_boxes: Sequence[TextBox],
) -> str:
    """Combine SLANet/SLANeXt HTML structure tokens, detected cell boxes, and OCR texts into HTML."""
    cell_texts: list[str] = []
    if cell_boxes:
        for cell_box in cell_boxes:
            matched: list[str] = []
            for tb in ocr_boxes:
                if box_intersection_over_area(tb.box, cell_box) >= 0.4:
                    matched.append(tb.text)
            cell_texts.append(" ".join(matched).strip())
    else:
        cell_texts = [tb.text for tb in ocr_boxes if tb.text.strip()]

    cell_idx = 0
    assembled: list[str] = []
    i = 0
    n = len(structure_tokens)
    while i < n:
        token = structure_tokens[i]
        if token in {"<html>", "</html>", "<body>", "</body>"}:
            i += 1
            continue

        if token == "<td></td>":
            text = cell_texts[cell_idx] if cell_idx < len(cell_texts) else ""
            cell_idx += 1
            assembled.append(f"<td>{text}</td>")
            i += 1
        elif token == "<td>":
            text = cell_texts[cell_idx] if cell_idx < len(cell_texts) else ""
            cell_idx += 1
            assembled.append(f"<td>{text}")
            i += 1
        elif token == "<td":
            # Collect attributes until ">"
            attrs: list[str] = [token]
            i += 1
            while i < n and structure_tokens[i] != ">":
                attrs.append(structure_tokens[i])
                i += 1
            if i < n and structure_tokens[i] == ">":
                attrs.append(">")
                i += 1
            text = cell_texts[cell_idx] if cell_idx < len(cell_texts) else ""
            cell_idx += 1
            assembled.append("".join(attrs) + text)
        else:
            assembled.append(token)
            i += 1

    html_str = "".join(assembled)
    if not html_str.startswith("<table"):
        html_str = f"<table>{html_str}</table>"
    return html_str

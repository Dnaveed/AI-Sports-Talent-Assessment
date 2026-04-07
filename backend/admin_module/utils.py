"""Admin utilities and report generation."""
from datetime import datetime, timedelta


def _csv_download(filename: str, headers: list[str], rows: list[list]) -> str:
    """Generate CSV content."""
    import io
    import csv
    from fastapi.responses import Response

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers)
    writer.writerows(rows)
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _pdf_escape(text: object) -> str:
    """Escape text for PDF."""
    safe = str(text).encode("latin-1", "replace").decode("latin-1")
    return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_pdf_line(text: str, limit: int = 92) -> list[str]:
    """Wrap text for PDF."""
    raw = str(text)
    if len(raw) <= limit:
        return [raw]
    chunks = []
    remaining = raw
    while len(remaining) > limit:
        split_at = remaining.rfind(" ", 0, limit)
        if split_at <= 18:
            split_at = limit
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _build_pdf(title: str, sections: list[tuple[str, list[str]]]) -> bytes:
    """Build PDF document."""
    page_width, page_height = 612, 792
    margin_left = 40
    margin_top = 52
    line_height = 12
    max_lines_per_page = 48

    raw_lines = []
    for heading, lines in sections:
        raw_lines.append(heading)
        raw_lines.extend(lines)
        raw_lines.append("")

    wrapped_lines = []
    for line in raw_lines:
        if line == "":
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(_wrap_pdf_line(line))

    pages = [wrapped_lines[i:i + max_lines_per_page] for i in range(0, len(wrapped_lines), max_lines_per_page)]
    if not pages:
        pages = [[title]]

    objects = []
    page_count = len(pages)
    page_ids = [4 + i * 2 for i in range(page_count)]
    content_ids = [5 + i * 2 for i in range(page_count)]

    objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{' '.join(f'{pid} 0 R' for pid in page_ids)}] /Count {page_count} >>")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for idx, page_lines in enumerate(pages):
        stream_lines = [
            "BT",
            "/F1 12 Tf",
            f"{margin_left} {page_height - margin_top} Td",
            f"({_pdf_escape(title)}) Tj",
            "/F1 9 Tf",
        ]
        for line in page_lines:
            if line == "":
                stream_lines.append(f"0 -{line_height * 2} Td")
            else:
                stream_lines.append(f"0 -{line_height} Td")
                stream_lines.append(f"({_pdf_escape(line)}) Tj")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines)
        content = f"<< /Length {len(stream.encode('utf-8'))} >>\nstream\n{stream}\nendstream"
        page_obj = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width} {page_height}] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_ids[idx]} 0 R >>"
        )
        objects.append(page_obj)
        objects.append(content)

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj_id, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{obj_id} 0 obj\n".encode("utf-8"))
        pdf.extend(body.encode("utf-8"))
        pdf.extend(b"\nendobj\n")

    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("utf-8"))
    pdf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("utf-8"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode("utf-8")
    )
    return bytes(pdf)

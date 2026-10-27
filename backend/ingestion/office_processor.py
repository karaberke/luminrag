"""Local-only Office/HTML/EPUB conversion through Microsoft's MarkItDown."""
from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath

from backend.ingestion.markdown_processor import chunk_markdown
from backend.utils.processes import MAX_DOCUMENT_TEXT_BYTES, run_parser

MAX_EXPANDED_BYTES = 100 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 10_000
MAX_XML_BYTES = 16 * 1024 * 1024
MAX_SPREADSHEET_SHEETS = 16
MAX_SPREADSHEET_ROWS = 10_000
MAX_SPREADSHEET_COLUMNS = 256
MAX_SPREADSHEET_CELLS = 100_000
SPREADSHEET_MAIN_TYPES = frozenset({
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.template.main+xml",
    "application/vnd.ms-excel.sheet.macroEnabled.main+xml",
    "application/vnd.ms-excel.template.macroEnabled.main+xml",
})
OFFICE_CONVERTERS = {".docx": "DocxConverter", ".pptx": "PptxConverter", ".xlsx": "XlsxConverter",
                     ".xls": "XlsConverter", ".html": "HtmlConverter", ".htm": "HtmlConverter", ".epub": "EpubConverter"}


def _spreadsheet_cells(rows: int, columns: int, total: int) -> int:
    """Account for the dense table pandas will create, including sparse gaps."""
    if rows > MAX_SPREADSHEET_ROWS or columns > MAX_SPREADSHEET_COLUMNS:
        raise ValueError("Spreadsheet row or column range exceeds the processing limit.")
    total += rows * columns
    if total > MAX_SPREADSHEET_CELLS:
        raise ValueError("Spreadsheet logical cell count exceeds the processing limit.")
    return total


def _validate_spreadsheet(path: Path) -> None:
    if path.suffix.lower() == ".xls":
        import xlrd
        with xlrd.open_workbook(path, on_demand=True) as workbook:
            if workbook.nsheets > MAX_SPREADSHEET_SHEETS:
                raise ValueError("Spreadsheet sheet count exceeds the processing limit.")
            total = 0
            for index in range(workbook.nsheets):
                sheet = workbook.sheet_by_index(index)
                total = _spreadsheet_cells(sheet.nrows, sheet.ncols, total)
                workbook.unload_sheet(index)
        return
    if path.suffix.lower() != ".xlsx":
        if not zipfile.is_zipfile(path):
            return
        from defusedxml.ElementTree import fromstring
        with zipfile.ZipFile(path) as archive:
            if "[Content_Types].xml" not in archive.namelist():
                return
            types = fromstring(archive.read("[Content_Types].xml"))
            if not any(item.get("ContentType", "") in SPREADSHEET_MAIN_TYPES for item in types):
                return
    # Use the standard safe XML iterator and openpyxl coordinate utilities.
    # Iterating worksheet rows through openpyxl/pandas would itself expand the
    # sparse gaps this check must reject before conversion.
    from defusedxml.ElementTree import iterparse
    from openpyxl import load_workbook
    from openpyxl.utils.cell import column_index_from_string, coordinate_from_string
    with path.open("rb") as stream:
        workbook = load_workbook(stream, read_only=True, data_only=True, keep_links=False)
        try:
            # Resolve the workbook's relationships with the same trusted reader
            # used by conversion; a worksheet need not use the conventional name.
            sheets = [sheet._worksheet_path for sheet in workbook.worksheets]
        finally:
            workbook.close()
    if len(sheets) > MAX_SPREADSHEET_SHEETS:
        raise ValueError("Spreadsheet sheet count exceeds the processing limit.")
    total = 0
    with zipfile.ZipFile(path) as archive:
        for name in sheets:
            rows = columns = current_row = current_column = stored_cells = 0
            with archive.open(name) as stream:
                for _, element in iterparse(stream, events=("end",), forbid_dtd=True,
                                            forbid_entities=True, forbid_external=True):
                    tag = element.tag.rsplit("}", 1)[-1]
                    if tag == "c":
                        reference = element.get("r")
                        if reference:
                            column, row = coordinate_from_string(reference)
                            current_column = column_index_from_string(column)
                            rows = max(rows, row)
                        else:
                            current_column += 1
                        columns = max(columns, current_column)
                        stored_cells += 1
                        if stored_cells > MAX_SPREADSHEET_CELLS:
                            raise ValueError("Spreadsheet cell count exceeds the processing limit.")
                    elif tag == "row":
                        row = int(element.get("r", current_row + 1))
                        if row <= current_row:
                            raise ValueError("Spreadsheet rows must have increasing indices.")
                        current_row = row
                        rows = max(rows, current_row)
                        current_column = 0
                    # Claimed dimensions are not trusted: actual row/cell
                    # coordinates are checked even if dimension says A1:A1.
                    if rows > MAX_SPREADSHEET_ROWS or columns > MAX_SPREADSHEET_COLUMNS:
                        raise ValueError("Spreadsheet row or column range exceeds the processing limit.")
                    if rows * columns + total > MAX_SPREADSHEET_CELLS:
                        raise ValueError("Spreadsheet logical cell count exceeds the processing limit.")
                    element.clear()
            total = _spreadsheet_cells(rows, columns, total)


def _validate_document_archive(path: Path) -> None:
    if path.suffix.lower() not in {".docx", ".pptx", ".xlsx", ".epub"} and not zipfile.is_zipfile(path):
        return
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > MAX_ARCHIVE_MEMBERS or sum(item.file_size for item in entries) > MAX_EXPANDED_BYTES:
            raise ValueError("Expanded document exceeds the processing limit.")
        if len({item.filename for item in entries}) != len(entries):
            raise ValueError("Document contains duplicate archive paths.")
        for item in entries:
            name = PurePosixPath(item.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in item.orig_filename:
                raise ValueError("Document contains an unsafe archive path.")
            # Use a parser that understands encoding declarations (including
            # UTF-16). Byte substring scans can miss entity declarations.
            if item.filename.lower().endswith((".xml", ".rels", ".opf", ".xhtml")):
                if item.file_size > MAX_XML_BYTES:
                    raise ValueError("Document XML member exceeds the processing limit.")
                from defusedxml.ElementTree import fromstring
                from defusedxml.common import DefusedXmlException
                from xml.etree.ElementTree import ParseError
                try:
                    # A standard EPUB HTML doctype is fine; entity declarations
                    # and external entity resolution are rejected by the parser.
                    fromstring(archive.read(item), forbid_dtd=False, forbid_entities=True, forbid_external=True)
                except (DefusedXmlException, ParseError) as exc:
                    raise ValueError("Unsafe or malformed document XML.") from exc


def _convert_office(path: Path) -> str:
    """Worker-only conversion; callers use the bounded process wrapper below."""
    from markitdown import StreamInfo, converters
    _validate_document_archive(path)
    _validate_spreadsheet(path)
    # Public, format-specific Microsoft converters retain their established
    # behavior without Magika's unnecessary ML sniffing/thread pool. There is
    # no URL dispatch, plugin discovery or provider configuration here.
    converter = getattr(converters, OFFICE_CONVERTERS[path.suffix.lower()])()
    with path.open("rb") as stream:
        converted = converter.convert(stream, stream_info=StreamInfo(extension=path.suffix.lower(), filename=path.name))
    text = converted.text_content
    if len(text.encode("utf-8")) > MAX_DOCUMENT_TEXT_BYTES:
        raise ValueError("Converted document text exceeds the processing limit.")
    return text


def process_office(path: str | Path, config: dict) -> list:
    path = Path(path)
    text = run_parser("office", path)
    chunks = chunk_markdown(text, path.stem, config, modality="office")
    for chunk in chunks:
        chunk.metadata["original_filename"] = path.name
    return chunks

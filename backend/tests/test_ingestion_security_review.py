"""Regressions for independently reviewed native-parser and legacy API paths."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock
import zipfile

import pymupdf
import pytest
from PIL import Image

from backend.graph.schema import ContentAttrs
from backend.ingestion import pdf_processor, slide_processor, office_processor
from backend.tests.test_ingest_safety import api_state, _staged_attempt
from backend.utils.processes import MIB, ProcessLimits, ProcessingLimitError


def pdf_file(tmp_path, pages=1, image=False):
    path = tmp_path / "source.pdf"
    with pymupdf.open() as document:
        for index in range(pages):
            page = document.new_page()
            page.insert_text((50, 50), f"Meaningful source sentence on page {index + 1}.")
            if image:
                from io import BytesIO
                data = BytesIO()
                Image.new("RGB", (100, 100), "green").save(data, format="PNG")
                page.insert_image(pymupdf.Rect(50, 100, 150, 200), stream=data.getvalue())
        document.save(path)
    return path


def test_pdf_text_images_and_page_provenance_survive_worker(tmp_path, monkeypatch):
    source = pdf_file(tmp_path, pages=2, image=True)
    native = MagicMock(side_effect=AssertionError("Native parsing in parent is forbidden"))
    monkeypatch.setattr(pdf_processor.fitz, "open", native)
    destination = tmp_path / "processed"
    chunks = pdf_processor.process_pdf(source, {"data": {"processed_dir": str(destination)}})
    assert chunks and any(chunk.metadata.get("page_spans") for chunk in chunks)
    assert chunks[-1].metadata["page_end"] == 2
    images = [Path(image) for chunk in chunks for image in chunk.metadata.get("image_paths", [])]
    assert images and all(image.is_file() and destination in image.parents for image in images)
    assert all(image.parent.name == "source" for image in images)
    assert not list(destination.glob("pdf-*"))
    native.assert_not_called()


def test_slide_native_work_is_credential_free_but_captioner_keeps_byok(tmp_path, monkeypatch):
    source = pdf_file(tmp_path, pages=2)
    captioner = MagicMock()
    captioner.caption.return_value = "A visual explanation."
    build = MagicMock(return_value=captioner)
    monkeypatch.setattr(slide_processor, "build_captioner", build)
    monkeypatch.setattr(slide_processor.fitz, "open", MagicMock(side_effect=AssertionError("No parent native parsing")))
    actual = slide_processor.run_parser
    options = []
    def capture(kind, path, **kwargs):
        options.append(kwargs["options"])
        return actual(kind, path, **kwargs)
    monkeypatch.setattr(slide_processor, "run_parser", capture)
    config = {"data": {"processed_dir": str(tmp_path / "processed")},
              "captioning": {"provider": "openrouter", "api_key": "test-only-request-key"}}
    chunks = slide_processor.process_slides(source, config)
    build.assert_called_once_with(config)
    assert captioner.caption.call_count == 2
    assert all(set(option) == {"image_dir", "image_dpi"} for option in options)
    assert all("[Visual: A visual explanation.]" in chunk.text for chunk in chunks)
    assert [chunk.metadata["page_number"] for chunk in chunks] == [1, 2]
    assert all(Path(chunk.metadata["slide_image_path"]).is_file() for chunk in chunks)


@pytest.mark.parametrize("processor", [pdf_processor, slide_processor])
def test_renamed_image_is_detected_by_mupdf_but_rejected_before_worker(tmp_path, monkeypatch, processor):
    path = tmp_path / "spoof.pdf"
    Image.new("RGB", (8, 8), "white").save(path, format="PNG")
    with pymupdf.open(path) as document:
        assert not document.is_pdf  # Confirms the reviewed content-detection behavior.
    run = MagicMock(side_effect=AssertionError("No parser should run"))
    monkeypatch.setattr(processor, "run_parser", run)
    function = processor.process_pdf if processor is pdf_processor else processor.process_slides
    with pytest.raises(ValueError, match="not a PDF"):
        function(path, {})
    run.assert_not_called()


def test_pdf_worker_deadline_cleans_private_assets(tmp_path, monkeypatch):
    from backend.utils import processes
    source = pdf_file(tmp_path)
    monkeypatch.setattr(processes, "PDF_LIMITS", ProcessLimits(.01, 128 * MIB))
    destination = tmp_path / "processed"
    with pytest.raises(ProcessingLimitError, match="time limit"):
        pdf_processor.process_pdf(source, {"data": {"processed_dir": str(destination)}})
    assert list(destination.iterdir()) == []


def test_worker_preserves_hosted_pdf_page_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMIN_HOSTED", "1")
    source = pdf_file(tmp_path, pages=501)
    with pytest.raises(ValueError, match="500 pages"):
        pdf_processor.process_pdf(source, {"data": {"processed_dir": str(tmp_path / "processed")}})


def test_inflated_xlsx_dimension_does_not_expand_pandas_table(tmp_path):
    from backend.tests.test_parser_security import workbook_path
    from markitdown.converters._xlsx_converter import _read_xlsx_sheets
    from io import BytesIO
    path = workbook_path(tmp_path, rows=3, columns=2)
    original = path.read_bytes()
    with zipfile.ZipFile(BytesIO(original)) as source, zipfile.ZipFile(path, "w") as destination:
        for item in source.infolist():
            data = source.read(item)
            if item.filename == "xl/worksheets/sheet1.xml":
                data = data.replace(b'ref="A1:B3"', b'ref="A1:DX1000"')
            destination.writestr(item, data)
    office_processor._validate_spreadsheet(path)
    with _read_xlsx_sheets(path.open("rb")) as (sheets, _):
        assert next(iter(sheets.values())).shape == (2, 2)


def test_duplicate_row_indices_cannot_amplify_dense_table(tmp_path):
    from backend.tests.test_parser_security import workbook_path
    from io import BytesIO
    path = workbook_path(tmp_path, rows=3, columns=2)
    original = path.read_bytes()
    with zipfile.ZipFile(BytesIO(original)) as source, zipfile.ZipFile(path, "w") as destination:
        for item in source.infolist():
            data = source.read(item)
            if item.filename == "xl/worksheets/sheet1.xml":
                data = data.replace(b'<row r="3">', b'<row r="1">')
            destination.writestr(item, data)
    with pytest.raises(ValueError, match="increasing indices"):
        office_processor._validate_spreadsheet(path)


def test_legacy_node_route_obeys_atomic_graph_limit(api_state, monkeypatch):
    from backend import workspace
    before = api_state.graph.graph.copy()
    stored = (api_state.processed / "graph.json").read_bytes()
    monkeypatch.setattr(workspace, "MAX_NODES", before.number_of_nodes())
    response = api_state.client.post("/api/graph/node", json={"name": "Over quota", "node_type": "topic"})
    assert response.status_code == 422
    assert api_state.main.state.graph_builder.graph.number_of_nodes() == before.number_of_nodes()
    assert (api_state.processed / "graph.json").read_bytes() == stored
    assert list(api_state.staging.iterdir()) == []


def test_legacy_edge_route_obeys_atomic_graph_limit(api_state, monkeypatch):
    from backend import workspace
    target = api_state.graph.add_node(ContentAttrs(name="Other concept"))
    api_state.graph.save()
    stored = (api_state.processed / "graph.json").read_bytes()
    source = next(node for node in api_state.graph.graph if node != target)
    monkeypatch.setattr(workspace, "MAX_EDGES", 0)
    response = api_state.client.post("/api/graph/edge", json={"source": source, "target": target, "relation": "RELATED_TO", "label": "explains"})
    assert response.status_code == 422
    assert api_state.main.state.graph_builder.graph.number_of_edges() == 0
    assert (api_state.processed / "graph.json").read_bytes() == stored


def test_legacy_edge_delete_rolls_back_when_publication_fails(api_state, monkeypatch):
    graph = api_state.graph
    target = graph.add_node(ContentAttrs(name="Other concept"))
    source = next(node for node in graph.graph if node != target)
    graph.add_related_to(source, target, "explains")
    graph.save()
    stored = (api_state.processed / "graph.json").read_bytes()
    monkeypatch.setattr(api_state.main, "_publish_storage", MagicMock(side_effect=ValueError("Simulated publication failure")))
    response = api_state.client.delete("/api/graph/edge", params={"source": source, "target": target, "relation": "RELATED_TO", "label": "explains"})
    assert response.status_code == 422
    assert api_state.main.state.graph_builder.graph.number_of_edges() == 1
    assert (api_state.processed / "graph.json").read_bytes() == stored


@pytest.mark.parametrize("payload", [{"name": "Topic", "summary": "x" * 100_001},
                                      {"name": "Topic", "key_terms": ["x" * 501]},
                                      {"name": "Topic", "parent_topic_keys": ["x"] * 201}])
def test_legacy_node_fields_are_bounded(api_state, payload):
    assert api_state.client.post("/api/graph/node", json=payload).status_code == 422


@pytest.mark.parametrize("hosted,token,optin,expected", [("1", None, None, False), ("0", "test-token", None, False),
                                                         ("0", None, None, True), ("1", "test-token", "1", True)])
def test_docs_disabled_for_hosted_and_token_protected_servers(monkeypatch, hosted, token, optin, expected):
    from backend import main
    monkeypatch.setenv("LUMIN_HOSTED", hosted)
    for name, value in (("LUMINRAG_API_TOKEN", token), ("LUMINRAG_ENABLE_DOCS", optin)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert main._api_docs_enabled() is expected


def test_ingest_unexpected_error_redacts_private_text_and_logs(api_state, monkeypatch, caplog):
    api = api_state
    job, staged, directory = _staged_attempt(api)
    private = "test-secret-at-private-filesystem-location"
    monkeypatch.setattr(api.main, "_ingest", MagicMock(side_effect=RuntimeError(private)))
    asyncio.run(api.main._run_ingest_job(job, staged, directory, 0))
    assert job.status == "failed"
    assert private not in job.error
    assert private not in caplog.text
    assert "Please check the file" in job.error

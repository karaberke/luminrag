"""Derived files must stay inside the selected instance's data directory."""

from io import BytesIO
from pathlib import Path

import fitz
from PIL import Image

from backend.ingestion.pdf_processor import process_pdf
from backend.ingestion.slide_processor import process_slides


def test_pdf_and_slide_assets_use_configured_data_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "lesson.pdf"
    picture = BytesIO()
    Image.new("RGB", (64, 64), "blue").save(picture, format="PNG")
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((50, 50), "Photosynthesis turns light into chemical energy.")
        page.insert_image(fitz.Rect(50, 100, 150, 200), stream=picture.getvalue())
        document.save(source)

    processed = tmp_path / "isolated" / "processed"
    config = {"data": {"processed_dir": str(processed)}}
    monkeypatch.setattr("backend.ingestion.slide_processor.build_captioner", lambda _config: None)
    monkeypatch.setattr("backend.ingestion.slide_processor.safe_caption", lambda *args: "A blue square.")
    chunks = process_pdf(source, config)
    slides = process_slides(source, config)
    images = [Path(path) for chunk in chunks for path in chunk.metadata.get("image_paths", [])]
    images.extend(Path(chunk.metadata["slide_image_path"]) for chunk in slides)
    assert len(images) >= 2
    assert all(path.is_file() and processed in path.parents for path in images)
    assert not (tmp_path / "backend" / "data").exists()

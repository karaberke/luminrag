"""Application schema boundaries and independently generated chunk defaults.

Successful constructor/value echoes are exercised by ingestion, retrieval and
generation tests rather than repeated for each Pydantic field here.
"""
import pytest
from pydantic import ValidationError

from backend.schemas import Chunk, GraphTriple, ReflectionVerdict, RetrievalResult


def test_chunk_ids_and_metadata_are_independent_and_explicit_ids_survive():
    first = Chunk(text="a", source_id="source", modality="pdf")
    second = Chunk(text="b", source_id="source", modality="pdf")
    explicit = Chunk(id="saved-id", text="c", source_id="source", modality="pdf")
    assert first.id and second.id and first.id != second.id
    assert explicit.id == "saved-id"
    first.metadata["page"] = 3
    assert second.metadata == explicit.metadata == {}


@pytest.mark.parametrize("model,values", [
    (Chunk, {"text": "x", "source_id": "source", "modality": "unknown"}),
    (Chunk, {"source_id": "source", "modality": "video"}),
    (GraphTriple, {"head": "A", "relation": "RELATED_TO", "source_chunk_ids": []}),
    (RetrievalResult, {"chunks": [], "subgraph": [], "routing_mode": "keyword"}),
    (ReflectionVerdict, {"needs_retrieval": True, "is_relevant": True,
                         "is_supported": True, "is_useful": True}),
])
def test_invalid_application_contract_is_rejected(model, values):
    with pytest.raises(ValidationError):
        model(**values)

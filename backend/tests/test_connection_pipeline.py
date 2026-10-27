"""Exercise scoped source context through ingestion without provider requests."""
from unittest.mock import Mock

import numpy as np

from backend.graph.graph_builder import GraphBuilder
from backend.graph.schema import ContentAttrs, ContentProposal, make_key
from backend.schemas import Chunk


def test_pipeline_verifies_new_concept_against_retained_document_passages(tmp_path, monkeypatch):
    # Importing the application must not read a developer's local credentials.
    monkeypatch.setenv("LUMINRAG_SKIP_DOTENV", "1")
    from backend import main

    retained = Chunk(id="retained-passage", source_id="existing-document", modality="pdf",
                     text="Light energy is absorbed by pigments during photosynthesis.")
    incoming = Chunk(id="incoming-passage", source_id="new-document", modality="pdf",
                     text="Chlorophyll is a pigment that absorbs light during photosynthesis.")
    builder = GraphBuilder(tmp_path / "graph.json")
    old_key = builder.add_node(ContentAttrs(name="Light energy", summary="Light used in photosynthesis.",
                                          source_ids=[retained.id]))
    new_title = "Chlorophyll pigment"
    new_key = make_key("content", new_title)
    topics = Mock(return_value=[])
    subtopics = Mock(return_value=[])
    contents = Mock(return_value=[ContentProposal(title=new_title, summary="A light-absorbing pigment.",
                                                  evidence_chunk_ids=[incoming.id])])
    monkeypatch.setattr("backend.graph.topic_extractor.extract_topics", topics)
    monkeypatch.setattr("backend.graph.subtopic_extractor.extract_subtopics", subtopics)
    monkeypatch.setattr("backend.graph.content_synthesizer.synthesize_contents", contents)

    def verify(prompt, _config, **_kwargs):
        # These assertions inspect the real bounded verifier prompt, rather
        # than replacing the integration helper with a permissive stub.
        assert retained.text in prompt and incoming.text in prompt
        assert retained.id in prompt and incoming.id in prompt
        assert "Light used in photosynthesis." in prompt
        assert "A light-absorbing pigment." in prompt
        return [{"index": 0, "related": True, "confidence": .85,
                 "source_chunk_ids": [retained.id, incoming.id]}]

    verifier = Mock(side_effect=verify)
    monkeypatch.setattr("backend.graph.semantic_linker.safe_call_json", verifier)
    provider = Mock(side_effect=AssertionError("This integration test must not call a provider."))
    monkeypatch.setattr("backend.llm.structured", provider)
    warmer = Mock(side_effect=AssertionError("Candidate ingestion must not warm the live graph."))
    monkeypatch.setattr(main, "warm_graph_retriever", warmer)
    embedder = Mock()
    embedder.embed.side_effect = lambda texts: np.array([
        [1., 0.] if text.startswith("Light energy") else [.7, .7] for text in texts
    ], dtype=np.float32)

    result = main._run_graph_pipeline(
        [incoming], ["new-document.pdf"], builder._path, {}, builder, embedder,
        warm_cache=False, context_chunks=[retained, incoming],
    )

    verifier.assert_called_once()
    contents.assert_called_once()
    assert contents.call_args.args[0] is incoming
    assert topics.call_args.args[1] == [incoming.id]
    assert subtopics.call_args.args[0] == [incoming]
    assert result.related_edges_added == 1
    assert set(builder.graph) == {old_key, new_key}
    attrs = next(iter(builder.graph.edges(data=True)))[2]
    assert attrs["relation"] == "RELATED_TO" and attrs["label"] is None
    assert attrs["supports"] == [{"kind": "automatic", "owner": None,
                                   "source_chunk_ids": sorted([retained.id, incoming.id]),
                                   "confidence": .85, "verified": True}]
    reloaded = GraphBuilder(builder._path)
    assert next(iter(reloaded.graph.edges(data=True)))[2]["supports"] == attrs["supports"]
    provider.assert_not_called()
    warmer.assert_not_called()

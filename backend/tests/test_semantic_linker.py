"""Associations require bounded source context, not invented relationship verbs."""
from __future__ import annotations

from unittest.mock import Mock, patch

import networkx as nx
import numpy as np
import pytest

from backend.graph.semantic_linker import (
    MAX_SOURCE_EXCERPT_CHARS, MAX_SUMMARY_CHARS,
    build_linkables_from_graph, link_nodes,
)
from backend.schemas import Chunk


@pytest.fixture
def config():
    return {"semantic_linker": {"sim_lo": .55, "sim_hi": .98, "max_candidate_pairs": 80,
                                "batch_size": 10, "min_confidence": .3, "max_tokens": 2048}}


def contextual_node(key, name, text, *, source_id=None):
    cid = source_id or "passage-" + key
    return {"key": key, "name": name, "text": name, "node_type": "content",
            "summary": "Summary for " + name, "source_ids": [cid],
            "source_excerpts": [{"id": cid, "text": text}]}


def nodes():
    return [contextual_node("a", "Photosynthesis", "Photosynthesis uses light to make sugars."),
            contextual_node("b", "Chlorophyll", "Chlorophyll absorbs light during photosynthesis.")]


def vectors():
    return np.array([[1., 0.], [.7, .7]], dtype=np.float32)


def decision(**changes):
    return {"index": 0, "related": True, "confidence": .8,
            "source_chunk_ids": ["passage-a", "passage-b"], **changes}


def link_response(response, config, concepts=None, embeddings=None):
    with patch("backend.graph.semantic_linker.safe_call_json", return_value=response) as llm:
        result = link_nodes(concepts or nodes(), Mock(), config, vectors=vectors() if embeddings is None else embeddings)
    return result, llm


def test_verified_association_has_grounding_but_no_named_predicate(config):
    proposals, llm = link_response([decision()], config)
    assert len(proposals) == 1
    assert {proposals[0].source_key, proposals[0].target_key} == {"a", "b"}
    assert proposals[0].label is None
    assert proposals[0].verified is True
    assert proposals[0].confidence == .8
    assert proposals[0].source_chunk_ids == ["passage-a", "passage-b"]
    prompt = llm.call_args.args[0]
    assert "Photosynthesis uses light to make sugars." in prompt
    assert "Chlorophyll absorbs light during photosynthesis." in prompt
    assert "Summary for Chlorophyll" in prompt
    assert "Do not return any relationship-name or label field" in prompt


@pytest.mark.parametrize("response", [
    [decision(related=False)], [decision(confidence=.1)],
    [decision(source_chunk_ids=[])], [decision(source_chunk_ids=["passage-a"])],
    [decision(source_chunk_ids=["passage-a", "foreign-workspace-passage"])],
    [decision(index=2)], [decision(index=-1)], [decision(index=True)], [decision(index="0")],
    [decision(related="true")], [decision(related=1)],
    [decision(confidence=float("nan"))], [decision(confidence=float("inf"))],
    [decision(confidence=2.)], [decision(confidence=-1.)], [decision(confidence=True)],
    [decision(label="explains")],
    [decision(), decision(related=False)], [decision(), decision()],
])
def test_invalid_or_unsupported_decisions_are_rejected(config, response):
    proposals, _ = link_response(response, config)
    assert proposals == []


def test_same_passage_can_ground_both_concepts(config):
    concepts = [contextual_node("a", "Photosynthesis", "Chlorophyll absorbs light for photosynthesis.", source_id="shared"),
                contextual_node("b", "Chlorophyll", "Chlorophyll absorbs light for photosynthesis.", source_id="shared")]
    proposals, _ = link_response([decision(source_chunk_ids=["shared"])], config, concepts)
    assert len(proposals) == 1 and proposals[0].source_chunk_ids == ["shared"]


def test_only_excerpts_actually_sent_can_be_cited(config):
    concepts = nodes()
    concepts[0]["source_ids"].append("not-in-prompt")
    proposals, _ = link_response([decision(source_chunk_ids=["not-in-prompt", "passage-b"])], config, concepts)
    assert proposals == []


def test_missing_or_unowned_context_skips_verification_calls(config):
    concepts = nodes()
    concepts[1]["source_excerpts"] = [{"id": "foreign-workspace", "text": "Real-looking but unowned."}]
    proposals, llm = link_response([decision()], config, concepts)
    assert proposals == []
    llm.assert_not_called()


def test_no_context_never_spends_an_ai_call(config):
    emb = Mock()
    concepts = [{"key": key, "name": key, "text": key, "node_type": "topic"} for key in ("a", "b")]
    with patch("backend.graph.semantic_linker.safe_call_json") as llm:
        assert link_nodes(concepts, emb, config) == []
    llm.assert_not_called()
    emb.embed.assert_not_called()


def test_unrelated_homonyms_remain_distinct_unconnected_nodes(config):
    concepts = [contextual_node("neuron", "Activation in neurons", "Neuronal activation follows a synaptic signal."),
                contextual_node("software", "Software activation", "A license key activates a software installation.")]
    proposals, llm = link_response([decision(related=False, source_chunk_ids=[])], config, concepts)
    assert proposals == [] and llm.call_count == 1
    assert [node["key"] for node in concepts] == ["neuron", "software"]


def test_out_of_range_or_single_node_never_calls_verifier(config):
    for concepts, embedded in [(nodes()[:1], vectors()[:1]), (nodes(), np.eye(2, dtype=np.float32))]:
        proposals, llm = link_response([decision()], config, concepts, embedded)
        assert proposals == []
        llm.assert_not_called()


def test_llm_failure_is_fail_closed(config):
    proposals, llm = link_response([], config)
    assert proposals == [] and llm.call_count == 1


def test_contexts_filter_vectors_without_mixing_node_identity(config):
    concepts = [nodes()[0], contextual_node("missing", "Missing source", ""), nodes()[1]]
    proposals, _ = link_response([decision()], config, concepts,
                                 np.array([[1., 0.], [0., 1.], [.7, .7]], dtype=np.float32))
    assert len(proposals) == 1
    assert {proposals[0].source_key, proposals[0].target_key} == {"a", "b"}


def test_batches_and_candidate_work_remain_bounded(config, monkeypatch):
    from backend.graph import semantic_linker as linker
    concepts = nodes()
    candidates = [(index, concepts[0], concepts[1], .7) for index in range(80)]
    candidate_lookup = Mock(return_value=candidates)
    monkeypatch.setattr(linker, "_candidate_pairs", candidate_lookup)
    with patch.object(linker, "safe_call_json", return_value=[]) as llm:
        assert link_nodes(concepts, Mock(), config, vectors=vectors()) == []
    assert llm.call_count == 8
    assert all(len(call.args[0].split("\n   B ")) == 11 for call in llm.call_args_list)
    assert candidate_lookup.call_args.args[4:6] == (80, 16)


def test_build_linkables_uses_actual_chunks_and_bounds_context():
    graph = nx.MultiDiGraph()
    graph.add_node("a", name="Concept", node_type="content", summary="s" * 1000,
                   source_ids=["one", "two", "three", "missing"])
    chunks = [Chunk(id=name, text=name + "x" * 2000, modality="pdf", source_id="local")
              for name in ("one", "two", "three")]
    linkable = build_linkables_from_graph(graph, chunks=chunks)[0]
    assert len(linkable["summary"]) == MAX_SUMMARY_CHARS
    assert [excerpt["id"] for excerpt in linkable["source_excerpts"]] == ["one", "two"]
    assert all(len(excerpt["text"]) == MAX_SOURCE_EXCERPT_CHARS for excerpt in linkable["source_excerpts"])
    assert linkable["source_excerpts"][0]["text"] == chunks[0].text[:MAX_SOURCE_EXCERPT_CHARS]
    assert "s" * 1000 in linkable["text"]


def test_build_linkables_cannot_substitute_a_different_chunk_under_an_owned_id():
    graph = nx.MultiDiGraph()
    graph.add_node("a", name="Concept", node_type="content", source_ids=["owned"])
    foreign = Chunk(id="foreign", text="Other material", modality="pdf", source_id="other")
    assert build_linkables_from_graph(graph, chunks={"owned": foreign})[0]["source_excerpts"] == []


def test_supported_long_source_identifiers_remain_eligible(config):
    cid = "source-" + "x" * 1017
    concepts = nodes()
    concepts[0]["source_ids"] = [cid]
    concepts[0]["source_excerpts"][0]["id"] = cid
    proposals, _ = link_response([decision(source_chunk_ids=[cid, "passage-b"])], config, concepts)
    assert len(proposals) == 1 and proposals[0].source_chunk_ids == [cid, "passage-b"]


def test_build_linkables_preserves_existing_text_and_node_type_filters():
    graph = nx.MultiDiGraph()
    graph.add_node("a", name="Topic", node_type="topic", summary="s")
    graph.add_node("b", name="Content", node_type="content", summary_intermediate="mid")
    graph.add_node("c", name="chunk", node_type="chunk_ref")
    assert [node["key"] for node in build_linkables_from_graph(graph)] == ["a", "b"]
    assert "mid" in build_linkables_from_graph(graph)[1]["text"]
    assert [node["key"] for node in build_linkables_from_graph(graph, node_types=["topic"])] == ["a"]

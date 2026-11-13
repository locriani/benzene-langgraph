"""Tests for Kekulé Benzene (C6H6) LangGraph topology, models, and resilience."""

from __future__ import annotations

from collections import Counter
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
import pytest
from langchain_google_genai import ChatGoogleGenerativeAI

from benzene_langgraph import models
from benzene_langgraph.chemistry import (
    CARBON_MODEL_ID,
    CARBON_NODES,
    HYDROGEN_MODEL_ID,
    HYDROGEN_NODES,
    KEKULE_ISOMERS,
    KekuleIsomer,
    verify_benzene_topology,
)
from benzene_langgraph.graph import build_benzene_graph
from benzene_langgraph.models import (
    ResilientAtomChatModel,
    build_raw_chat_model,
    create_carbon_model,
    create_hydrogen_model,
)


@pytest.mark.parametrize("isomer", ["A", "B"])
def test_kekule_stoichiometry_valency_and_double_bonds(isomer: KekuleIsomer) -> None:
    """Every Carbon has valency 4, every Hydrogen has valency 1, and C=C bonds have 2 edges."""
    cg = build_benzene_graph(isomer=isomer, cycles=1, mock=True)
    report = verify_benzene_topology(cg, isomer=isomer)

    assert report.total_covalent_edges == 15
    for h in HYDROGEN_NODES:
        assert report.valencies[h] == 1
    for c in CARBON_NODES:
        assert report.valencies[c] == 4

    if isomer == "A":
        assert report.ring_bond_orders == {
            "C1=C2": 2,
            "C2-C3": 1,
            "C3=C4": 2,
            "C4-C5": 1,
            "C5=C6": 2,
            "C6-C1": 1,
        }
    else:
        assert report.ring_bond_orders == {
            "C1-C2": 1,
            "C2=C3": 2,
            "C3-C4": 1,
            "C4=C5": 2,
            "C5-C6": 1,
            "C6=C1": 2,
        }


@pytest.mark.parametrize("isomer", ["A", "B"])
@pytest.mark.parametrize("cycles", [1, 2])
def test_runtime_double_bond_traversal_ratio(isomer: KekuleIsomer, cycles: int) -> None:
    """At runtime, C=C double bonds exchange 2x the passes per cycle of C-C single bonds."""
    cg = build_benzene_graph(isomer=isomer, cycles=cycles, mock=True)
    result = cg.invoke({"prompt": "Prove Kekulé double bonding under Pregel execution"})

    assert set(result["hydrogen_orbitals"].keys()) == set(HYDROGEN_NODES)
    assert set(result["carbon_orbitals"].keys()) == set(CARBON_NODES)
    assert all(r == 2 * cycles + 1 for r in result["carbon_rounds"].values())
    assert "Kekulé Benzene" in result["aromatic_synthesis"]

    topology = KEKULE_ISOMERS[isomer]
    pair_events: Counter[frozenset[str]] = Counter()
    for t in result["bond_traversals"]:
        pair_events[frozenset((t["source"], t["target"]))] += 1

    for h, c in topology.ch_bonds:
        assert pair_events[frozenset((h, c))] == 1

    for ca, cb in topology.single_bonds:
        assert pair_events[frozenset((ca, cb))] == cycles

    for ca, cb in topology.double_bonds:
        assert pair_events[frozenset((ca, cb))] == 4 * cycles
        assert f"{ca}=>{cb}" in result["pi_channels"]
        assert f"{cb}=>{ca}" in result["pi_channels"]

    for ca, cb in topology.single_bonds:
        assert f"{ca}=>{cb}" not in result["pi_channels"]
        assert f"{cb}=>{ca}" not in result["pi_channels"]


def test_benchmark_stoichiometry_and_huckel() -> None:
    """The ledger sees every graph call vs. 1 per monatomic model, and only benzene is aromatic."""
    from benzene_langgraph.bench import run

    rows = {r["contestant"].split()[0]: r for r in run(("Is benzene aromatic?",), mock=True)}

    assert set(rows) == {"C6H6", "C3H8", "H", "C", "Ar"}
    assert rows["C6H6"]["calls"] == 24  # 6 Hydrogens + 6 Carbons x 3 rounds
    assert rows["C3H8"]["calls"] == 11  # 8 Hydrogens + 3 Carbons, each Carbon exactly once
    assert all(f"**{c} (CH" in rows["C3H8"]["trials"][0]["answer"] for c in ("C1", "C2", "C3"))
    assert all(rows[k]["calls"] == 1 for k in ("H", "C", "Ar"))
    assert [k for k, r in rows.items() if r["huckel"]] == ["C6H6"]


def test_gemini_4_model_configuration() -> None:
    """Hydrogen & Carbon raw factories configure Gemini 4 IDs, thinking_config, and temperature=None."""
    h_raw = build_raw_chat_model(
        model_id=HYDROGEN_MODEL_ID,
        atom="H1",
        thinking_level="xhigh",
        api_key="dummy-key",
        vertexai=False,
    )
    c_raw = build_raw_chat_model(
        model_id=CARBON_MODEL_ID,
        atom="C1",
        thinking_level="high",
        api_key="dummy-key",
        vertexai=False,
    )

    assert isinstance(h_raw, ChatGoogleGenerativeAI)
    assert h_raw.model == HYDROGEN_MODEL_ID
    assert h_raw.temperature is None
    assert h_raw.thinking_config == {"thinking_level": "xhigh"}

    assert isinstance(c_raw, ChatGoogleGenerativeAI)
    assert c_raw.model == CARBON_MODEL_ID
    assert c_raw.temperature is None
    assert c_raw.thinking_config == {"thinking_level": "high"}

    assert isinstance(create_hydrogen_model(atom="H1", api_key="dummy-key"), ResilientAtomChatModel)
    assert isinstance(create_carbon_model(atom="C1", api_key="dummy-key"), ResilientAtomChatModel)


def test_carbon_without_quota_raises_instead_of_becoming_hydrogen(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Free-Tier 429 surfaces as an error: no reroute, no retry, no Hydrogen stand-in."""
    creates: list[Any] = []

    class FreeTierInteractions:
        def create(self, **kwargs: Any) -> Any:
            creates.append(kwargs)
            raise RuntimeError("429 RESOURCE_EXHAUSTED limit: 0 on FreeTier")

    class FreeTierClient:
        interactions = FreeTierInteractions()

    def no_hydrogen(**kw: Any) -> Any:
        raise AssertionError("Carbon must not fall back to a Hydrogen model")

    monkeypatch.setattr(models, "build_genai_client", lambda **kw: FreeTierClient())
    monkeypatch.setattr(models, "build_raw_chat_model", no_hydrogen)

    with pytest.raises(RuntimeError, match="429"):
        create_carbon_model(atom="C1", api_key="free-tier-key").invoke([HumanMessage(content="test")])
    assert len(creates) == 1


def test_free_tier_429_and_remote_disconnect_resilience(monkeypatch: pytest.MonkeyPatch) -> None:
    """Carbon survives RemoteProtocolError drops during background polling."""
    monkeypatch.setattr(models.time, "sleep", lambda _: None)

    created_backends: list[bool | None] = []
    poll_count = 0

    class FakeInteraction:
        def __init__(self, status: str, output_text: str = "") -> None:
            self.id = "inter-123"
            self.status = status
            self.output_text = output_text

    class FakeInteractionsAPI:
        def __init__(self, vertexai: bool | None) -> None:
            self.vertexai = vertexai

        def create(self, **kwargs: Any) -> FakeInteraction:
            created_backends.append(self.vertexai)
            return FakeInteraction("in_progress")

        def get(self, interaction_id: str) -> FakeInteraction:
            nonlocal poll_count
            poll_count += 1
            if poll_count == 1:
                raise RuntimeError("Server disconnected without sending a response")
            return FakeInteraction("completed", "stabilized-orbital")

    class FakeClient:
        def __init__(self, vertexai: bool | None) -> None:
            self.interactions = FakeInteractionsAPI(vertexai)

    monkeypatch.setattr(
        models,
        "build_genai_client",
        lambda **kw: FakeClient(kw.get("vertexai")),
    )

    c_model = create_carbon_model(atom="C1", api_key="free-tier-key")
    res = c_model.invoke([HumanMessage(content="test")])

    assert res.content == "stabilized-orbital"
    assert created_backends == [None]
    assert poll_count == 2


def test_only_hydrogens_get_url_context_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hydrogens may visit web pages; a Carbon on the Hydrogen code path stays bond-local."""
    seen_tools: list[Any] = []

    class FakeDelegate:
        def invoke(self, messages: Any, stop: Any = None, **kwargs: Any) -> AIMessage:
            seen_tools.append(kwargs.get("tools"))
            return AIMessage(content="orbital")

    monkeypatch.setattr(models, "build_raw_chat_model", lambda **kw: FakeDelegate())

    msgs = [HumanMessage(content="Summarize https://example.com")]
    create_hydrogen_model(atom="H1", api_key="dummy-key").invoke(msgs)
    create_carbon_model(atom="C1", api_key="dummy-key")._invoke_hydrogen(msgs, None)

    assert seen_tools == [[{"url_context": {}}], None]

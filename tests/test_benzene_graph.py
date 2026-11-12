"""Tests for Kekulé Benzene (C6H6) LangGraph topology, models, and double bonding."""

from __future__ import annotations

from collections import Counter

import pytest
from langchain_google_genai import ChatGoogleGenerativeAI

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
from benzene_langgraph.models import create_carbon_model, create_hydrogen_model


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

    # All 6 Hydrogens and all 6 Carbons ran
    assert set(result["hydrogen_orbitals"].keys()) == set(HYDROGEN_NODES)
    assert set(result["carbon_orbitals"].keys()) == set(CARBON_NODES)
    assert all(r == 2 * cycles + 1 for r in result["carbon_rounds"].values())
    assert "Kekulé Benzene" in result["aromatic_synthesis"]

    topology = KEKULE_ISOMERS[isomer]
    pair_events: Counter[frozenset[str]] = Counter()
    for t in result["bond_traversals"]:
        pair_events[frozenset((t["source"], t["target"]))] += 1

    # C-H single bonds fire exactly once at excitation
    for h, c in topology.ch_bonds:
        assert pair_events[frozenset((h, c))] == 1

    # C-C single bonds fire 1x per cycle (1 directional edge * 1 round/cycle)
    for ca, cb in topology.single_bonds:
        assert pair_events[frozenset((ca, cb))] == cycles

    # C=C double bonds fire 4x directed traversals per cycle (2 parallel edges * 2 rounds/cycle)
    # and carry dedicated pi_channels that single bonds NEVER populate!
    for ca, cb in topology.double_bonds:
        assert pair_events[frozenset((ca, cb))] == 4 * cycles
        assert f"{ca}=>{cb}" in result["pi_channels"]
        assert f"{cb}=>{ca}" in result["pi_channels"]

    for ca, cb in topology.single_bonds:
        assert f"{ca}=>{cb}" not in result["pi_channels"]
        assert f"{cb}=>{ca}" not in result["pi_channels"]


def test_gemini_4_model_configuration() -> None:
    """Hydrogen & Carbon factories configure Gemini 4 IDs, thinking_config, and temperature=None."""
    h_llm = create_hydrogen_model(atom="H1", thinking_level="xhigh", api_key="dummy-key")
    c_llm = create_carbon_model(atom="C1", thinking_level="high", api_key="dummy-key")

    assert isinstance(h_llm, ChatGoogleGenerativeAI)
    assert h_llm.model == HYDROGEN_MODEL_ID
    assert h_llm.temperature is None
    assert h_llm.thinking_config == {"thinking_level": "xhigh"}

    assert isinstance(c_llm, ChatGoogleGenerativeAI)
    assert c_llm.model == CARBON_MODEL_ID
    assert c_llm.temperature is None
    assert c_llm.thinking_config == {"thinking_level": "high"}

"""Kekulé Benzene (C6H6) molecular topology and structural invariant checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

KekuleIsomer = Literal["A", "B"]

HYDROGEN_MODEL_ID = "gemini-4-hydrogen"
CARBON_MODEL_ID = "gemini-4-carbon"

HYDROGEN_NODES: tuple[str, ...] = ("H1", "H2", "H3", "H4", "H5", "H6")
CARBON_NODES: tuple[str, ...] = ("C1", "C2", "C3", "C4", "C5", "C6")
RING_POSITIONS: dict[str, str] = {
    "C1": "ipso",
    "C2": "ortho",
    "C3": "meta",
    "C4": "para",
    "C5": "meta",
    "C6": "ortho",
}

HYDROGEN_FACETS: dict[str, tuple[str, str]] = {
    "H1": (
        "First-Principles Decomposition",
        "Extract core assumptions, constraints, and problem structure rapidly.",
    ),
    "H2": (
        "Implementation & Execution Sketch",
        "Draft concrete algorithms, data structures, or actionable mechanisms.",
    ),
    "H3": (
        "Adversarial & Edge-Case Probe",
        "Identify boundary conditions, failure modes, and subtle pitfalls.",
    ),
    "H4": (
        "Efficiency & Scaling Analysis",
        "Evaluate computational complexity, latency, and resource trade-offs.",
    ),
    "H5": (
        "Invariants & Verification",
        "Define testable invariants, correctness proofs, and validation checks.",
    ),
    "H6": (
        "Architecture & Interface Ergonomics",
        "Assess API boundaries, composability, and system-level integration.",
    ),
}


@dataclass(frozen=True)
class KekuleTopology:
    """Alternating C=C double bonds and C-C single bonds for a Kekulé isomer."""

    isomer: KekuleIsomer
    # Each (Ca, Cb) pair has bond order 2: Ca -> Cb (sigma_bond) AND Cb -> Ca (pi_bond)
    double_bonds: tuple[tuple[str, str], ...]
    # Each (Cb, Cc) pair has bond order 1: Cb -> Cc (sigma_bond only)
    single_bonds: tuple[tuple[str, str], ...]
    # Each (Hi, Ci) pair has bond order 1: Hi -> Ci (C-H sigma bond)
    ch_bonds: tuple[tuple[str, str], ...] = (
        ("H1", "C1"),
        ("H2", "C2"),
        ("H3", "C3"),
        ("H4", "C4"),
        ("H5", "C5"),
        ("H6", "C6"),
    )

    def double_partner_of(self, carbon: str) -> tuple[str, Literal["forward", "reverse"]]:
        """Return (partner_carbon, orientation) for this carbon's C=C double bond."""
        for ca, cb in self.double_bonds:
            if carbon == ca:
                return cb, "forward"
            if carbon == cb:
                return ca, "reverse"
        raise KeyError(f"Unknown carbon node: {carbon}")

    def outgoing_single_target(self, carbon: str) -> str | None:
        """Return target carbon if `carbon` initiates a unidirectional C-C single bond."""
        for src, dst in self.single_bonds:
            if src == carbon:
                return dst
        return None

    def incoming_single_source(self, carbon: str) -> str | None:
        """Return source carbon if `carbon` receives a unidirectional C-C single bond."""
        for src, dst in self.single_bonds:
            if dst == carbon:
                return src
        return None


KEKULE_ISOMERS: dict[KekuleIsomer, KekuleTopology] = {
    # 1,3,5-cyclohexatriene (C1=C2, C3=C4, C5=C6; single C2-C3, C4-C5, C6-C1)
    "A": KekuleTopology(
        isomer="A",
        double_bonds=(("C1", "C2"), ("C3", "C4"), ("C5", "C6")),
        single_bonds=(("C2", "C3"), ("C4", "C5"), ("C6", "C1")),
    ),
    # Resonance-shifted 2,4,6-cyclohexatriene (C2=C3, C4=C5, C6=C1; single C1-C2, C3-C4, C5-C6)
    "B": KekuleTopology(
        isomer="B",
        double_bonds=(("C2", "C3"), ("C4", "C5"), ("C6", "C1")),
        single_bonds=(("C1", "C2"), ("C3", "C4"), ("C5", "C6")),
    ),
}


@dataclass(frozen=True)
class TopologyReport:
    """Verification summary of the compiled Benzene LangGraph."""

    isomer: KekuleIsomer
    carbon_nodes: tuple[str, ...]
    hydrogen_nodes: tuple[str, ...]
    total_covalent_edges: int
    valencies: dict[str, int]
    ring_bond_orders: dict[str, int]
    ch_bond_orders: dict[str, int]


def verify_benzene_topology(compiled_graph: Any, isomer: KekuleIsomer = "A") -> TopologyReport:
    """Verify C6H6 stoichiometry, valencies (C=4, H=1), and preserved C=C double bonds."""
    topology = KEKULE_ISOMERS[isomer]
    drawable = compiled_graph.get_graph()

    atom_nodes = {
        name: node
        for name, node in drawable.nodes.items()
        if not name.startswith("__")
    }
    expected_atoms = set(CARBON_NODES) | set(HYDROGEN_NODES)
    if set(atom_nodes) != expected_atoms:
        raise AssertionError(
            f"Expected 12 C6H6 atoms {sorted(expected_atoms)}, got {sorted(atom_nodes)}"
        )

    for h in HYDROGEN_NODES:
        meta = atom_nodes[h].metadata or {}
        if meta.get("model") != HYDROGEN_MODEL_ID or meta.get("element") != "H":
            raise AssertionError(f"Hydrogen node {h} has invalid metadata: {meta}")

    for c in CARBON_NODES:
        meta = atom_nodes[c].metadata or {}
        if meta.get("model") != CARBON_MODEL_ID or meta.get("element") != "C":
            raise AssertionError(f"Carbon node {c} has invalid metadata: {meta}")

    atomic_edges = [
        e
        for e in drawable.edges
        if not e.source.startswith("__") and not e.target.startswith("__")
    ]

    valencies: dict[str, int] = {atom: 0 for atom in expected_atoms}
    pair_counts: dict[frozenset[str], int] = {}
    for edge in atomic_edges:
        valencies[edge.source] += 1
        valencies[edge.target] += 1
        key = frozenset((edge.source, edge.target))
        pair_counts[key] = pair_counts.get(key, 0) + 1

    # 1. Total covalent bonds in C6H6: 6 (C-H) + 3 (C-C) + 3*2 (C=C) = 15
    if len(atomic_edges) != 15:
        raise AssertionError(f"Expected 15 covalent bonds in C6H6, found {len(atomic_edges)}")

    # 2. Valency invariants: every Hydrogen = 1, every Carbon = 4
    for h in HYDROGEN_NODES:
        if valencies[h] != 1:
            raise AssertionError(f"Hydrogen {h} violated valency=1 (got {valencies[h]})")
    for c in CARBON_NODES:
        if valencies[c] != 4:
            raise AssertionError(f"Carbon {c} violated valency=4 (got {valencies[c]})")

    # 3. C-H single bond orders = 1
    ch_bond_orders: dict[str, int] = {}
    for h, c in topology.ch_bonds:
        order = pair_counts.get(frozenset((h, c)), 0)
        if order != 1:
            raise AssertionError(f"Expected single bond {h}-{c} (order 1), got {order}")
        ch_bond_orders[f"{h}-{c}"] = order

    # 4. Alternating Kekulé ring bond orders (C=C -> 2, C-C -> 1)
    ring_pairs = [
        ("C1", "C2"),
        ("C2", "C3"),
        ("C3", "C4"),
        ("C4", "C5"),
        ("C5", "C6"),
        ("C6", "C1"),
    ]
    double_set = {frozenset(pair) for pair in topology.double_bonds}
    ring_bond_orders: dict[str, int] = {}
    for ca, cb in ring_pairs:
        key = frozenset((ca, cb))
        order = pair_counts.get(key, 0)
        expected_order = 2 if key in double_set else 1
        sep = "=" if expected_order == 2 else "-"
        if order != expected_order:
            raise AssertionError(
                f"Ring bond {ca}{sep}{cb} expected bond order {expected_order}, got {order}"
            )
        ring_bond_orders[f"{ca}{sep}{cb}"] = order

    return TopologyReport(
        isomer=isomer,
        carbon_nodes=CARBON_NODES,
        hydrogen_nodes=HYDROGEN_NODES,
        total_covalent_edges=len(atomic_edges),
        valencies=valencies,
        ring_bond_orders=ring_bond_orders,
        ch_bond_orders=ch_bond_orders,
    )

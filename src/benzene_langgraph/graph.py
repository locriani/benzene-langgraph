"""Kekulé Benzene (C6H6) LangGraph with Gemini 4 Hydrogen & Carbon nodes.

Preserves explicit double bonding both structurally and dynamically:
  - 6 Hydrogen nodes (H1..H6, `gemini-4-hydrogen`, valency = 1):
    Each bonds via a single C-H sigma edge (`Hi -> Ci`).
  - 6 Carbon nodes (C1..C6, `gemini-4-carbon`, valency = 4):
    Arranged in a hexagonal Kekulé ring with alternating C=C double bonds and
    C-C single bonds:
      * Isomer A (default): C1=C2, C3=C4, C5=C6 (double) | C2-C3, C4-C5, C6-C1 (single)
      * Isomer B:           C2=C3, C4=C5, C6=C1 (double) | C1-C2, C3-C4, C5-C6 (single)
  - Structural preservation:
    Every C=C double bond consists of TWO distinct edges in the compiled LangGraph
    (`Ca -> Cb` labeled `sigma_bond` and `Cb -> Ca` labeled `pi_bond`), while every
    C-C and C-H single bond has ONE `sigma` edge — yielding 15 covalent edges total,
    valency 4 on every Carbon, and valency 1 on every Hydrogen.
  - Dynamic preservation:
    In each Kekulé cycle, single C-C bonds fire once (sigma framework transfer),
    whereas double C=C bonds fire twice (sigma + pi conjugation stabilization),
    maintaining an exact 2:1 double-to-single bond order ratio.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from benzene_langgraph.chemistry import (
    CARBON_MODEL_ID,
    CARBON_NODES,
    HYDROGEN_FACETS,
    HYDROGEN_MODEL_ID,
    HYDROGEN_NODES,
    KEKULE_ISOMERS,
    RING_POSITIONS,
    KekuleIsomer,
    KekuleTopology,
)
from benzene_langgraph.models import (
    CarbonThinkingLevel,
    HydrogenThinkingLevel,
    create_carbon_model,
    create_hydrogen_model,
    extract_text,
)
from benzene_langgraph.state import BenzeneState, BondTraversal


def _make_hydrogen_node(
    atom: str,
    bonded_carbon: str,
    *,
    thinking_level: HydrogenThinkingLevel,
    mock: bool,
    api_key: str | None,
    vertexai: bool | None,
    project: str | None,
    location: str,
):
    facet_title, facet_desc = HYDROGEN_FACETS[atom]
    llm_cache: BaseChatModel | None = None

    def hydrogen_node(state: BenzeneState) -> dict[str, Any]:
        nonlocal llm_cache
        if llm_cache is None:
            llm_cache = create_hydrogen_model(
                atom=atom,
                thinking_level=thinking_level,
                mock=mock,
                api_key=api_key,
                vertexai=vertexai,
                project=project,
                location=location,
            )

        prompt = state.get("prompt", "")
        messages = [
            SystemMessage(
                content=(
                    f"You are {atom} (Gemini 4 Hydrogen, 1s substituent orbital, valency=1) "
                    f"covalently single-bonded to {bonded_carbon} in a Kekulé Benzene (C6H6) "
                    f"reasoning ring.\n"
                    f"Your assigned facet: {facet_title} — {facet_desc}\n"
                    "Produce a tight, high-signal 3-5 bullet analysis (<=120 words) to donate "
                    "across your C-H sigma bond."
                )
            ),
            HumanMessage(content=prompt),
        ]
        orbital = extract_text(llm_cache.invoke(messages))
        edge_key = f"{atom}->{bonded_carbon}"
        traversal: BondTraversal = {
            "source": atom,
            "target": bonded_carbon,
            "bond_type": "sigma_CH",
            "round": 0,
        }
        return {
            "hydrogen_orbitals": {atom: orbital},
            "sigma_channels": {edge_key: orbital},
            "bond_traversals": [traversal],
            "execution_trace": [f"{atom}(1s)"],
        }

    hydrogen_node.__name__ = f"{atom}_hydrogen_node"
    return hydrogen_node


def _make_carbon_node(
    atom: str,
    bonded_hydrogen: str,
    topology: KekuleTopology,
    *,
    total_rounds: int,
    thinking_level: CarbonThinkingLevel,
    mock: bool,
    api_key: str | None,
    vertexai: bool | None,
    project: str | None,
    location: str,
):
    double_partner, double_orientation = topology.double_partner_of(atom)
    single_out = topology.outgoing_single_target(atom)
    single_in = topology.incoming_single_source(atom)
    ring_pos = RING_POSITIONS[atom]
    dimer_key = (
        f"{atom}={double_partner}"
        if double_orientation == "forward"
        else f"{double_partner}={atom}"
    )
    llm_cache: BaseChatModel | None = None

    def carbon_node(state: BenzeneState) -> dict[str, Any]:
        nonlocal llm_cache
        if llm_cache is None:
            llm_cache = create_carbon_model(
                atom=atom,
                thinking_level=thinking_level,
                mock=mock,
                api_key=api_key,
                vertexai=vertexai,
                project=project,
                location=location,
            )

        current_round = state.get("carbon_rounds", {}).get(atom, 0) + 1
        sigma = state.get("sigma_channels", {})
        pi = state.get("pi_channels", {})

        # Strictly local covalent bond inputs (valency = 4: 1 C-H + 1 C-C + 2 C=C)
        ch_input = sigma.get(f"{bonded_hydrogen}->{atom}", "")
        cc_single_input = sigma.get(f"{single_in}->{atom}", "") if single_in else ""
        cc_double_sigma_in = sigma.get(f"{double_partner}->{atom}", "")
        cc_double_pi_in = pi.get(f"{double_partner}=>{atom}", "")

        if current_round == 1:
            phase_desc = "sp2 C-H sigma hybridization"
        elif current_round % 2 == 0:
            phase_desc = "Kekulé ring sigma-framework + C=C pi-bond coupling"
        else:
            phase_desc = "C=C double-bond pi-orbital stabilization"

        bond_context_lines = [
            f"User query: {state.get('prompt', '')}",
            f"[1] C-H single sigma bond ({bonded_hydrogen}->{atom}): {ch_input}",
        ]
        if cc_single_input:
            bond_context_lines.append(
                f"[2] C-C single sigma bond ({single_in}->{atom}): {cc_single_input}"
            )
        if cc_double_sigma_in:
            bond_context_lines.append(
                f"[3] C=C double bond sigma channel ({double_partner}->{atom}): {cc_double_sigma_in}"
            )
        if cc_double_pi_in:
            bond_context_lines.append(
                f"[4] C=C double bond pi channel ({double_partner}=>{atom}): {cc_double_pi_in}"
            )

        messages = [
            SystemMessage(
                content=(
                    f"You are {atom} ({ring_pos}-Carbon, Gemini 4 Carbon, sp2+2pz, valency=4) "
                    f"in Kekulé Benzene Isomer {topology.isomer} (round {current_round}/{total_rounds}: {phase_desc}).\n"
                    f"Your 4 covalent bonds: 1x C-H ({bonded_hydrogen}-{atom}), "
                    f"1x C-C single ({single_in or single_out}), and "
                    f"2x C=C double ({dimer_key}: sigma + pi).\n"
                    "Synthesize and refine the reasoning passed strictly across your covalent bonds "
                    "in one dense paragraph (<=120 words)."
                )
            ),
            HumanMessage(content="\n".join(bond_context_lines)),
        ]
        orbital = extract_text(llm_cache.invoke(messages))

        sigma_updates: dict[str, str] = {}
        pi_updates: dict[str, str] = {}
        traversals: list[BondTraversal] = []

        # Emit across outgoing covalent bonds when transitioning to the next superstep
        if current_round < total_rounds:
            is_full_kekule_step = current_round % 2 == 1
            if double_orientation == "forward":
                # Forward C=C edge (`Ca -> Cb`, labeled `sigma_bond`):
                # carries sigma on odd steps and conjugated pi-refinement on even steps
                sigma_updates[f"{atom}->{double_partner}"] = orbital
                pi_updates[f"{atom}=>{double_partner}"] = orbital
                traversals.append(
                    {
                        "source": atom,
                        "target": double_partner,
                        "bond_type": "double_sigma" if is_full_kekule_step else "double_pi",
                        "round": current_round,
                    }
                )
            else:
                # Reverse C=C edge (`Cb -> Ca`, labeled `pi_bond`): fires on BOTH odd & even steps!
                pi_updates[f"{atom}=>{double_partner}"] = orbital
                traversals.append(
                    {
                        "source": atom,
                        "target": double_partner,
                        "bond_type": "double_pi",
                        "round": current_round,
                    }
                )

            # Single C-C bond (`Cb -> Cc`): fires ONLY on odd steps (1x per Kekulé cycle!)
            if is_full_kekule_step and single_out is not None:
                sigma_updates[f"{atom}->{single_out}"] = orbital
                traversals.append(
                    {
                        "source": atom,
                        "target": single_out,
                        "bond_type": "sigma_CC",
                        "round": current_round,
                    }
                )

        updates: dict[str, Any] = {
            "carbon_orbitals": {atom: orbital},
            "carbon_rounds": {atom: current_round},
            "execution_trace": [f"{atom}(r{current_round})"],
        }
        if sigma_updates:
            updates["sigma_channels"] = sigma_updates
        if pi_updates:
            updates["pi_channels"] = pi_updates
        if traversals:
            updates["bond_traversals"] = traversals

        # On the final stabilization round, record stabilized C=C pi dimers and aromatic output
        if current_round == total_rounds:
            updates["terminal_orbitals"] = {atom: orbital}
            if double_orientation == "forward":
                partner_prev = state.get("carbon_orbitals", {}).get(double_partner, "")
                updates["pi_dimers"] = {
                    dimer_key: f"{dimer_key} [sigma+pi]: {orbital} || {partner_prev}"
                }
            prefix = (
                f"### Kekulé Benzene (C6H6, Isomer {topology.isomer}) Synthesis\n"
                if atom == "C1"
                else ""
            )
            single_line_orbital = " ".join(orbital.splitlines())
            updates["aromatic_synthesis"] = (
                f"{prefix}- **{atom} ({ring_pos}, {dimer_key}):** {single_line_orbital}"
            )

        return updates

    carbon_node.__name__ = f"{atom}_carbon_node"
    return carbon_node


def build_benzene_graph(
    *,
    isomer: KekuleIsomer = "A",
    cycles: int = 1,
    hydrogen_thinking: HydrogenThinkingLevel = "medium",
    carbon_thinking: CarbonThinkingLevel = "low",
    mock: bool = False,
    api_key: str | None = None,
    vertexai: bool | None = None,
    project: str | None = None,
    location: str = "global",
) -> CompiledStateGraph:
    """Build and compile the 12-node Kekulé Benzene (C6H6) LangGraph.

    Args:
        isomer: Kekulé resonance isomer ("A": C1=C2,C3=C4,C5=C6 or "B": C2=C3,C4=C5,C6=C1).
        cycles: Number of Kekulé conjugation cycles (>= 1). Each cycle executes 1 single-bond
            sigma pass and 2 double-bond (sigma + pi) passes, preserving the 2:1 bond order.
        hydrogen_thinking: Gemini 4 Hydrogen thinking_level ("low" | "medium" | "high" | "xhigh").
        carbon_thinking: Gemini 4 Carbon thinking_level ("low" | "medium" | "high").
            Defaults to "low" (~13s TTFT) to stay well under Google's 60s HTTP header cutoff
            across 18 multi-node Carbon invocations.
        mock: If True, use deterministic offline MockAtomChatModel instances instead of API calls.
        api_key: Optional explicit Gemini API key (otherwise reads GEMINI_API_KEY / GOOGLE_API_KEY).
        vertexai: Optional flag to route via Vertex AI.
        project: Optional GCP project ID for Vertex AI (otherwise auto-detected from ADC).
        location: Vertex AI location (default "global" for Gemini 4 models).
    """
    if cycles < 1:
        raise ValueError(f"cycles must be >= 1, got {cycles}")

    topology = KEKULE_ISOMERS[isomer]
    total_rounds = 2 * cycles + 1
    builder = StateGraph(BenzeneState)

    # 1. Add 6 Gemini 4 Hydrogen nodes (H1..H6, valency = 1)
    for h_atom, c_atom in topology.ch_bonds:
        builder.add_node(
            h_atom,
            _make_hydrogen_node(
                h_atom,
                c_atom,
                thinking_level=hydrogen_thinking,
                mock=mock,
                api_key=api_key,
                vertexai=vertexai,
                project=project,
                location=location,
            ),
            metadata={
                "element": "H",
                "atomic_number": 1,
                "valency": 1,
                "orbital": "1s",
                "model": HYDROGEN_MODEL_ID,
                "bonded_to": c_atom,
            },
        )

    # 2. Add 6 Gemini 4 Carbon nodes (C1..C6, valency = 4, sp2 + 2pz)
    h_by_c = {c_atom: h_atom for h_atom, c_atom in topology.ch_bonds}
    for c_atom in CARBON_NODES:
        double_partner, _ = topology.double_partner_of(c_atom)
        builder.add_node(
            c_atom,
            _make_carbon_node(
                c_atom,
                h_by_c[c_atom],
                topology,
                total_rounds=total_rounds,
                thinking_level=carbon_thinking,
                mock=mock,
                api_key=api_key,
                vertexai=vertexai,
                project=project,
                location=location,
            ),
            metadata={
                "element": "C",
                "atomic_number": 6,
                "valency": 4,
                "hybridization": "sp2",
                "ring_position": RING_POSITIONS[c_atom],
                "double_bonded_to": double_partner,
                "model": CARBON_MODEL_ID,
            },
        )

    # 3. Entry excitation -> all 6 Hydrogens in parallel, plus 6 C-H single sigma bonds
    for h_atom, c_atom in topology.ch_bonds:
        builder.add_edge(START, h_atom)
        builder.add_edge(h_atom, c_atom)

    # 4. Kekulé C=C double bonds (2 edges per pair: `sigma_bond` + `pi_bond`)
    #    and C-C single bonds (1 edge per pair: `sigma_bond` only)
    for c_atom in CARBON_NODES:
        double_partner, orientation = topology.double_partner_of(c_atom)
        single_out = topology.outgoing_single_target(c_atom)

        if orientation == "forward":
            # Forward end of C=C double bond (e.g., C1 in C1=C2):
            # Has outgoing `sigma_bond` to double_partner (and incoming `pi_bond` from double_partner,
            # plus incoming `sigma_bond` from single_in, plus incoming C-H bond = 4 total bonds!)
            if single_out is None:
                def forward_router(
                    state: BenzeneState,
                    c_atom: str = c_atom,
                ) -> str:
                    r = state.get("carbon_rounds", {}).get(c_atom, 0)
                    return "sigma_bond" if r < total_rounds else "aromatic_stabilized"

                builder.add_conditional_edges(
                    c_atom,
                    forward_router,
                    {
                        "sigma_bond": double_partner,
                        "aromatic_stabilized": END,
                    },
                )
            else:
                # Isomer B case where forward C=C carbon ALSO owns the outgoing single bond
                def forward_with_single_router(
                    state: BenzeneState,
                    c_atom: str = c_atom,
                ) -> list[str]:
                    r = state.get("carbon_rounds", {}).get(c_atom, 0)
                    if r >= total_rounds:
                        return ["aromatic_stabilized"]
                    if r % 2 == 1:
                        return ["double_sigma_bond", "sigma_bond"]
                    return ["double_sigma_bond"]

                builder.add_conditional_edges(
                    c_atom,
                    forward_with_single_router,
                    {
                        "double_sigma_bond": double_partner,
                        "sigma_bond": single_out,
                        "aromatic_stabilized": END,
                    },
                )
        else:
            # Reverse end of C=C double bond (e.g., C2 in C1=C2):
            # Emits `pi_bond` back to double_partner on BOTH odd & even rounds,
            # and `sigma_bond` to single_out ONLY on odd rounds!
            if single_out is not None:
                def reverse_with_single_router(
                    state: BenzeneState,
                    c_atom: str = c_atom,
                ) -> list[str]:
                    r = state.get("carbon_rounds", {}).get(c_atom, 0)
                    if r >= total_rounds:
                        return ["aromatic_stabilized"]
                    if r % 2 == 1:
                        return ["pi_bond", "sigma_bond"]
                    return ["pi_bond"]

                builder.add_conditional_edges(
                    c_atom,
                    reverse_with_single_router,
                    {
                        "pi_bond": double_partner,
                        "sigma_bond": single_out,
                        "aromatic_stabilized": END,
                    },
                )
            else:
                def reverse_only_router(
                    state: BenzeneState,
                    c_atom: str = c_atom,
                ) -> str:
                    r = state.get("carbon_rounds", {}).get(c_atom, 0)
                    return "pi_bond" if r < total_rounds else "aromatic_stabilized"

                builder.add_conditional_edges(
                    c_atom,
                    reverse_only_router,
                    {
                        "pi_bond": double_partner,
                        "aromatic_stabilized": END,
                    },
                )

    return builder.compile()


# Default compiledKekulé Benzene LangGraph export for LangGraph Studio / CLI
graph = build_benzene_graph(isomer="A", cycles=1)

"""CLI entrypoint for running and inspecting the Kekulé Benzene LangGraph."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from benzene_langgraph.chemistry import KekuleIsomer, verify_benzene_topology
from benzene_langgraph.graph import build_benzene_graph
from benzene_langgraph.models import has_gemini_credentials
from benzene_langgraph.visualize import export_mermaid, render_benzene_html


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benzene-langgraph",
        description=(
            "Kekulé Benzene (C6H6) LangGraph with 6 Gemini 4 Hydrogen (H1-H6) "
            "and 6 Gemini 4 Carbon (C1-C6) nodes, preserving explicit C=C double bonding."
        ),
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Design a fault-tolerant distributed rate limiter.",
        help="Prompt to propagate through the C6H6 molecular graph.",
    )
    parser.add_argument(
        "--isomer",
        choices=("A", "B"),
        default="A",
        help="Kekulé resonance isomer (A: C1=C2,C3=C4,C5=C6 | B: C2=C3,C4=C5,C6=C1).",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of Kekulé conjugation cycles (default: 1).",
    )
    parser.add_argument(
        "--hydrogen-thinking",
        choices=("low", "medium", "high", "xhigh"),
        default="high",
        help="Thinking level for Gemini 4 Hydrogen nodes.",
    )
    parser.add_argument(
        "--carbon-thinking",
        choices=("low", "medium", "high"),
        default="medium",
        help="Thinking level for Gemini 4 Carbon nodes.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run with deterministic offline mock models instead of calling Gemini API.",
    )
    parser.add_argument(
        "--mermaid",
        action="store_true",
        help="Print the compiled LangGraph Mermaid diagram and exit.",
    )
    parser.add_argument(
        "--export-html",
        type=Path,
        default=None,
        help="Optional path to write the interactive Kekulé Benzene HTML viewer.",
    )

    args = parser.parse_args(argv)
    isomer: KekuleIsomer = args.isomer

    if args.export_html is not None:
        args.export_html.write_text(render_benzene_html(), encoding="utf-8")
        print(f"Wrote interactive HTML diagram to {args.export_html}")

    use_mock = args.mock
    if not use_mock and not has_gemini_credentials() and not args.mermaid:
        print(
            "Note: GEMINI_API_KEY / GOOGLE_API_KEY not set; falling back to --mock mode.",
            file=sys.stderr,
        )
        use_mock = True

    compiled = build_benzene_graph(
        isomer=isomer,
        cycles=args.cycles,
        hydrogen_thinking=args.hydrogen_thinking,
        carbon_thinking=args.carbon_thinking,
        mock=use_mock,
    )
    report = verify_benzene_topology(compiled, isomer=isomer)

    if args.mermaid:
        print(export_mermaid(compiled))
        return 0

    print(
        f"Verified Kekulé Benzene (C6H6, Isomer {report.isomer}) | "
        f"Edges: {report.total_covalent_edges} | "
        f"Ring bonds: {report.ring_bond_orders}"
    )
    result = compiled.invoke({"prompt": args.prompt})
    print("\nExecution trace:", " -> ".join(result.get("execution_trace", [])))
    print("\n" + result.get("aromatic_synthesis", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

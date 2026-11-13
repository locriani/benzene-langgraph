"""Aromatic Reasoning Benchmark: the C6H6 graph vs. a C3H8 propane graph and monatomic H, C and Ar.

Every number is measured, never estimated:
  reaction time   wall-clock seconds per answer
  stoichiometry   model calls per answer (distinct replies seen by a callback ledger)
  molar mass      total tokens per answer, as reported by the API
  yield           0-100 quality score from a blind judge (it never sees which contestant answered)
  Hückel          pass iff the contestant's pi-electron count is 4n+2
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from benzene_langgraph.chemistry import ARGON_MODEL_ID, CARBON_MODEL_ID, HYDROGEN_MODEL_ID, KEKULE_ISOMERS
from benzene_langgraph.graph import build_benzene_graph
from benzene_langgraph.models import create_argon_model, create_carbon_model, create_hydrogen_model, extract_text
from benzene_langgraph.state import merge_dicts

PROMPTS = (
    "Design a fault-tolerant distributed rate limiter.",
    "Explain Hückel's rule and whether cyclooctatetraene is aromatic.",
    "What are the trade-offs between optimistic and pessimistic locking?",
)

JUDGE_PROMPT = (
    "Score the answer below from 0 to 100 for correctness, depth and clarity in answering the question. "
    "Reply with only the integer.\n\nQUESTION:\n{prompt}\n\nANSWER:\n{answer}"
)


class Ledger(BaseCallbackHandler):
    """Counts distinct model replies and their tokens; a wrapper and its delegate share one reply id."""

    def __init__(self) -> None:
        self.tokens: dict[str, int] = {}

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        for generations in response.generations:
            for gen in generations:
                msg = getattr(gen, "message", None)
                if msg is not None and msg.id not in self.tokens:
                    self.tokens[msg.id] = (msg.usage_metadata or {}).get("total_tokens", 0)


def huckel(pi_electrons: int) -> bool:
    return pi_electrons > 0 and (pi_electrons - 2) % 4 == 0


# Propane CH3-CH2-CH3: the Hydrogens on each Carbon, in chain order.
PROPANE = {"C1": ("H1", "H2", "H3"), "C2": ("H4", "H5"), "C3": ("H6", "H7", "H8")}


class PropaneState(TypedDict, total=False):
    prompt: str
    orbitals: Annotated[dict[str, str], merge_dicts]


def build_propane_graph(*, mock: bool = False) -> CompiledStateGraph:
    """Propane (C3H8), the aliphatic control: benzene's atoms and prompts, sigma bonds only.

    Its 8 Hydrogens are equivalent, so they get no facets. Each Carbon waits for its Hydrogens
    and the Carbon before it, so the chain fires C1 -> C2 -> C3 once each: 11 calls, 0 pi electrons.
    """
    def atom_node(atom: str, model: Any, system: str, inputs: tuple[str, ...]) -> Callable[[PropaneState], dict]:
        def node(state: PropaneState) -> dict:
            bonds = "".join(f"\n[{src}->{atom}] {state['orbitals'][src]}" for src in inputs)
            human = f"User query: {state['prompt']}{bonds}" if inputs else state["prompt"]
            reply = model.invoke([SystemMessage(content=system), HumanMessage(content=human)])
            return {"orbitals": {atom: extract_text(reply)}}
        return node

    builder = StateGraph(PropaneState)
    prev: str | None = None
    for c, hs in PROPANE.items():
        for h in hs:
            builder.add_node(h, atom_node(h, create_hydrogen_model(atom=h, mock=mock), (
                f"You are {h} (Gemini 4 Hydrogen, 1s substituent orbital, valency=1) covalently "
                f"single-bonded to {c} in a propane (C3H8) reasoning chain.\n"
                "Produce a tight, high-signal 3-5 bullet analysis (<=120 words) to donate "
                "across your C-H sigma bond."), ()))
            builder.add_edge(START, h)
        inputs = (*hs, prev) if prev else hs
        builder.add_node(c, atom_node(c, create_carbon_model(atom=c, mock=mock), (
            f"You are {c} (Gemini 4 Carbon, sp3, valency=4) in propane (C3H8).\n"
            f"Your 4 covalent bonds: {len(hs)}x C-H and {4 - len(hs)}x C-C single.\n"
            "Synthesize and refine the reasoning passed strictly across your covalent bonds "
            "in one dense paragraph (<=120 words)."), inputs))
        builder.add_edge(list(inputs), c)  # join: wait for every bonded atom upstream
        prev = c
    builder.add_edge(prev, END)
    return builder.compile()


def propane_synthesis(state: PropaneState) -> str:
    items = (f"- **{c} (CH{len(hs)}):** {' '.join(state['orbitals'][c].splitlines())}" for c, hs in PROPANE.items())
    return "\n\n".join(["### Propane (C3H8) Synthesis", *items])


def contestants(mock: bool) -> list[tuple[str, int, Callable[[str, dict], str]]]:
    """(name, pi electrons, answer(prompt, config)) for each contestant."""
    benzene = build_benzene_graph(mock=mock)
    propane = build_propane_graph(mock=mock)
    h = create_hydrogen_model(atom="H", mock=mock)
    c = create_carbon_model(atom="C", mock=mock)
    ar = create_argon_model(atom="Ar", mock=mock)
    ask = lambda model: lambda p, cfg: extract_text(model.invoke([HumanMessage(content=p)], config=cfg))
    return [
        ("C6H6 benzene graph", 2 * len(KEKULE_ISOMERS["A"].double_bonds),
         lambda p, cfg: benzene.invoke({"prompt": p}, config=cfg)["aromatic_synthesis"]),
        ("C3H8 propane graph", 0, lambda p, cfg: propane_synthesis(propane.invoke({"prompt": p}, config=cfg))),
        (f"H ({HYDROGEN_MODEL_ID})", 0, ask(h)),
        (f"C ({CARBON_MODEL_ID})", 0, ask(c)),
        (f"Ar ({ARGON_MODEL_ID})", 0, ask(ar)),
    ]


def judge_score(judge: Any, prompt: str, answer: str) -> int | None:
    reply = extract_text(judge.invoke([HumanMessage(content=JUDGE_PROMPT.format(prompt=prompt, answer=answer))]))
    match = re.search(r"\b(100|\d{1,2})\b", reply)
    return int(match.group(1)) if match else None


def run(prompts: tuple[str, ...] = PROMPTS, *, mock: bool = False) -> list[dict[str, Any]]:
    judge = create_carbon_model(atom="judge", mock=mock)
    rows: list[dict[str, Any]] = []
    for name, pi, answer in contestants(mock):
        trials = []
        for p in prompts:
            ledger = Ledger()
            t0 = time.monotonic()
            text = answer(p, {"callbacks": [ledger]})
            seconds = time.monotonic() - t0
            trials.append({
                "prompt": p,
                "seconds": round(seconds, 2),
                "calls": len(ledger.tokens),
                "tokens": sum(ledger.tokens.values()),
                "yield": judge_score(judge, p, text),
                "answer": text,
            })
            print(f"  {name} | {p[:40]}... | {seconds:.1f}s", flush=True)
        yields = [t["yield"] for t in trials if t["yield"] is not None]
        rows.append({
            "contestant": name,
            "pi_electrons": pi,
            "huckel": huckel(pi),
            "seconds": round(statistics.mean(t["seconds"] for t in trials), 2),
            "calls": statistics.mean(t["calls"] for t in trials),
            "tokens": round(statistics.mean(t["tokens"] for t in trials)),
            "yield": round(statistics.mean(yields), 1) if yields else None,
            "trials": trials,
        })
    return rows


def table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Contestant | Reaction time (s) | Stoichiometry (calls) | Molar mass (tokens) | Yield (0-100) | Specific yield (per 1k tokens) | π e⁻ | Hückel |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in rows:
        specific = f"{1000 * r['yield'] / r['tokens']:.2f}" if r["yield"] is not None and r["tokens"] else "—"
        lines.append(
            f"| {r['contestant']} | {r['seconds']:.1f} | {r['calls']:g} | {r['tokens']:,} | "
            f"{r['yield'] if r['yield'] is not None else '—'} | {specific} | {r['pi_electrons']} | "
            f"{'✅' if r['huckel'] else '❌'} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benzene-bench", description=__doc__.splitlines()[0])
    parser.add_argument("--mock", action="store_true", help="Offline mock models (no API calls).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Results JSON (default benchmarks/results.json; --mock writes only if given).")
    args = parser.parse_args(argv)

    rows = run(mock=args.mock)
    out = args.out or (None if args.mock else Path("benchmarks/results.json"))  # mock never clobbers real data
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

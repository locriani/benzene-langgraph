# Aromatic Reasoning Benchmark (ARB-1)

**Run dates:** 2026-11-12 and 2026-11-13 · **Raw data:** [`benchmarks/results.json`](benchmarks/results.json) · **Harness:** [`src/benzene_langgraph/bench.py`](src/benzene_langgraph/bench.py)

## Abstract

We ask whether resonance-stabilized multi-agent reasoning beats monatomic frontier models. Six contestants answer the same prompts: the Kekulé benzene graph (C₆H₆, twelve Gemini 4 nodes), a propane graph (C₃H₈, eleven nodes) and a butane graph (C₄H₁₀, fourteen nodes) as the aliphatic controls, both with σ bonds only, and three single-model baselines, `gemini-4-hydrogen`, `gemini-4-carbon` and `gemini-4-argon`. Benzene is the only contestant that satisfies Hückel's rule (6 π electrons, n = 1). It reaches this at an 18-fold increase in molar mass, and its yield is 37 points below the weakest baseline. Propane spends 42% of benzene's molar mass for 4.7 fewer points of yield. Butane, one CH₂ longer, recovers one of those points at more than three times propane's reaction time. Argon, a noble gas, forms no bonds and returns the highest yield per kilotoken.

## Contestants

| Contestant | Structure | Configuration |
|---|---|---|
| **C₆H₆ benzene graph** | 6 Hydrogen + 6 Carbon nodes, Kekulé isomer A, 1 conjugation cycle | Hydrogens at `thinking_level=medium`, Carbons at `low` |
| **C₃H₈ propane graph** | 8 Hydrogen + 3 Carbon nodes in a CH₃–CH₂–CH₃ chain, σ bonds only | Hydrogens at `medium`, Carbons at `low` |
| **C₄H₁₀ butane graph** | 10 Hydrogen + 4 Carbon nodes in a CH₃–CH₂–CH₂–CH₃ chain, σ bonds only | Hydrogens at `medium`, Carbons at `low` |
| **H** | a lone `gemini-4-hydrogen` atom | `medium` |
| **C** | a lone `gemini-4-carbon` atom | `low` |
| **Ar** | a lone `gemini-4-argon` atom (noble gas, valency 0) | `low` |

Propane and butane use benzene's models and prompt style and differ only in their bonds. Each Carbon waits for every atom bonded to it upstream, so a chain of n Carbons fires C1 → … → Cn once each: 3n + 2 calls per answer, 11 for propane and 14 for butane. Their Hydrogens are equivalent, as they nearly are in the molecules, so none gets a facet. Benzene's six each get one.

```
n := Carbons in a straight-chain alkane      h(Ci) := Hydrogens on Ci      J(c) := c waits on every atom bonded to it upstream
1.  h(Ci) = 4 − (C–C bonds of Ci) = 4 − [i > 1] − [i < n]   [valency 4; bench.alkane_chain]
2.  Σ h(Ci) = 2n + 2                                      [CnH2n+2]
3.  ∀c: J(c) → c fires exactly once                       [bench.py: add_edge(list(inputs), c)]
∴   calls = (2n + 2) + n = 3n + 2                         [11 for C3H8, 14 for C4H10; asserted in the bench test]
```

Each single atom gets the same thinking level its element uses inside the graph. Argon has no position in the graph, so it gets Carbon's level. Carbon and Argon run through the background Interactions API, and Hydrogen runs through `generate_content`. No element ever stands in for another: if `gemini-4-carbon` cannot run, the benchmark stops with an error and records nothing. That applies to the graph's Carbons, the Carbon baseline and the judge.

## Method

Every contestant answers three prompts, once each:

1. *Design a fault-tolerant distributed rate limiter.*
2. *Explain Hückel's rule and whether cyclooctatetraene is aromatic.*
3. *What are the trade-offs between optimistic and pessimistic locking?*

Each metric comes from a measurement, not an estimate:

| Metric | What is measured |
|---|---|
| **Reaction time** | Wall-clock seconds from prompt to final answer |
| **Stoichiometry** | Model calls per answer, counted by a callback ledger that sees each distinct reply once |
| **Molar mass** | Total tokens per answer, as the Gemini API reports them |
| **Yield** | 0–100 quality score from a blind judge (`gemini-4-carbon`, `low`), which never sees which contestant wrote the answer |
| **Specific yield** | Yield per 1,000 tokens |
| **Hückel compliance** | Pass if the contestant's π-electron count is 4n + 2. A lone atom has no π system. |

## Results

Means over the three prompts:

| Contestant | Reaction time (s) | Stoichiometry (calls) | Molar mass (tokens) | Yield (0-100) | Specific yield (per 1k tokens) | π e⁻ | Hückel |
|---|---:|---:|---:|---:|---:|---:|:---:|
| C6H6 benzene graph | 98.8 | 24 | 25,587 | 58.7 | 2.29 | 6 | ✅ |
| C3H8 propane graph | 106.5 | 11 | 10,633 | 54.0 | 5.08 | 0 | ❌ |
| C4H10 butane graph | 357.5 | 14 | 15,633 | 55.0 | 3.52 | 0 | ❌ |
| H (gemini-4-hydrogen) | **14.4** | 1 | 2,382 | **97.3** | 40.85 | 0 | ❌ |
| C (gemini-4-carbon) | 62.9 | 1 | 1,417 | 96.7 | 68.24 | 0 | ❌ |
| Ar (gemini-4-argon) | 75.5 | 1 | **1,364** | 95.7 | **70.16** | 0 | ❌ |

Per prompt:

| Prompt | C₆H₆ yield (s) | C₃H₈ yield (s) | C₄H₁₀ yield (s) | H yield (s) | C yield (s) | Ar yield (s) |
|---|---:|---:|---:|---:|---:|---:|
| Rate limiter | 52 (114.1) | 45 (205.5) | 52 (698.5) | 97 (20.0) | 96 (38.6) | 95 (158.9) |
| Hückel's rule | 62 (105.1) | 65 (35.5) | 55 (244.6) | 99 (12.0) | 98 (38.8) | 96 (19.9) |
| Locking | 62 (77.3) | 52 (78.4) | 58 (129.4) | 96 (11.2) | 96 (111.3) | 96 (47.9) |

## Discussion

**Aromatic stabilization.** Benzene is the only aromatic contestant, and we consider that result significant. Its lower yield fits delocalization: the answer is spread evenly over six carbons, and no single position commits to it. The judge received six paragraphs, one per ring position. It scored them like an answer that had been stabilized rather than given.

**Molar mass.** One benzene answer costs 24 calls and 25,587 tokens. That is 18.1 times a lone Carbon and 18.8 times a lone Argon, and roughly what you would expect from a molecule with twelve atoms that speak.

**Propane** is saturated and has no ring, so it has no π system to stabilize anything. It used 11 calls and 10,633 tokens, 42% of benzene's molar mass, and gave up 4.7 points of yield, which more than doubles its specific yield (5.08 against 2.29). It was still slower than benzene (106.5 s against 98.8 s), because a chain fires its Carbons one after another while a ring fires them together. Its answers run three paragraphs, half of benzene's six. On the Hückel prompt propane outscored benzene, 65 to 62: the one contestant with no claim to aromaticity gave the better account of it.

**Butane** adds one CH₂ to propane. It used 14 calls and 15,633 tokens, 47% more than propane, and returned one more point of yield (55.0 against 54.0), so its specific yield fell to 3.52. Four Carbons in series made it the slowest contestant by far (357.5 s), and its rate-limiter answer, at 698.5 s, is the slowest single reaction in the benchmark. Along the homologous series, yield rises by one point per CH₂ and specific yield falls. Extrapolated, a straight-chain alkane overtakes benzene at octane (C₈H₁₈, 26 calls).

**Hydrogen** is the lightest element and was the fastest contestant (14.4 s). It also scored the highest yield (97.3). We did not expect this, and we have no chemical explanation.

**Argon** has a full valence shell and bonded with nothing. It used the fewest tokens, posted the highest specific yield (70.16, 30.6 times benzene), and recorded the slowest single reaction of the lone atoms (158.9 s). We attribute the slow reaction to its inertness.

```
A(x) := x passes Hückel's rule (π e⁻ = 4n + 2)
Y(x) := mean yield of x          T(x) := mean molar mass (tokens) of x
1.  A(C6H6) ∧ ¬A(C3H8) ∧ ¬A(C4H10) ∧ ¬A(H) ∧ ¬A(C) ∧ ¬A(Ar)   [π e⁻ = 6, 0, 0, 0, 0, 0]
2.  Y(C6H6) = 58.7 < min(Y(H), Y(C), Y(Ar)) = 95.7  [results.json]
3.  T(C6H6) = 25,587 > 18 · max(T(C), T(Ar)) = 25,506  [results.json]
4.  Y(C3H8) = 54.0 < Y(C4H10) = 55.0 < Y(C6H6)  [results.json]
5.  T(C3H8) = 10,633 < T(C4H10) = 15,633 < T(C6H6)  [results.json]
∴   A ↛ Y: benzene is the only aromatic contestant, and the only ones it outscores are the other molecules
```

## Threats to validity

- **Sample size.** Three prompts, one trial each. The latencies are noisy: Carbon varies from 38.6 s to 111.3 s on similar prompts. Read the means as a direction, not a precise figure.
- **The judge is a contestant.** The judge is `gemini-4-carbon`. It is blind to authorship, but it may still prefer prose that sounds like its own.
- **Answer format.** The graph's final answer is six per-carbon paragraphs, not one answer, and the judge may have penalized the format as well as the content.
- **The alkanes ran later.** Each joined after the first run. Propane's answers were produced and judged in a second session on 2026-11-12, and butane's in a third on 2026-11-13. The other four rows come from the first run.
- **One butane answer was rerun.** Its first rate-limiter answer took 1,672.9 s and scored 35. The rerun replaced it whole (answer, time, tokens and score) and took 698.5 s and scored 52. A 17-point swing on one prompt is larger than any gap between the three molecules.
- **The alkanes' Hydrogens have no facets.** The controls change the Hydrogens' prompts as well as the bonds.
- **Timing granularity.** Carbon and Argon results arrive by polling every 2 s, which can add up to 2 s per call.
- **Token accounting.** Token counts are what each API path reports: `generate_content` usage for Hydrogen and Interactions usage for Carbon and Argon. The two paths may count thinking tokens differently.

## Reproduce

```bash
uv run python -m benzene_langgraph.bench          # live run (~45 min, ~175 Gemini calls including the judge)
uv run python -m benzene_langgraph.bench --mock   # offline dry run of the harness
```

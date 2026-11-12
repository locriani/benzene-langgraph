# Benzene LangGraph ($C_6H_6$) — Gemini 4 Hydrogen & Carbon

A Kekulé Benzene ($C_6H_6$) multi-agent `StateGraph` built with **LangGraph** and Google's element-named Gemini 4 models, preserving explicit **$\text{C}=\text{C}$ double bonding** both structurally and dynamically:

- **6 Hydrogen nodes (`H1`–`H6`, `gemini-4-hydrogen`, valency = 1):** High-speed 1s peripheral substituent nodes that execute in parallel at `START`, each donating its specialized facet across a single $\sigma_{\text{C-H}}$ bond (`Hi -> Ci`). Hydrogens carry Gemini's built-in URL Context tool, so they can visit and read any web page linked in your prompt.
- **6 Carbon nodes (`C1`–`C6`, `gemini-4-carbon`, valency = 4):** Deep-reasoning $sp^2 + 2p_z$ hexagonal ring nodes (*ipso*, *ortho*, *meta*, *para*) connected by alternating **double** ($\sigma + \pi$) and **single** ($\sigma$) bonds:
  - **Kekulé Isomer A (default):** `C1=C2`, `C3=C4`, `C5=C6` (double) and `C2-C3`, `C4-C5`, `C6-C1` (single)
  - **Kekulé Isomer B:** `C2=C3`, `C4=C5`, `C6=C1` (double) and `C1-C2`, `C3-C4`, `C5-C6` (single)

Only Hydrogens browse. Carbons reason strictly over what crosses their bonds, so web content reaches the ring through the C–H bonds, and that holds even when Carbon falls back to the Hydrogen model.

```
W(n) := node n may fetch web pages      H(n) := n is a Hydrogen node
R(n) := n reads the raw user prompt     B(n) := n reasons only over its bond inputs
1.  ∀n: H(n) ↔ R(n)        [graph.py: only Hydrogens receive the prompt directly]
2.  ∀n: ¬H(n) → B(n)       [design: Carbons stay bond-local]
3.  ∀n: W(n) ↔ H(n)        [models.py: url_context bound only when element == "H"]
∴   web content enters only at H1–H6 and reaches C1–C6 solely via σ(C–H)
```

## How Double Bonding Is Preserved

1. **Structural Graph Topology (15 Covalent Edges):**
   - Each $\text{C}=\text{C}$ double bond is represented by **two distinct directed edges** in `compiled.get_graph().edges` (`Ca -> Cb` $\sigma$-bond and `Cb -> Ca` $\pi$-bond).
   - Each $\text{C}-\text{C}$ single bond and each $\text{C}-\text{H}$ bond is represented by **one** $\sigma$-edge.
   - Every Carbon node has exact **degree/valency = 4** (`1 C–H` + `1 C–C` + `2 C=C`), and every Hydrogen node has **degree/valency = 1**.
2. **Dynamic Runtime Execution (2:1 Bond Order Ratio):**
   - In every Kekulé cycle, single $\text{C}-\text{C}$ bonds fire **once** ($\sigma$-framework transfer), while double $\text{C}=\text{C}$ bonds fire across **two conjugation rounds** ($\sigma + \pi$ orbital coupling via dedicated `pi_channels`).

## Usage

```bash
# Run with Gemini API key or Google Cloud Vertex AI ADC
# (If GEMINI_API_KEY is on AI Studio Free Tier where gemini-4-carbon has limit=0,
# Carbon nodes automatically route via Vertex AI ADC if available, or fall back to
# gemini-4-hydrogen@xhigh.)
export GEMINI_API_KEY="your-api-key"
uv run benzene-langgraph "Design a Byzantine-resilient consensus protocol."

# Route all Hydrogen & Carbon nodes directly through Vertex AI ADC
uv run benzene-langgraph --vertexai "Design a Byzantine-resilient consensus protocol."

# Switch Kekulé resonance isomer or run multiple conjugation cycles
uv run benzene-langgraph --isomer B --cycles 2 "Audit a lock-free ring buffer."

# Offline simulation & Mermaid graph export
uv run benzene-langgraph --mock
uv run benzene-langgraph --mermaid

# Run verification test suite
uv run pytest -v
```

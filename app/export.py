"""
Report export (Section 4.9 / 10): produces a re-runnable Jupyter notebook
of the whole session -- every question asked, the code that was actually
executed for the winning attempt of each step, and the verified answer.
"""
from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from app.models import AgentState


def session_to_notebook(turns: list[AgentState]) -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell(
            "# Autonomous Data-Analysis Agent -- session export\n\n"
            "Re-runnable record of this session: each question, the code that produced its "
            "verified answer, and the answer itself."
        ),
        new_code_cell("import pandas as pd\nimport numpy as np\nimport duckdb\n"),
    ]

    for i, state in enumerate(turns, start=1):
        cells.append(new_markdown_cell(f"## Question {i}\n\n**{state.get('question', '')}**"))

        transcript = state.get("transcript", [])
        winning_by_step: dict[tuple[int, int], object] = {}
        for attempt in transcript:
            if attempt.error is None and not attempt.policy_denied:
                winning_by_step[(attempt.hypothesis_id, attempt.step_id)] = attempt

        for (hyp_id, step_id), attempt in sorted(winning_by_step.items()):
            cells.append(new_markdown_cell(f"*Hypothesis {hyp_id}, step {step_id}*"))
            cells.append(new_code_cell(attempt.code))

        answer = state.get("final_answer") or "(no verified answer)"
        scope_note = state.get("scope_note")
        ci = state.get("confidence_interval")
        md = f"**Answer:** {answer}\n\n**Lineage:** {state.get('lineage', '')}"
        if ci:
            md += f"\n\n**Confidence interval:** {ci.low:.3g} - {ci.high:.3g} ({ci.basis})"
        if scope_note:
            md += f"\n\n**Scope note:** {scope_note}"
        flags = state.get("consistency_flags", [])
        if flags:
            md += "\n\n**Consistency flags:**\n" + "\n".join(f"- {f.description}" for f in flags)
        cells.append(new_markdown_cell(md))

    nb = new_notebook(cells=cells)
    return nb


def save_notebook(turns: list[AgentState], out_path: str) -> str:
    nb = session_to_notebook(turns)
    with open(out_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return out_path

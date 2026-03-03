#!/usr/bin/env python3
"""
report.py - Generate a standalone HTML benchmark report from draftbench JSON results.

Produces the same style of report as the draftbench sweep but with a cleaner
layout: summary cards, four Chart.js charts (wall time, TTFT, TPS, acceptance
rate), and a full data table.

Usage:
    python report.py results.json
    python report.py results.json -o my_report.html
    python report.py results.json --open
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime


# ---------------------------------------------------------------------------
# Data loading and metric computation
# ---------------------------------------------------------------------------

def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_metrics(data: dict) -> dict:
    """
    Derive all display-ready metrics from a draftbench results dict.
    Returns a flat dict consumed by build_html().
    """
    results   = [r for r in data.get("results", []) if "error" not in r]
    settings  = data.get("settings", {})

    # Metadata
    timestamp = data.get("timestamp", "")
    try:
        date_str = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d")
    except Exception:
        date_str = timestamp[:10] if len(timestamp) >= 10 else "—"

    targets = list(dict.fromkeys(r["target"] for r in results))
    drafts  = list(dict.fromkeys(r["draft"]  for r in results if r["draft"] is not None))

    # Fast lookup
    rmap: dict[tuple, dict] = {}
    for r in results:
        rmap[(r["target"], r.get("draft"))] = r

    # Single-target focus; multi-target: first target only (noted in output)
    target = targets[0] if targets else "Unknown"
    baseline = rmap.get((target, None), {})

    base_wall = baseline.get("wall_time",    0)
    base_ttft = baseline.get("mean_ttft",    0)
    base_tps  = baseline.get("mean_tps",     0) or None  # treat 0 as absent

    # Per-draft metrics
    draft_rows = []
    for d in drafts:
        r = rmap.get((target, d))
        if not r:
            continue
        wall = r.get("wall_time",    0)
        ttft = r.get("mean_ttft",    0)
        tps  = r.get("mean_tps",     0) or None
        acc  = r.get("acceptance_rate")

        speedup    = (base_wall / wall)                      if (wall > 0 and base_wall > 0) else None
        ttft_delta = (ttft - base_ttft) / base_ttft * 100   if base_ttft > 0               else None

        draft_rows.append({
            "label":      d,
            "wall":       wall,
            "ttft":       ttft,
            "tps":        tps,
            "acc":        acc,
            "speedup":    speedup,
            "ttft_delta": ttft_delta,
        })

    # Rank drafts by TPS first (best throughput signal when available),
    # falling back to wall-time speedup for runs where TPS wasn't reported.
    def draft_score(m):
        return (m["tps"] or 0, m["speedup"] or 0)

    valid  = [m for m in draft_rows if m["speedup"] is not None]
    winner = max(valid, key=draft_score) if valid else None
    worst  = min(valid, key=draft_score) if valid else None

    return {
        # metadata
        "name":         data.get("name", "Benchmark"),
        "hardware":     data.get("hardware", "Unknown"),
        "backend":      data.get("backend", "unknown"),
        "model_family": data.get("model_family", ""),
        "date_str":     date_str,
        "target":       target,
        "multi_target": len(targets) > 1,
        "runs":         settings.get("runs",       "?"),
        "max_tokens":   settings.get("max_tokens", "?"),
        "temperature":  settings.get("temperature","?"),
        # baseline
        "base_wall": base_wall,
        "base_ttft": base_ttft,
        "base_tps":  base_tps,
        # draft rows
        "draft_rows": draft_rows,
        "winner":     winner,
        "worst":      worst,
    }


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

BASELINE_COLOR = "rgba(113,128,150,0.85)"
WINNER_COLOR   = "rgba(124,58,237,0.90)"
GOOD_COLOR     = "rgba(72,187,120,0.85)"
WARN_COLOR     = "rgba(237,137,54,0.85)"
BAD_COLOR      = "rgba(229,62,62,0.85)"


def bar_color(m: dict, winner: dict | None) -> str:
    if winner and m["label"] == winner["label"]:
        return WINNER_COLOR
    sp = m["speedup"]
    if sp is None or sp < 1.0:
        return BAD_COLOR
    if sp < 1.5:
        return WARN_COLOR
    return GOOD_COLOR


def to_border(rgba: str) -> str:
    """Drop the alpha to 1 for border colour."""
    return re.sub(r",[\d.]+\)$", ",1)", rgba)


def _fmt_tps(v) -> str:
    return "&mdash;" if v is None else f"{v:.1f}"


def _fmt_acc(v) -> str:
    return "N/A" if v is None else f"{v:.1%}"


def _fmt_ttft_delta(v) -> str:
    if v is None:
        return "&mdash;"
    return f"{'+' if v > 0 else ''}{v:.0f}%"


def _badge(m: dict, winner: dict | None) -> str:
    if winner and m["label"] == winner["label"]:
        return '<span class="badge badge-winner">&#127942; Winner</span>'
    sp = m["speedup"] or 0
    if sp >= 1.5:
        return '<span class="badge badge-good">Fast</span>'
    if sp >= 1.0:
        return '<span class="badge badge-warn">Moderate</span>'
    return '<span class="badge badge-bad">Slower</span>'


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def build_html(m: dict) -> str:
    dr       = m["draft_rows"]
    winner   = m["winner"]
    worst    = m["worst"]
    base_wall = m["base_wall"]
    base_ttft = m["base_ttft"]
    base_tps  = m["base_tps"]

    # ---- chart data --------------------------------------------------------
    chart_labels = ["Baseline"] + [r["label"] for r in dr]
    all_colors   = [BASELINE_COLOR] + [bar_color(r, winner) for r in dr]
    border_colors = [to_border(c) for c in all_colors]

    wall_data  = [base_wall]  + [r["wall"]    for r in dr]
    ttft_data  = [base_ttft]  + [r["ttft"]    for r in dr]
    tps_data   = [base_tps]   + [r["tps"]     for r in dr]   # None = no bar
    acc_data   = [None]       + [r["acc"]     for r in dr]   # None = no bar

    labels_json  = json.dumps(chart_labels)
    colors_json  = json.dumps(all_colors)
    borders_json = json.dumps(border_colors)
    wall_json    = json.dumps(wall_data)
    ttft_json    = json.dumps(ttft_data)
    tps_json     = json.dumps(tps_data)
    acc_json     = json.dumps(acc_data)

    # ---- winner banner -----------------------------------------------------
    if winner:
        parts = [f"{winner['speedup']:.2f}&times; faster wall time"]
        if winner["acc"] is not None:
            parts.append(f"{winner['acc']:.0%} acceptance rate")
        if winner["tps"] is not None:
            parts.append(f"{winner['tps']:.0f} tokens/sec")
        if winner["ttft_delta"] is not None:
            direction = "lower" if winner["ttft_delta"] < 0 else "higher"
            parts.append(f"{abs(winner['ttft_delta']):.0f}% {direction} TTFT vs baseline")
        winner_banner = f"""
  <div class="winner-banner">
    <div class="winner-icon">&#127942;</div>
    <div class="winner-text">
      <h3>Best draft model: {winner['label']}</h3>
      <p>{" &bull; ".join(parts)}</p>
    </div>
  </div>"""
    else:
        winner_banner = ""

    # ---- summary cards -----------------------------------------------------
    best_ttft_row = min(dr, key=lambda r: r["ttft"]) if dr else None

    def delta_chip(pct, good_is_negative=True):
        if pct is None:
            return ""
        is_good = (pct < 0) == good_is_negative
        cls = "delta-good" if is_good else "delta-bad"
        sign = "+" if pct > 0 else ""
        return f'<div class="card-delta {cls}">{sign}{pct:.0f}%</div>'

    cards = f"""
  <div class="cards">
    <div class="card" style="--accent:#718096">
      <div class="card-label">Baseline TTFT</div>
      <div class="card-value">{base_ttft:.1f}s</div>
      <div class="card-sub">No draft model</div>
    </div>"""

    if best_ttft_row:
        cards += f"""
    <div class="card" style="--accent:#7c3aed">
      <div class="card-label">Best TTFT</div>
      <div class="card-value">{best_ttft_row['ttft']:.1f}s</div>
      <div class="card-sub">{best_ttft_row['label']}</div>
      {delta_chip(best_ttft_row['ttft_delta'], good_is_negative=True)}
    </div>"""

    if winner:
        cards += f"""
    <div class="card" style="--accent:#7c3aed">
      <div class="card-label">Best Speedup</div>
      <div class="card-value">{winner['speedup']:.2f}&times;</div>
      <div class="card-sub">Wall time &bull; {winner['label']}</div>
      <div class="card-delta delta-good">vs baseline</div>
    </div>"""

    if worst and worst["label"] != (winner["label"] if winner else None):
        overhead = (1 - worst["speedup"]) * 100
        if overhead > 0:
            chip = f'<div class="card-delta delta-bad">+{overhead:.0f}% overhead</div>'
            sub  = "Slower than baseline"
        else:
            chip = f'<div class="card-delta delta-good">{worst["speedup"]:.2f}&times; speedup</div>'
            sub  = "Modest speedup"
        cards += f"""
    <div class="card" style="--accent:#e53e3e">
      <div class="card-label">Worst: {worst['label']}</div>
      <div class="card-value">{worst['speedup']:.2f}&times;</div>
      <div class="card-sub">{sub}</div>
      {chip}
    </div>"""

    cards += "\n  </div>"

    # ---- table rows --------------------------------------------------------
    table_rows = f"""
        <tr class="row-baseline">
          <td><strong>No draft</strong> (baseline)</td>
          <td class="num-right">{base_wall:.1f}</td>
          <td class="num-right">1.00&times;</td>
          <td class="num-right">{base_ttft:.2f}</td>
          <td class="num-right">&mdash;</td>
          <td class="num-right">{_fmt_tps(base_tps)}</td>
          <td class="num-right">&mdash;</td>
          <td><span class="badge badge-baseline">Baseline</span></td>
        </tr>"""

    for r in dr:
        is_winner = winner and r["label"] == winner["label"]
        row_cls   = ' class="row-winner"' if is_winner else ""
        sp_str    = f"{r['speedup']:.2f}&times;" if r["speedup"] is not None else "&mdash;"

        def w(s):
            return f"<strong>{s}</strong>" if is_winner else s

        table_rows += f"""
        <tr{row_cls}>
          <td>{w(r['label'])}</td>
          <td class="num-right">{w(f"{r['wall']:.1f}")}</td>
          <td class="num-right">{w(sp_str)}</td>
          <td class="num-right">{w(f"{r['ttft']:.2f}")}</td>
          <td class="num-right">{_fmt_ttft_delta(r['ttft_delta'])}</td>
          <td class="num-right">{w(_fmt_tps(r['tps']))}</td>
          <td class="num-right">{w(_fmt_acc(r['acc']))}</td>
          <td>{_badge(r, winner)}</td>
        </tr>"""

    # ---- multi-target notice -----------------------------------------------
    multi_notice = ""
    if m["multi_target"]:
        multi_notice = """
  <div class="notice">
    &#9432; This results file contains multiple target models.
    Only the first target is shown. Run separate sweeps per target for full comparison.
  </div>"""

    # ---- title pieces ------------------------------------------------------
    family  = m["model_family"]
    target  = m["target"]
    # Bold the family name in the h1 if present
    if family and family in target:
        h1_inner = target.replace(family, f"{family} <span>Speculative Decoding</span>", 1)
    else:
        h1_inner = f"{target} <span>Speculative Decoding</span>"

    # ---- assemble ----------------------------------------------------------
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{m['name']} — Speculative Decoding Report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      min-height: 100vh;
      padding: 2rem;
    }}

    .container {{ max-width: 1100px; margin: 0 auto; }}

    .header {{
      margin-bottom: 2.5rem;
      border-bottom: 1px solid #2d3748;
      padding-bottom: 1.5rem;
    }}
    .header h1 {{
      font-size: 1.75rem;
      font-weight: 700;
      color: #f7fafc;
      letter-spacing: -0.02em;
    }}
    .header h1 span {{ color: #7c3aed; }}
    .meta {{
      display: flex;
      gap: 1.5rem;
      margin-top: 0.75rem;
      flex-wrap: wrap;
    }}
    .meta-item {{
      font-size: 0.8rem;
      color: #718096;
      display: flex;
      align-items: center;
      gap: 0.35rem;
    }}
    .meta-item strong {{ color: #a0aec0; font-weight: 500; }}

    .notice {{
      background: #1a2744;
      border: 1px solid #2a4a8a;
      border-radius: 8px;
      padding: 0.75rem 1rem;
      margin-bottom: 1.5rem;
      font-size: 0.85rem;
      color: #90cdf4;
    }}

    .winner-banner {{
      background: linear-gradient(135deg, #1a1040 0%, #2d1b69 100%);
      border: 1px solid #553c9a;
      border-radius: 12px;
      padding: 1.25rem 1.5rem;
      margin-bottom: 2rem;
      display: flex;
      align-items: center;
      gap: 1rem;
    }}
    .winner-icon {{ font-size: 2rem; }}
    .winner-text h3 {{ font-size: 1rem; font-weight: 600; color: #d6bcfa; margin-bottom: 0.2rem; }}
    .winner-text p  {{ font-size: 0.85rem; color: #a78bfa; }}

    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin-bottom: 2.5rem;
    }}
    .card {{
      background: #1a202c;
      border: 1px solid #2d3748;
      border-radius: 10px;
      padding: 1.25rem;
      position: relative;
      overflow: hidden;
    }}
    .card::before {{
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: var(--accent, #7c3aed);
    }}
    .card-label  {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #718096; margin-bottom: 0.5rem; }}
    .card-value  {{ font-size: 1.6rem; font-weight: 700; color: #f7fafc; line-height: 1; }}
    .card-sub    {{ font-size: 0.78rem; color: #718096; margin-top: 0.35rem; }}
    .card-delta  {{ display: inline-block; margin-top: 0.5rem; font-size: 0.78rem; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 20px; }}
    .delta-good  {{ background: #1c4532; color: #68d391; }}
    .delta-bad   {{ background: #742a2a; color: #fc8181; }}

    .charts-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.25rem;
      margin-bottom: 2.5rem;
    }}
    @media (max-width: 700px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}

    .chart-card {{
      background: #1a202c;
      border: 1px solid #2d3748;
      border-radius: 10px;
      padding: 1.25rem;
    }}
    .chart-card h3 {{
      font-size: 0.85rem;
      font-weight: 600;
      color: #a0aec0;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 1rem;
    }}
    .chart-card canvas {{ max-height: 220px; }}

    .table-card {{
      background: #1a202c;
      border: 1px solid #2d3748;
      border-radius: 10px;
      padding: 1.25rem;
      overflow-x: auto;
    }}
    .table-card h3 {{
      font-size: 0.85rem;
      font-weight: 600;
      color: #a0aec0;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 1rem;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
    thead tr {{ border-bottom: 1px solid #2d3748; }}
    th {{
      text-align: left;
      padding: 0.6rem 0.75rem;
      font-size: 0.72rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #718096;
      white-space: nowrap;
    }}
    td {{ padding: 0.75rem 0.75rem; border-bottom: 1px solid #1e2533; color: #e2e8f0; white-space: nowrap; }}
    tbody tr:last-child td {{ border-bottom: none; }}
    tbody tr:hover {{ background: #232a3b; }}

    .row-winner    {{ background: #1a1040 !important; }}
    .row-winner td {{ color: #d6bcfa; }}
    .row-baseline td {{ color: #718096; }}

    .badge         {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }}
    .badge-baseline {{ background: #2d3748; color: #a0aec0; }}
    .badge-winner   {{ background: #553c9a; color: #d6bcfa; }}
    .badge-good     {{ background: #1c4532; color: #68d391; }}
    .badge-warn     {{ background: #744210; color: #f6ad55; }}
    .badge-bad      {{ background: #742a2a; color: #fc8181; }}

    .num-right {{ text-align: right; }}
  </style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>{h1_inner}</h1>
    <div class="meta">
      <div class="meta-item">&#128197; <strong>{m['date_str']}</strong></div>
      <div class="meta-item">&#9881;&nbsp; <strong>Backend:</strong> {m['backend']}</div>
      <div class="meta-item">&#127919; <strong>Target:</strong> {m['target']}</div>
      <div class="meta-item">&#128260; <strong>Runs:</strong> {m['runs']} &nbsp;|&nbsp;
        <strong>Max tokens:</strong> {m['max_tokens']} &nbsp;|&nbsp;
        <strong>Temp:</strong> {m['temperature']}</div>
    </div>
  </div>
{multi_notice}
{winner_banner}
{cards}

  <div class="charts-grid">
    <div class="chart-card">
      <h3>Wall Time (seconds) &mdash; lower is better</h3>
      <canvas id="wallTimeChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Mean Time to First Token (seconds) &mdash; lower is better</h3>
      <canvas id="ttftChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Throughput &mdash; Tokens / Second &mdash; higher is better</h3>
      <canvas id="tpsChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Draft Acceptance Rate &mdash; higher is better</h3>
      <canvas id="acceptChart"></canvas>
    </div>
  </div>

  <div class="table-card">
    <h3>Full Results</h3>
    <table>
      <thead>
        <tr>
          <th>Draft Model</th>
          <th class="num-right">Wall Time (s)</th>
          <th class="num-right">Speedup</th>
          <th class="num-right">Mean TTFT (s)</th>
          <th class="num-right">TTFT Delta</th>
          <th class="num-right">Mean TPS</th>
          <th class="num-right">Acceptance Rate</th>
          <th>Rating</th>
        </tr>
      </thead>
      <tbody>{table_rows}
      </tbody>
    </table>
  </div>

</div>

<script>
  const LABELS       = {labels_json};
  const BAR_COLORS   = {colors_json};
  const BORDER_COLORS = {borders_json};
  const GRID_COLOR   = "rgba(45,55,72,0.8)";
  const TICK_COLOR   = "#718096";
  const LABEL_COLOR  = "#a0aec0";

  function baseOpts(unitLabel) {{
    return {{
      responsive: true,
      maintainAspectRatio: true,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: ctx => ctx.parsed.y !== null
              ? " " + ctx.parsed.y.toFixed(2) + (unitLabel ? " " + unitLabel : "")
              : " N/A"
          }}
        }}
      }},
      scales: {{
        x: {{
          ticks: {{ color: TICK_COLOR, font: {{ size: 11 }} }},
          grid:  {{ color: GRID_COLOR }},
        }},
        y: {{
          ticks: {{ color: TICK_COLOR, font: {{ size: 11 }} }},
          grid:  {{ color: GRID_COLOR }},
          title: {{ display: !!unitLabel, text: unitLabel, color: LABEL_COLOR, font: {{ size: 11 }} }},
        }}
      }}
    }};
  }}

  function dataset(data, label) {{
    return {{
      label,
      data,
      backgroundColor:  BAR_COLORS,
      borderColor:      BORDER_COLORS,
      borderWidth: 1,
      borderRadius: 4,
    }};
  }}

  new Chart(document.getElementById("wallTimeChart"), {{
    type: "bar",
    data: {{ labels: LABELS, datasets: [dataset({wall_json}, "Wall Time")] }},
    options: baseOpts("s"),
  }});

  new Chart(document.getElementById("ttftChart"), {{
    type: "bar",
    data: {{ labels: LABELS, datasets: [dataset({ttft_json}, "TTFT")] }},
    options: baseOpts("s"),
  }});

  new Chart(document.getElementById("tpsChart"), {{
    type: "bar",
    data: {{ labels: LABELS, datasets: [dataset({tps_json}, "TPS")] }},
    options: {{
      ...baseOpts("tok/s"),
      plugins: {{
        ...baseOpts("tok/s").plugins,
        tooltip: {{
          callbacks: {{
            label: ctx => ctx.parsed.y !== null
              ? " " + ctx.parsed.y.toFixed(1) + " tok/s"
              : " N/A (not reported for this run)"
          }}
        }}
      }}
    }},
  }});

  new Chart(document.getElementById("acceptChart"), {{
    type: "bar",
    data: {{ labels: LABELS, datasets: [dataset({acc_json}, "Acceptance Rate")] }},
    options: {{
      ...baseOpts(),
      scales: {{
        ...baseOpts().scales,
        y: {{
          ...baseOpts().scales.y,
          min: 0, max: 1,
          ticks: {{
            color: TICK_COLOR,
            font: {{ size: 11 }},
            callback: v => (v * 100).toFixed(0) + "%",
          }},
        }}
      }},
      plugins: {{
        ...baseOpts().plugins,
        tooltip: {{
          callbacks: {{
            label: ctx => ctx.parsed.y !== null
              ? " " + (ctx.parsed.y * 100).toFixed(1) + "%"
              : " N/A"
          }}
        }}
      }}
    }},
  }});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a standalone HTML report from draftbench JSON results."
    )
    parser.add_argument("results", help="Path to draftbench results JSON file")
    parser.add_argument("-o", "--output", help="Output HTML path (default: results file with .html)")
    parser.add_argument("--open", action="store_true", help="Open the report in the default browser after generating")
    args = parser.parse_args()

    if not os.path.isfile(args.results):
        print(f"Error: file not found: {args.results}", file=sys.stderr)
        sys.exit(1)

    out_path = args.output or re.sub(r"\.json$", "", args.results) + "_report.html"

    data    = load(args.results)
    metrics = compute_metrics(data)
    html    = build_html(metrics)

    with open(out_path, "w") as f:
        f.write(html)

    print(f"Report saved to: {out_path}")

    if args.open:
        if sys.platform == "darwin":
            subprocess.run(["open", out_path])
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", out_path])
        else:
            subprocess.run(["start", out_path], shell=True)


if __name__ == "__main__":
    main()

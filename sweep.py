#!/usr/bin/env python3
"""
sweep.py - Automated speculative decoding benchmark sweep.

Runs all target+draft model combinations, collects results, and generates
interactive Plotly charts.

Usage:
    python sweep.py --config sweep_config.json
    python sweep.py --config sweep_config.json --results results.json --chart chart.html
    python sweep.py --config-dir configs/              # Run all configs in directory
    python sweep.py --results results.json --chart-only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone

from bench import run_bench, PROMPTS
from server import LlamaCppBackend, VLLMBackend


# ---------------------------------------------------------------------------
# Tee — mirror stdout to a log file
# ---------------------------------------------------------------------------

class _Tee:
    """Write to both the original stdout and a log file simultaneously."""

    def __init__(self, log_path: str):
        self._stdout = sys.stdout
        self._log = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = self

    def write(self, data):
        self._stdout.write(data)
        self._log.write(data)

    def flush(self):
        self._stdout.flush()
        self._log.flush()

    def close(self):
        sys.stdout = self._stdout
        self._log.close()


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def parse_acceptance_rate(log_path: str) -> float | None:
    """Extract the last draft acceptance rate from a llama.cpp server log."""
    if not log_path or not os.path.isfile(log_path):
        return None
    pattern = re.compile(r"draft acceptance rate = ([\d.]+)")
    last = None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                last = float(m.group(1))
    return last


def run_single(
    target_path: str,
    draft_path: str | None,
    label: str,
    settings: dict,
    backend_type: str = "llamacpp",
    draft_method: str = "draft_model",
) -> dict:
    """Start a server, benchmark it, stop it, and return result dict."""
    t_wall_start = time.monotonic()

    port = settings.get("port", 8080)
    log_file = os.path.join(tempfile.gettempdir(), f"draftbench_server_{port}.log")

    if backend_type == "vllm":
        backend = VLLMBackend(
            model=target_path,
            draft_model=draft_path,
            draft_method=draft_method,
            host="0.0.0.0",
            port=port,
            num_speculative_tokens=settings.get("num_speculative_tokens", 5),
            extra_args=settings.get("extra_args", []),
        )
        model_name = target_path
        log_file = None
    else:
        backend = LlamaCppBackend(
            model_path=target_path,
            draft_path=draft_path,
            host="127.0.0.1",
            port=port,
            gpu_layers=settings.get("gpu_layers", 99),
            ctx_size=settings.get("ctx_size", 4096),
            llama_bin=settings.get("llama_bin"),
            log_file=log_file,
        )
        model_name = os.path.basename(target_path)

    backend.start()
    print(f"  Waiting for server ...")
    if not backend.wait_ready(timeout=180):
        print(f"  ERROR: Server failed to start", file=sys.stderr)
        backend.stop()
        return {"error": "server_failed"}

    print(f"  Server ready. Benchmarking ...")
    summary = run_bench(
        label=label,
        base_url=backend.base_url,
        model=model_name,
        prompts=PROMPTS,
        runs=settings.get("runs", 1),
        max_tokens=settings.get("max_tokens", 512),
        temperature=settings.get("temperature", 0.0),
        api_key=None,
    )

    backend.stop()

    wall_time = round(time.monotonic() - t_wall_start, 2)

    # Parse acceptance rate from logs (llama.cpp only)
    acceptance = parse_acceptance_rate(log_file) if (draft_path and log_file) else None

    tps_stat = summary.stat("tps")
    ttft_stat = summary.stat("ttft")
    total_stat = summary.stat("total_time")

    result = {
        "mean_tps": round(tps_stat.get("mean", 0), 2),
        "median_tps": round(tps_stat.get("median", 0), 2),
        "mean_ttft": round(ttft_stat.get("mean", 0), 3),
        "mean_total_time": round(total_stat.get("mean", 0), 2),
        "wall_time": wall_time,
        "acceptance_rate": round(acceptance, 4) if acceptance else None,
    }

    return result


def _load_existing_results(results_path: str) -> list[dict]:
    """Load existing results from a previous run, if any."""
    if not os.path.isfile(results_path):
        return []
    try:
        with open(results_path) as f:
            data = json.load(f)
        results = data.get("results", [])
        # Only keep successful results (no errors)
        return [r for r in results if "error" not in r]
    except (json.JSONDecodeError, KeyError):
        return []


def run_sweep(config: dict, results_path: str) -> list[dict]:
    """Run the full sweep and save results incrementally."""
    targets = config["targets"]
    drafts = config.get("drafts", [])
    settings = config.get("settings", {})
    backend_type = config.get("backend", "llamacpp")

    total_runs = len(targets) * (1 + len(drafts))

    # Load existing results for resume support
    results = _load_existing_results(results_path)
    completed = {(r["target"], r.get("draft")) for r in results}
    skipped = len(completed)

    print(f"\n{'='*60}")
    print(f"  Sweep: {len(targets)} targets x {len(drafts)} drafts = {total_runs} runs")
    if skipped:
        print(f"  Resuming: {skipped} already completed, {total_runs - skipped} remaining")
    print(f"{'='*60}\n")

    run_idx = 0

    for target in targets:
        target_label = target["label"]
        target_path = target["path"]

        # --- baseline (no draft) ---
        run_idx += 1
        if (target_label, None) in completed:
            print(f"[{run_idx}/{total_runs}] {target_label} (baseline) -- already done, skipping")
        else:
            print(f"[{run_idx}/{total_runs}] {target_label} (baseline)")
            try:
                result = run_single(target_path, None, f"{target_label} baseline", settings, backend_type, "draft_model")
            except Exception as e:
                print(f"  SKIPPED: {e}")
                # Skip all drafts for this target since we have no baseline
                run_idx += len(drafts)
                print(f"  Skipping {len(drafts)} draft runs for {target_label}\n")
                continue
            entry = {"target": target_label, "draft": None, **result}
            results.append(entry)
            completed.add((target_label, None))
            _save_results(results, config, results_path)
            _print_summary(entry)
            print()
            time.sleep(3)

        # --- with each draft ---
        for draft in drafts:
            run_idx += 1
            draft_label = draft["label"]
            draft_path = draft["path"]
            combo_label = f"{target_label} + {draft_label}"

            if (target_label, draft_label) in completed:
                print(f"[{run_idx}/{total_runs}] {combo_label} -- already done, skipping")
                continue

            print(f"[{run_idx}/{total_runs}] {combo_label}")
            try:
                result = run_single(target_path, draft_path, combo_label, settings, backend_type, draft.get("method", "draft_model"))
            except Exception as e:
                print(f"  SKIPPED: {e}\n")
                continue
            entry = {"target": target_label, "draft": draft_label, **result}
            results.append(entry)
            completed.add((target_label, draft_label))
            _save_results(results, config, results_path)
            _print_summary(entry)
            print()

            time.sleep(3)

    return results


def _print_summary(entry: dict):
    """Print a one-line summary of a run."""
    if "error" in entry:
        print(f"  ERROR: {entry['error']}")
        return
    parts = [f"{entry['mean_tps']} tok/s"]
    if entry.get("acceptance_rate"):
        parts.append(f"acceptance: {entry['acceptance_rate']:.0%}")
    print(f"  Result: {', '.join(parts)}")


def _save_results(results: list[dict], config: dict, path: str):
    """Incrementally save results to JSON."""
    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": config.get("name", "unnamed"),
        "hardware": config.get("hardware", "unknown"),
        "backend": config.get("backend", "unknown"),
        "model_family": config.get("model_family", "unknown"),
        "settings": config.get("settings", {}),
        "results": results,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def generate_chart(results: list[dict], output_path: str, metadata: dict = None):
    """Generate a standalone HTML file with Plotly charts."""
    metadata = metadata or {}
    hardware = metadata.get("hardware", "Unknown Hardware")
    backend = metadata.get("backend", "unknown")
    model_family = metadata.get("model_family", "")
    chart_subtitle = f"{model_family} on {hardware} ({backend})".strip()

    # Collect unique targets and drafts, preserving the order they appear in results
    # (which matches config order).
    targets_seen = list(dict.fromkeys(r["target"] for r in results))
    drafts_seen = list(dict.fromkeys(r["draft"] for r in results if r["draft"] is not None))

    # Fast lookup: (target, draft_or_None) -> result dict
    result_map: dict[tuple, dict] = {}
    for r in results:
        if "error" not in r:
            result_map[(r["target"], r.get("draft"))] = r

    # Color palette — cycles if there are more drafts than colors
    PALETTE = [
        "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3",
        "#FF6692", "#B6E880", "#FF97FF", "#FECB52", "#636EFA",
    ]
    BASELINE_COLOR = "#636EFA"
    draft_colors = {d: PALETTE[i % len(PALETTE)] for i, d in enumerate(drafts_seen)}

    def wall_speedup(target: str, draft: str) -> float | None:
        """Wall-time speedup % of draft vs baseline. None if data is missing."""
        base = result_map.get((target, None), {}).get("wall_time", 0)
        draft_r = result_map.get((target, draft), {}).get("wall_time", 0)
        if base > 0 and draft_r > 0:
            return round((base - draft_r) / base * 100, 1)
        return None

    # --- Chart 1: Mean total time per request (lower is better) ---
    time_traces = [{
        "x": targets_seen,
        "y": [result_map.get((t, None), {}).get("mean_total_time", 0) for t in targets_seen],
        "text": [f"{result_map.get((t, None), {}).get('mean_total_time', 0):.1f}s" for t in targets_seen],
        "textposition": "outside",
        "name": "Baseline (no draft)",
        "type": "bar",
        "marker": {"color": BASELINE_COLOR},
    }]
    for draft in drafts_seen:
        y_vals, text_vals, hover_vals = [], [], []
        for t in targets_seen:
            r = result_map.get((t, draft))
            if r:
                v = r.get("mean_total_time", 0)
                acc = r.get("acceptance_rate") or 0
                y_vals.append(v)
                text_vals.append(f"{v:.1f}s")
                hover_vals.append(f"{draft}<br>{v:.2f}s avg response<br>{acc:.0%} acceptance")
            else:
                y_vals.append(None)
                text_vals.append("")
                hover_vals.append("")
        time_traces.append({
            "x": targets_seen, "y": y_vals, "text": text_vals,
            "textposition": "outside", "hovertext": hover_vals, "hoverinfo": "text",
            "name": draft, "type": "bar", "marker": {"color": draft_colors[draft]},
        })

    # --- Chart 2: Wall-time speedup vs baseline (%) ---
    speedup_traces = []
    for draft in drafts_seen:
        y_vals, text_vals, hover_vals = [], [], []
        has_data = False
        for t in targets_seen:
            sp = wall_speedup(t, draft)
            r = result_map.get((t, draft), {})
            acc = r.get("acceptance_rate") or 0
            if sp is not None:
                sign = "+" if sp >= 0 else ""
                y_vals.append(sp)
                text_vals.append(f"{sign}{sp:.0f}%")
                hover_vals.append(f"{draft}<br>{sign}{sp:.1f}% wall-time speedup<br>{acc:.0%} acceptance")
                has_data = True
            else:
                y_vals.append(None)
                text_vals.append("")
                hover_vals.append("")
        if has_data:
            speedup_traces.append({
                "x": targets_seen, "y": y_vals, "text": text_vals,
                "textposition": "outside", "hovertext": hover_vals, "hoverinfo": "text",
                "name": draft, "type": "bar", "marker": {"color": draft_colors[draft]},
            })

    # --- Chart 3: Draft acceptance rate (%) ---
    accept_traces = []
    for draft in drafts_seen:
        y_vals, text_vals = [], []
        has_data = False
        for t in targets_seen:
            r = result_map.get((t, draft))
            if r and r.get("acceptance_rate") is not None:
                acc = r["acceptance_rate"]
                y_vals.append(round(acc * 100, 1))
                text_vals.append(f"{acc:.0%}")
                has_data = True
            else:
                y_vals.append(None)
                text_vals.append("")
        if has_data:
            accept_traces.append({
                "x": targets_seen, "y": y_vals, "text": text_vals,
                "textposition": "outside",
                "name": draft, "type": "bar", "marker": {"color": draft_colors[draft]},
            })

    # --- Heatmap: wall-time speedup for every draft × target combo ---
    heatmap_z = []
    heatmap_customdata = []  # acceptance rate, surfaced via hovertemplate
    for draft in drafts_seen:
        row, cdata_row = [], []
        for t in targets_seen:
            sp = wall_speedup(t, draft)
            r = result_map.get((t, draft), {})
            acc = r.get("acceptance_rate")
            row.append(sp)
            cdata_row.append(round(acc * 100, 1) if acc is not None else None)
        heatmap_z.append(row)
        heatmap_customdata.append(cdata_row)

    # Compute colorscale bounds from actual data
    flat_z = [v for row in heatmap_z for v in row if v is not None]
    if flat_z:
        z_lo, z_hi = min(flat_z), max(flat_z)
        # Symmetric around 0 when negatives exist so the neutral colour sits at 0
        if z_lo < 0:
            bound = max(abs(z_lo), abs(z_hi))
            z_lo, z_hi = -bound, bound
        # Pad slightly and snap to a round number
        z_lo = min(z_lo - 5, -5)
        z_hi = max(z_hi + 5, 5)
        tick_step = max(10, round((z_hi - z_lo) / 5 / 10) * 10)
        tick_vals = list(range(int(z_lo), int(z_hi) + 1, tick_step))
        tick_text = [f"{v:+d}%" for v in tick_vals]
    else:
        z_lo, z_hi = -50, 100
        tick_vals = [-50, -25, 0, 25, 50, 75, 100]
        tick_text = [f"{v:+d}%" for v in tick_vals]

    # Layout helpers derived from actual content
    num_drafts = len(drafts_seen)
    heatmap_height = max(400, num_drafts * 40 + 120)
    left_margin = max(120, max((len(d) for d in drafts_seen), default=10) * 8)

    # Serialise everything for the template
    time_json             = json.dumps(time_traces)
    speedup_json          = json.dumps(speedup_traces)
    accept_json           = json.dumps(accept_traces)
    heatmap_z_json        = json.dumps(heatmap_z)
    heatmap_customdata_json = json.dumps(heatmap_customdata)
    targets_json          = json.dumps(targets_seen)
    drafts_json           = json.dumps(drafts_seen)
    z_lo_json             = json.dumps(z_lo)
    z_hi_json             = json.dumps(z_hi)
    tick_vals_json        = json.dumps(tick_vals)
    tick_text_json        = json.dumps(tick_text)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>draftbench - Speculative Decoding Sweep</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        h1 {{ text-align: center; color: #fff; margin-bottom: 10px; }}
        h2 {{ text-align: center; color: #888; font-weight: normal; margin-top: 0; }}
        .chart {{
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            margin: 30px 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        .chart-title {{
            color: #fff;
            font-size: 1.2em;
            margin-bottom: 10px;
            padding-left: 10px;
        }}
        .chart-note {{
            color: #666;
            font-size: 0.85em;
            padding-left: 10px;
            margin-bottom: 6px;
        }}
    </style>
</head>
<body>
    <h1>Speculative Decoding Benchmark</h1>
    <h2>{chart_subtitle}</h2>

    <div class="chart">
        <div class="chart-title">Mean Response Time (seconds) — lower is better</div>
        <div class="chart-note">Average total inference time per request</div>
        <div id="time-chart"></div>
    </div>

    <div class="chart">
        <div class="chart-title">Wall-Time Speedup vs Baseline — higher is better</div>
        <div class="chart-note">Based on total wall time across all benchmark runs</div>
        <div id="speedup-chart"></div>
    </div>

    <div class="chart">
        <div class="chart-title">Draft Acceptance Rate — higher is better</div>
        <div class="chart-note">Fraction of draft tokens accepted by the target model</div>
        <div id="accept-chart"></div>
    </div>

    <div class="chart">
        <div class="chart-title">Full Results Heatmap — wall-time speedup %</div>
        <div class="chart-note">Hover for acceptance rate. Red = slower than baseline.</div>
        <div id="heatmap-chart"></div>
    </div>

    <script>
        var darkLayout = {{
            paper_bgcolor: '#16213e',
            plot_bgcolor: '#16213e',
            font: {{ color: '#eee' }},
            xaxis: {{ gridcolor: '#2a3a5e', title: 'Target Model' }},
            yaxis: {{ gridcolor: '#2a3a5e' }},
        }};

        var barDefaults = {{
            ...darkLayout,
            barmode: 'group',
            legend: {{ orientation: 'h', y: -0.2, font: {{ color: '#eee' }} }},
            margin: {{ b: 100, t: 20 }},
            height: 400,
        }};

        Plotly.newPlot('time-chart', {time_json}, {{
            ...barDefaults,
            yaxis: {{ ...darkLayout.yaxis, title: 'Seconds per Request', rangemode: 'tozero' }},
        }}, {{ responsive: true }});

        Plotly.newPlot('speedup-chart', {speedup_json}, {{
            ...barDefaults,
            yaxis: {{ ...darkLayout.yaxis, title: 'Wall-Time Speedup (%)' }},
            shapes: [{{
                type: 'line', x0: 0, x1: 1, xref: 'paper',
                y0: 0, y1: 0, yref: 'y',
                line: {{ color: '#555', width: 2, dash: 'dash' }}
            }}],
        }}, {{ responsive: true }});

        Plotly.newPlot('accept-chart', {accept_json}, {{
            ...barDefaults,
            yaxis: {{ ...darkLayout.yaxis, title: 'Acceptance Rate (%)', range: [0, 105] }},
        }}, {{ responsive: true }});

        var heatmapTrace = [{{
            z: {heatmap_z_json},
            x: {targets_json},
            y: {drafts_json},
            customdata: {heatmap_customdata_json},
            hovertemplate: (
                '<b>%{{y}}</b> \u2192 %{{x}}<br>' +
                'Speedup: <b>%{{z:.1f}}%</b><br>' +
                'Acceptance: <b>%{{customdata:.1f}}%</b>' +
                '<extra></extra>'
            ),
            type: 'heatmap',
            colorscale: [
                [0,   '#dc3545'],
                [0.4, '#fd7e14'],
                [0.5, '#ffc107'],
                [0.7, '#28a745'],
                [1,   '#00ff88'],
            ],
            zmin: {z_lo_json},
            zmax: {z_hi_json},
            colorbar: {{
                title: 'Speedup %',
                tickfont:  {{ color: '#eee', size: 12 }},
                titlefont: {{ color: '#eee', size: 13 }},
                tickvals: {tick_vals_json},
                ticktext: {tick_text_json},
                len: 0.9,
            }},
            hoverongaps: false,
            xgap: 2,
            ygap: 2,
        }}];

        Plotly.newPlot('heatmap-chart', heatmapTrace, {{
            ...darkLayout,
            xaxis: {{
                ...darkLayout.xaxis,
                side: 'top',
                tickfont: {{ size: 13, color: '#eee' }},
                tickangle: 0,
            }},
            yaxis: {{
                ...darkLayout.yaxis,
                title: '',
                autorange: 'reversed',
                tickfont: {{ size: 12, color: '#eee' }},
                dtick: 1,
            }},
            margin: {{ l: {left_margin}, t: 60, b: 20, r: 100 }},
            height: {heatmap_height},
        }}, {{ responsive: true }});
    </script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)

    print(f"  Chart saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _generate_output_paths(config: dict) -> tuple[str, str]:
    """Generate output paths from config metadata."""
    hardware = config.get("hardware", "unknown")
    backend = config.get("backend", "unknown")
    name = config.get("name", "sweep")

    base = f"results/{hardware}_{backend}_{name}"
    return f"{base}.json", f"{base}.html"


def _run_config_file(config_path: str, results_path: str | None = None, chart_path: str | None = None):
    """Run a single config file and generate charts."""
    with open(config_path) as f:
        config = json.load(f)

    # Auto-generate paths from config metadata if not specified
    auto_results, auto_chart = _generate_output_paths(config)
    results_path = results_path or auto_results
    chart_path = chart_path or auto_chart
    log_path = results_path.replace(".json", ".log")

    tee = _Tee(log_path)
    try:
        print(f"\n{'#'*60}")
        print(f"  Loading config: {config_path}")
        print(f"  Log: {log_path}")
        print(f"{'#'*60}")

        results = run_sweep(config, results_path)

        print(f"\n{'='*60}")
        print(f"  Sweep complete. Results saved to {results_path}")
        print(f"{'='*60}\n")

        generate_chart(results, chart_path, config)
    finally:
        tee.close()

    return results_path, chart_path


def main():
    parser = argparse.ArgumentParser(
        description="Run speculative decoding benchmark sweep and generate charts.",
    )
    parser.add_argument("--config", help="Path to sweep config JSON file")
    parser.add_argument("--config-dir", help="Path to directory containing config files (runs all *.json except example_*.json)")
    parser.add_argument("--results", help="Path to results JSON file (auto-generated from config if not specified)")
    parser.add_argument("--chart", help="Path to output HTML chart (auto-generated from config if not specified)")
    parser.add_argument("--chart-only", action="store_true", help="Skip benchmarking, just generate chart from existing results")

    args = parser.parse_args()

    if args.chart_only:
        if not args.results:
            parser.error("--results is required when using --chart-only")
        if not os.path.isfile(args.results):
            print(f"Error: results file not found: {args.results}", file=sys.stderr)
            sys.exit(1)
        with open(args.results) as f:
            data = json.load(f)
        chart_path = args.chart or args.results.replace(".json", ".html")
        generate_chart(data["results"], chart_path, data)
        return

    # Handle --config-dir: run all configs in directory
    if args.config_dir:
        if not os.path.isdir(args.config_dir):
            print(f"Error: directory not found: {args.config_dir}", file=sys.stderr)
            sys.exit(1)

        # Find all JSON files, excluding example_*.json templates
        config_files = sorted(glob.glob(os.path.join(args.config_dir, "*.json")))
        config_files = [f for f in config_files if not os.path.basename(f).startswith("example_")]

        if not config_files:
            print(f"Error: no config files found in {args.config_dir}", file=sys.stderr)
            print(f"  (files matching example_*.json are excluded)", file=sys.stderr)
            sys.exit(1)

        print(f"\n{'#'*60}")
        print(f"  Running {len(config_files)} config(s) from {args.config_dir}")
        print(f"{'#'*60}")
        for i, cf in enumerate(config_files, 1):
            print(f"  [{i}] {os.path.basename(cf)}")

        completed = []
        for config_file in config_files:
            try:
                results_path, chart_path = _run_config_file(config_file)
                completed.append((config_file, results_path, chart_path))
            except Exception as e:
                print(f"\n  ERROR running {config_file}: {e}", file=sys.stderr)
                continue

        print(f"\n{'#'*60}")
        print(f"  All sweeps complete! {len(completed)}/{len(config_files)} succeeded")
        print(f"{'#'*60}")
        for cf, rp, cp in completed:
            print(f"  {os.path.basename(cf)}:")
            print(f"    Results: {rp}")
            print(f"    Chart:   {cp}")
        return

    # Handle single --config
    if not args.config:
        parser.error("--config or --config-dir is required (unless using --chart-only)")

    _run_config_file(args.config, args.results, args.chart)


if __name__ == "__main__":
    main()

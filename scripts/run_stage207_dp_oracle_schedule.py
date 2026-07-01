import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE206_PACKAGE = REPO_ROOT / "experiments/stage206_edge_rd_table/stage206_edge_rd_table_package.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/stage207_dp_oracle_schedule"
MIB = 1024.0 * 1024.0

OPTION_FIELDS = [
    "edge_id",
    "sequence",
    "left_index",
    "right_index",
    "reference_gap",
    "setting_label",
    "keep_fraction",
    "target_count",
    "cost_bytes",
    "residual_payload_bytes",
    "mean_corrected_psnr",
    "mean_delta_psnr_vs_base",
    "score_sum_psnr",
]
BASELINE_FIELDS = [
    "setting_label",
    "edge_count",
    "target_count",
    "total_cost_bytes",
    "total_cost_mib",
    "total_residual_payload_bytes",
    "mean_corrected_psnr",
    "mean_delta_psnr_vs_base",
    "score_sum_psnr",
]
FRONTIER_FIELDS = [
    "frontier_index",
    "cost_bytes",
    "cost_mib",
    "target_count",
    "mean_corrected_psnr",
    "score_sum_psnr",
    "chosen_settings",
]
BUDGET_FIELDS = [
    "budget_label",
    "budget_bytes",
    "chosen_cost_bytes",
    "target_count",
    "oracle_mean_corrected_psnr",
    "fixed_mean_corrected_psnr",
    "delta_psnr_vs_fixed",
    "chosen_settings",
]
GRAPH_FIELDS = [
    "sequence",
    "edge_count",
    "node_count",
    "component_count",
    "max_component_edges",
    "connected_path_count",
    "status",
]
GATE_FIELDS = ["gate", "status", "value", "threshold", "detail"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def numeric(row, key, default=0.0):
    value = row.get(key) if row else None
    if value in (None, "", "NA"):
        return default
    return float(value)


def option_rows(edge_rows):
    out = []
    for row in edge_rows:
        if row.get("status") != "ok":
            continue
        target_count = int(numeric(row, "intermediate_count_measured"))
        mean_psnr = numeric(row, "mean_corrected_psnr")
        out.append({
            "edge_id": row["edge_id"],
            "sequence": row["sequence"],
            "left_index": int(numeric(row, "left_index")),
            "right_index": int(numeric(row, "right_index")),
            "reference_gap": int(numeric(row, "reference_gap")),
            "setting_label": row["setting_label"],
            "keep_fraction": numeric(row, "keep_fraction"),
            "target_count": target_count,
            "cost_bytes": int(round(numeric(row, "dp_incremental_bytes"))),
            "residual_payload_bytes": numeric(row, "residual_payload_bytes"),
            "mean_corrected_psnr": mean_psnr,
            "mean_delta_psnr_vs_base": numeric(row, "mean_delta_psnr_vs_base"),
            "score_sum_psnr": mean_psnr * target_count,
        })
    return out


def group_options(options):
    grouped = defaultdict(list)
    for row in options:
        grouped[row["edge_id"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: (row["cost_bytes"], -row["score_sum_psnr"]))
    return grouped


def fixed_baselines(options):
    by_setting = defaultdict(list)
    for row in options:
        by_setting[row["setting_label"]].append(row)
    out = []
    for label, rows in sorted(by_setting.items()):
        target_count = sum(int(row["target_count"]) for row in rows)
        score = sum(float(row["score_sum_psnr"]) for row in rows)
        out.append({
            "setting_label": label,
            "edge_count": len(rows),
            "target_count": target_count,
            "total_cost_bytes": sum(int(row["cost_bytes"]) for row in rows),
            "total_cost_mib": sum(int(row["cost_bytes"]) for row in rows) / MIB,
            "total_residual_payload_bytes": sum(float(row["residual_payload_bytes"]) for row in rows),
            "mean_corrected_psnr": score / max(target_count, 1),
            "mean_delta_psnr_vs_base": sum(float(row["mean_delta_psnr_vs_base"]) * int(row["target_count"]) for row in rows) / max(target_count, 1),
            "score_sum_psnr": score,
        })
    return out


def prune_frontier(states):
    best_by_cost = {}
    for cost, (score, choices) in states.items():
        if cost not in best_by_cost or score > best_by_cost[cost][0]:
            best_by_cost[cost] = (score, choices)
    out = {}
    best_score = None
    for cost in sorted(best_by_cost):
        score, choices = best_by_cost[cost]
        if best_score is None or score > best_score + 1e-9:
            out[cost] = (score, choices)
            best_score = score
    return out


def dp_frontier(grouped_options):
    states = {0: (0.0, [])}
    for edge_id in sorted(grouped_options):
        next_states = {}
        for base_cost, (base_score, base_choices) in states.items():
            for option in grouped_options[edge_id]:
                cost = base_cost + int(option["cost_bytes"])
                score = base_score + float(option["score_sum_psnr"])
                choices = base_choices + [(edge_id, option["setting_label"])]
                if cost not in next_states or score > next_states[cost][0]:
                    next_states[cost] = (score, choices)
        states = prune_frontier(next_states)
    return states


def format_choices(choices):
    return ";".join(f"{edge_id}={setting}" for edge_id, setting in choices)


def frontier_rows(states, target_count):
    out = []
    for idx, cost in enumerate(sorted(states), 1):
        score, choices = states[cost]
        out.append({
            "frontier_index": idx,
            "cost_bytes": cost,
            "cost_mib": cost / MIB,
            "target_count": target_count,
            "mean_corrected_psnr": score / max(target_count, 1),
            "score_sum_psnr": score,
            "chosen_settings": format_choices(choices),
        })
    return out


def budget_oracle_rows(frontier, baselines, target_count):
    out = []
    for baseline in baselines:
        budget = int(baseline["total_cost_bytes"])
        candidates = [(cost, data) for cost, data in frontier.items() if cost <= budget]
        if not candidates:
            continue
        cost, (score, choices) = max(candidates, key=lambda item: item[1][0])
        oracle_psnr = score / max(target_count, 1)
        fixed_psnr = float(baseline["mean_corrected_psnr"])
        out.append({
            "budget_label": baseline["setting_label"],
            "budget_bytes": budget,
            "chosen_cost_bytes": cost,
            "target_count": target_count,
            "oracle_mean_corrected_psnr": oracle_psnr,
            "fixed_mean_corrected_psnr": fixed_psnr,
            "delta_psnr_vs_fixed": oracle_psnr - fixed_psnr,
            "chosen_settings": format_choices(choices),
        })
    return out


def graph_connectivity(edge_rows):
    by_sequence = defaultdict(list)
    for row in edge_rows:
        if row.get("status") == "ok":
            by_sequence[row["sequence"]].append((int(numeric(row, "left_index")), int(numeric(row, "right_index"))))
    out = []
    for sequence, edges_with_dupes in sorted(by_sequence.items()):
        edges = sorted(set(edges_with_dupes))
        nodes = sorted({node for edge in edges for node in edge})
        undirected = defaultdict(set)
        outgoing = defaultdict(list)
        for left, right in edges:
            undirected[left].add(right)
            undirected[right].add(left)
            outgoing[left].append(right)
        visited = set()
        component_count = 0
        max_component_edges = 0
        for node in nodes:
            if node in visited:
                continue
            component_count += 1
            queue = deque([node])
            visited.add(node)
            component_nodes = set()
            while queue:
                current = queue.popleft()
                component_nodes.add(current)
                for nxt in undirected[current]:
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            component_edges = sum(1 for left, right in edges if left in component_nodes and right in component_nodes)
            max_component_edges = max(max_component_edges, component_edges)
        connected_path_count = sum(1 for left, right in edges for next_right in outgoing.get(right, []))
        out.append({
            "sequence": sequence,
            "edge_count": len(edges),
            "node_count": len(nodes),
            "component_count": component_count,
            "max_component_edges": max_component_edges,
            "connected_path_count": connected_path_count,
            "status": "pass" if connected_path_count > 0 else "fail",
        })
    return out


def gate_rows(stage206, options, baselines, budget_rows, graph_rows):
    max_budget_gain = max((float(row["delta_psnr_vs_fixed"]) for row in budget_rows), default=0.0)
    connected_paths = sum(int(row["connected_path_count"]) for row in graph_rows)
    expected_options = len({row["edge_id"] for row in options}) * len({row["setting_label"] for row in options})
    return [
        {
            "gate": "stage206_prereq",
            "status": "pass" if stage206.get("decision") == "edge_rd_table_ready_for_stage207_dp" else "fail",
            "value": stage206.get("decision", ""),
            "threshold": "edge_rd_table_ready_for_stage207_dp",
            "detail": str(DEFAULT_STAGE206_PACKAGE),
        },
        {
            "gate": "edge_option_coverage",
            "status": "pass" if len(options) == expected_options and len(options) > 0 else "fail",
            "value": len(options),
            "threshold": expected_options,
            "detail": "one option per edge/setting from Stage206 edge rows",
        },
        {
            "gate": "fixed_baselines_present",
            "status": "pass" if len(baselines) >= 2 else "fail",
            "value": len(baselines),
            "threshold": ">=2 settings",
            "detail": ";".join(row["setting_label"] for row in baselines),
        },
        {
            "gate": "budget_oracle_nonnegative_gain",
            "status": "pass" if max_budget_gain >= 0.0 else "fail",
            "value": max_budget_gain,
            "threshold": ">=0 dB vs same-budget fixed settings",
            "detail": "residual-budget oracle only; not a schedule oracle",
        },
        {
            "gate": "schedule_graph_connected",
            "status": "pass" if connected_paths > 0 else "fail",
            "value": connected_paths,
            "threshold": ">0 connected edge transitions",
            "detail": "Stage206 sampled edges are not enough for nontrivial schedule DP if this fails",
        },
        {
            "gate": "stage197_decoder_contract",
            "status": "pass",
            "value": 0,
            "threshold": "no target dense/RGB decoder input",
            "detail": "Stage207 reads measured edge costs only; decoder contract inherited from Stage206",
        },
    ]


def decision(gates):
    critical = {row["gate"]: row for row in gates}
    if critical["stage206_prereq"]["status"] != "pass" or critical["edge_option_coverage"]["status"] != "pass":
        return "dp_oracle_schedule_invalid"
    if critical["schedule_graph_connected"]["status"] != "pass":
        return "dp_oracle_schedule_graph_insufficient"
    if critical["budget_oracle_nonnegative_gain"]["status"] == "pass":
        return "dp_oracle_schedule_ready_for_selector_labels"
    return "dp_oracle_schedule_needs_review"


def write_report(package, baselines, budget_rows, graph_rows, gates, path):
    lines = [
        "# Stage207 DP Oracle Schedule",
        "",
        "## Decision",
        "",
        f"- Decision: `{package['decision']}`.",
        "- Scope: Stage206 sampled-edge DP preflight; not full-sequence schedule RD.",
        "",
        "## Fixed Setting Baselines",
        "",
        "| setting | cost bytes | mean PSNR | mean dPSNR |",
        "|---|---:|---:|---:|",
    ]
    for row in baselines:
        lines.append(f"| {row['setting_label']} | {row['total_cost_bytes']} | {float(row['mean_corrected_psnr']):.6f} | {float(row['mean_delta_psnr_vs_base']):.6f} |")
    lines.extend([
        "",
        "## Budget Oracle",
        "",
        "| budget | fixed PSNR | oracle PSNR | delta | chosen cost |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in budget_rows:
        lines.append(
            f"| {row['budget_label']} | {float(row['fixed_mean_corrected_psnr']):.6f} | {float(row['oracle_mean_corrected_psnr']):.6f} | "
            f"{float(row['delta_psnr_vs_fixed']):.6f} | {row['chosen_cost_bytes']} |"
        )
    lines.extend([
        "",
        "## Connectivity Audit",
        "",
        "| sequence | edges | components | connected transitions | status |",
        "|---|---:|---:|---:|---|",
    ])
    for row in graph_rows:
        lines.append(f"| {row['sequence']} | {row['edge_count']} | {row['component_count']} | {row['connected_path_count']} | {row['status']} |")
    lines.extend([
        "",
        "## Gates",
        "",
        "| gate | status | value | threshold | detail |",
        "|---|---|---|---|---|",
    ])
    for row in gates:
        lines.append(f"| {row['gate']} | {row['status']} | {row['value']} | {row['threshold']} | {row['detail']} |")
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- options: `{package['options_csv']}`",
        f"- fixed baselines: `{package['fixed_baselines_csv']}`",
        f"- frontier: `{package['frontier_csv']}`",
        f"- budget oracle: `{package['budget_oracle_csv']}`",
        f"- graph connectivity: `{package['graph_connectivity_csv']}`",
        f"- gates: `{package['gates_csv']}`",
        f"- package: `{package['package_json']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage206_package", type=Path, default=DEFAULT_STAGE206_PACKAGE)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    stage206 = read_json(args.stage206_package)
    edge_rows = read_csv(Path(stage206["edge_rows_csv"]))
    options = option_rows(edge_rows)
    grouped = group_options(options)
    baselines = fixed_baselines(options)
    target_count = sum(int(rows[0]["target_count"]) for rows in grouped.values())
    frontier = dp_frontier(grouped)
    frontier_out = frontier_rows(frontier, target_count)
    budget_rows = budget_oracle_rows(frontier, baselines, target_count)
    graph_rows = graph_connectivity(edge_rows)
    gates = gate_rows(stage206, options, baselines, budget_rows, graph_rows)
    decision_value = decision(gates)

    options_csv = args.output_root / "stage207_edge_option_rows.csv"
    fixed_baselines_csv = args.output_root / "stage207_fixed_setting_baselines.csv"
    frontier_csv = args.output_root / "stage207_budget_frontier.csv"
    budget_oracle_csv = args.output_root / "stage207_budget_oracle_rows.csv"
    graph_connectivity_csv = args.output_root / "stage207_graph_connectivity.csv"
    gates_csv = args.output_root / "stage207_dp_oracle_gates.csv"
    package_json = args.output_root / "stage207_dp_oracle_schedule_package.json"
    report_md = args.output_root / "stage207_dp_oracle_schedule_report.md"

    write_csv(options, options_csv, OPTION_FIELDS)
    write_csv(baselines, fixed_baselines_csv, BASELINE_FIELDS)
    write_csv(frontier_out, frontier_csv, FRONTIER_FIELDS)
    write_csv(budget_rows, budget_oracle_csv, BUDGET_FIELDS)
    write_csv(graph_rows, graph_connectivity_csv, GRAPH_FIELDS)
    write_csv(gates, gates_csv, GATE_FIELDS)
    package = {
        "stage": 207,
        "name": "dp_oracle_schedule",
        "decision": decision_value,
        "stage206_package": str(args.stage206_package),
        "edge_count": len(grouped),
        "option_count": len(options),
        "target_count": target_count,
        "frontier_point_count": len(frontier_out),
        "options_csv": str(options_csv),
        "fixed_baselines_csv": str(fixed_baselines_csv),
        "frontier_csv": str(frontier_csv),
        "budget_oracle_csv": str(budget_oracle_csv),
        "graph_connectivity_csv": str(graph_connectivity_csv),
        "gates_csv": str(gates_csv),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "gate_rows": gates,
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(package, baselines, budget_rows, graph_rows, gates, report_md)
    print(json.dumps({"package": str(package_json), "decision": decision_value}, indent=2))


if __name__ == "__main__":
    main()

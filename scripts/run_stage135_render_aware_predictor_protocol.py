import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE134_PACKAGE = REPO_ROOT / "experiments/stage134_mlp_render_regression_diagnostic/stage134_mlp_render_regression_diagnostic_package.json"
DEFAULT_SUMMARY_ROOT = REPO_ROOT / "experiments/stage135_render_aware_predictor_protocol"

SCALE_CANDIDATES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
SETTINGS = [
    {"label": "q4_top20", "role": "primary", "keep_fraction": 0.2, "side_bits": 4},
    {"label": "q4_top10", "role": "low_rate", "keep_fraction": 0.1, "side_bits": 4},
]

PROTOCOL_FIELDS = [
    "item",
    "value",
]


def write_csv(rows, path, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_report(protocol, package, path):
    lines = [
        "# Stage135 Render-Aware Predictor Protocol",
        "",
        "## Protocol",
        "",
        "- Predictor family: adapter-delta selected residual predictor.",
        "- Residual rule: `scaled_residual = adapter_delta_scale * (adapter_attrs - linear_attrs)` at deterministic endpoint-diff indices.",
        "- Scale candidates: `0.0, 0.25, 0.5, 0.75, 1.0, 1.25`.",
        "- Settings: q4/top20 primary and q4/top10 low-rate.",
        "- Target RGB is used only for offline protocol selection and validation.",
        "- Target dense anchors and teacher residuals are not used.",
        "",
        "## Acceptance Criteria",
        "",
        "- Choose the scale/setting with the highest rendered PSNR on Stage136 smoke.",
        "- Keep only candidates with non-negative mean delta vs linear base on Stage137 broader validation.",
        "- If no scale beats Stage132, retain Stage132 adapter-delta q4/top20 policy.",
        "",
        "## Outputs",
        "",
        f"- protocol CSV: `{package['protocol_csv']}`",
        f"- protocol JSON: `{package['protocol_json']}`",
        f"- package JSON: `{package['package_json']}`",
        f"- report Markdown: `{package['report_md']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage134_package", type=Path, default=DEFAULT_STAGE134_PACKAGE)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stage134 = read_json(args.stage134_package)
    protocol = {
        "stage": 135,
        "protocol_name": "render_aware_adapter_delta_scale_calibration_v1",
        "source_diagnostic": str(args.stage134_package),
        "source_diagnostic_mode": stage134["mode"],
        "predictor_family": "adapter_delta_selected_predictor",
        "residual_rule": "adapter_delta_scale * (adapter_attrs - linear_attrs) at deterministic endpoint-diff selected indices",
        "scale_candidates": SCALE_CANDIDATES,
        "settings": SETTINGS,
        "forbidden_training_or_decoder_inputs": [
            "teacher residual side-info",
            "target dense anchor as decoder input",
            "target residual as decoder input",
            "target RGB as decoder input",
            "transmitted selected indices",
            "transmitted residual values",
        ],
        "offline_validation_inputs": ["target RGB for rendered metric only"],
        "acceptance_criteria": [
            "Stage136 chooses the best rendered scale/setting on a smoke slice.",
            "Stage137 keeps only candidates with non-negative broader mean delta vs linear base.",
            "If no calibrated candidate improves Stage132, keep Stage132 q4/top20.",
        ],
        "next_stages": {
            "stage136": "render-aware scale calibration smoke",
            "stage137": "broader validation of selected scale",
            "stage138": "policy update if broader validation is successful",
        },
    }
    protocol_csv = args.summary_root / "stage135_render_aware_predictor_protocol.csv"
    protocol_json = args.summary_root / "stage135_render_aware_predictor_protocol.json"
    package_json = args.summary_root / "stage135_render_aware_predictor_protocol_package.json"
    report_md = args.summary_root / "stage135_render_aware_predictor_protocol_report.md"
    rows = [
        {"item": "protocol_name", "value": protocol["protocol_name"]},
        {"item": "predictor_family", "value": protocol["predictor_family"]},
        {"item": "scale_candidates", "value": " ".join(str(x) for x in SCALE_CANDIDATES)},
        {"item": "settings", "value": " ".join(row["label"] for row in SETTINGS)},
        {"item": "forbidden_inputs", "value": "; ".join(protocol["forbidden_training_or_decoder_inputs"])},
    ]
    write_csv(rows, protocol_csv, PROTOCOL_FIELDS)
    protocol_json.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    package = {
        "stage": 135,
        "mode": "render-aware predictor protocol package",
        "protocol_csv": str(protocol_csv),
        "protocol_json": str(protocol_json),
        "package_json": str(package_json),
        "report_md": str(report_md),
        "protocol_name": protocol["protocol_name"],
        "scale_candidates": SCALE_CANDIDATES,
        "settings": SETTINGS,
    }
    package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    write_report(protocol, package, report_md)
    print(json.dumps({"package": str(package_json), "protocol": protocol["protocol_name"]}, indent=2))


if __name__ == "__main__":
    main()

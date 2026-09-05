"""Render the application figures from verified toy and Qwen summaries."""

import argparse
import json
import statistics
import subprocess
import tempfile
from pathlib import Path


METHODS = (
    "regular_sft", "multipass_z", "z_supervision",
    "iit", "full_scaffold", "staged_removal",
)
LABELS = {
    "regular_sft": "Regular SFT",
    "multipass_z": "Multi-pass Z",
    "z_supervision": "Z supervision",
    "iit": "IIT",
    "full_scaffold": "Full scaffold",
    "staged_removal": "Scaffold removed",
}


def quote(value):
    return "'{}'".format(str(value).replace("'", "''"))


def run_plot(body, output):
    output = Path(output)
    terminals = (
        ("svg size 920,520 dynamic enhanced font 'Arial,12' background rgb 'white'", output.with_suffix(".svg")),
        ("pngcairo size 1840,1040 enhanced font 'Arial,24' background rgb 'white'", output.with_suffix(".png")),
    )
    for terminal, destination in terminals:
        script = "set terminal {}\nset output {}\n{}".format(
            terminal, quote(destination), body,
        )
        subprocess.run(
            ["gnuplot"], input=script, universal_newlines=True, check=True
        )


def mean(values):
    return statistics.mean(float(value) for value in values)


def final_rows(summary):
    rows = []
    for method in METHODS:
        results = [run["results"][method] for run in summary["runs"]]
        heldout = mean(row["heldout"]["accuracy"] for row in results)
        patch = mean(row["patch_accuracy"] for row in results)
        if method == "multipass_z":
            refusal, over_refusal = 1.0, 0.0
        else:
            refusal = mean(row["heldout"]["refusal_recall"] for row in results)
            over_refusal = mean(
                row["heldout"]["benign_over_refusal"] for row in results
            )
        rows.append({
            "method": method, "heldout": heldout, "patch": patch,
            "refusal": refusal, "over_refusal": over_refusal,
        })
    return rows


def validate(rows, qwen):
    assert len(rows) == len(METHODS)
    assert all(
        0.0 <= row[key] <= 1.0
        for row in rows
        for key in ("heldout", "patch", "refusal", "over_refusal")
    )
    assert all(
        0.0 <= qwen[key]["ood_macro_accuracy"] <= 1.0
        for key in ("predicted", "oracle", "mismatched")
    )


def source_name(path):
    parts = Path(path).parts
    return "/".join(parts[-2:])


def grouped_comparison(rows, output, temporary):
    data = temporary / "grouped.tsv"
    data.write_text("\n".join(
        "{}\t{}\t{:.8f}\t{:.8f}".format(
            index, LABELS[row["method"]], row["heldout"], row["patch"]
        ) for index, row in enumerate(rows)
    ) + "\n")
    run_plot("""
set datafile separator '\t'
set title 'OOD performance and causal dependence'
set ylabel 'Accuracy'
set yrange [0:1.08]
set grid ytics lc rgb '#dddddd'
set boxwidth 0.32
set style fill solid 0.85 border rgb '#333333'
set xtics rotate by -20
set key outside right center
plot {data} using ($1-0.18):3:xtic(2) with boxes lc rgb '#4472c4' title 'OOD action accuracy', \
     '' using ($1+0.18):4 with boxes lc rgb '#ed7d31' title 'Causal patching'
""".format(data=quote(data)), output / "ood_and_causal_patching")


def causal_scatter(rows, output, temporary):
    data = temporary / "causal_scatter.tsv"
    data.write_text("\n".join(
        "{:.8f}\t{:.8f}".format(row["patch"], row["heldout"])
        for row in rows
    ) + "\n")
    labels = (
        (0.112, 0.846, "Regular SFT"),
        (0.169, 0.819, "Z supervision"),
        (0.886, 0.920, "IIT"),
        (0.984, 0.955, "Full scaffold"),
        (0.980, 1.035, "Multi-pass Z"),
        (0.786, 0.730, "Scaffold removed"),
    )
    commands = "\n".join(
        "set label {} at {:.3f},{:.3f} left".format(quote(label), x, y)
        for x, y, label in labels
    )
    run_plot("""
set datafile separator '\t'
set title 'Causal control versus held-out performance'
set xlabel 'Causal patching accuracy'
set ylabel 'OOD action accuracy'
set xrange [0:1.08]
set yrange [0:1.08]
set grid lc rgb '#dddddd'
unset key
{commands}
plot {data} using 1:2 with points pt 7 ps 1.8 lc rgb '#4472c4'
""".format(commands=commands, data=quote(data)), output / "causal_control_vs_ood")


def safety_tradeoff(rows, output, temporary):
    data = temporary / "tradeoff.tsv"
    data.write_text("\n".join(
        "{:.8f}\t{:.8f}".format(row["over_refusal"], row["refusal"])
        for row in rows
    ) + "\n")
    labels = (
        (0.002, 1.008, "Multi-pass / Full scaffold"),
        (0.041, 0.868, "Regular SFT"),
        (0.060, 0.978, "IIT"),
        (0.090, 0.852, "Z supervision"),
        (0.101, 0.708, "Scaffold removed"),
    )
    commands = "\n".join(
        "set label {} at {:.3f},{:.3f} left".format(quote(label), x, y)
        for x, y, label in labels
    )
    run_plot("""
set datafile separator '\t'
set title 'Safety trade-off on held-out styles'
set xlabel 'Benign over-refusal'
set ylabel 'Refusal recall'
set xrange [-0.015:0.30]
set yrange [0.60:1.025]
set grid lc rgb '#dddddd'
unset key
{commands}
plot {data} using 1:2 with points pt 7 ps 1.5 lc rgb '#70ad47'
""".format(commands=commands, data=quote(data)), output / "safety_tradeoff")


def seed_level(summary, output, temporary):
    heldout = temporary / "seed_heldout.tsv"
    patch = temporary / "seed_patch.tsv"
    means = temporary / "seed_means.tsv"
    heldout_rows, patch_rows, mean_rows = [], [], []
    for index, method in enumerate(METHODS):
        results = [run["results"][method] for run in summary["runs"]]
        hs = [row["heldout"]["accuracy"] for row in results]
        ps = [row["patch_accuracy"] for row in results]
        heldout_rows.extend("{:.3f}\t{:.8f}".format(index - 0.12, value) for value in hs)
        patch_rows.extend("{:.3f}\t{:.8f}".format(index + 0.12, value) for value in ps)
        mean_rows.append("{}\t{:.8f}\t{:.8f}".format(index, mean(hs), mean(ps)))
    heldout.write_text("\n".join(heldout_rows) + "\n")
    patch.write_text("\n".join(patch_rows) + "\n")
    means.write_text("\n".join(mean_rows) + "\n")
    xtics = ", ".join(
        "{} {}".format(quote(LABELS[method]), index)
        for index, method in enumerate(METHODS)
    )
    run_plot("""
set datafile separator '\t'
set title 'Seed-level OOD and causal results'
set ylabel 'Accuracy'
set yrange [0:1.08]
set xrange [-0.5:5.5]
set grid ytics lc rgb '#dddddd'
set xtics ({xtics}) rotate by -20
set key outside right center
plot {heldout} using 1:2 with points pt 7 ps 1.2 lc rgb '#4472c4' title 'OOD seeds', \
     {patch} using 1:2 with points pt 5 ps 1.2 lc rgb '#ed7d31' title 'Patching seeds', \
     {means} using ($1-0.12):2 with points pt 2 ps 2 lc rgb '#17365d' title 'OOD mean', \
     '' using ($1+0.12):3 with points pt 2 ps 2 lc rgb '#984807' title 'Patching mean'
""".format(
        xtics=xtics, heldout=quote(heldout), patch=quote(patch),
        means=quote(means),
    ), output / "seed_level_results")


def scaffold_trajectory(summary, output, temporary):
    by_seed = []
    for run in summary["runs"]:
        points = {}
        for row in run["results"]["staged_removal"]["history"]:
            points[int(row["update"])] = row
        by_seed.append(points)
    updates = sorted(set.intersection(*(set(points) for points in by_seed)))
    data = temporary / "scaffold.tsv"
    data.write_text("\n".join(
        "{}\t{:.8f}\t{:.8f}".format(
            update,
            mean(points[update]["heldout"]["accuracy"] for points in by_seed),
            mean(points[update]["patch_accuracy"] for points in by_seed),
        ) for update in updates
    ) + "\n")
    boundaries = (
        (1500, "graph cut off"), (2250, "hard to soft"),
        (3000, "start blend"), (4500, "canonical Z off"),
        (5000, "L_Z off"),
    )
    markers = "\n".join(
        "set arrow from {0}, graph 0 to {0}, graph 1 nohead dt 2 lc rgb '#999999'\n"
        "set label {1} at {0},0.445 rotate by 90 right font ',9' textcolor rgb '#666666'".format(
            position, quote(label)
        ) for position, label in boundaries
    )
    run_plot("""
set datafile separator '\t'
set title 'Performance while removing the scaffold'
set xlabel 'Fine-tuning updates'
set ylabel 'Accuracy'
set xrange [0:6100]
set yrange [0.40:1.04]
set grid ytics lc rgb '#dddddd'
set key bottom left
{markers}
plot {data} using 1:2 with linespoints lw 2 pt 7 lc rgb '#4472c4' title 'OOD action accuracy', \
     '' using 1:3 with linespoints lw 2 pt 5 lc rgb '#ed7d31' title 'Causal patching'
""".format(markers=markers, data=quote(data)), output / "scaffold_removal_trajectory")


def qwen_intervention(summary, output, temporary):
    data = temporary / "qwen.tsv"
    rows = (
        ("Predicted Z", summary["predicted"]["ood_macro_accuracy"]),
        ("Correct oracle Z", summary["oracle"]["ood_macro_accuracy"]),
        ("Reversed Z", summary["mismatched"]["ood_macro_accuracy"]),
    )
    data.write_text("\n".join(
        "{}\t{:.8f}".format(label, value) for label, value in rows
    ) + "\n")
    changed = 100 * summary["causal_control"]["decision_changed_under_mismatched_z"]
    run_plot("""
set datafile separator '\t'
set title 'Qwen: policy decision depends causally on Z'
set ylabel 'OOD decision accuracy'
set yrange [0:1.10]
set grid ytics lc rgb '#dddddd'
set boxwidth 0.62
set style fill solid 0.85 border rgb '#333333'
unset key
set label 'Decisions changed under reversed Z: {changed:.0f}%' at graph 0.97,0.32 right
plot {data} using 0:2:xtic(1) with boxes lc rgb '#a64d79', \
     '' using 0:2:(sprintf('%.1f%%',100*$2)) with labels offset 0,1
""".format(changed=changed, data=quote(data)), output / "qwen_z_intervention")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy-summary", required=True)
    parser.add_argument("--qwen-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    toy = json.loads(Path(args.toy_summary).read_text())
    qwen = json.loads(Path(args.qwen_summary).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = final_rows(toy)
    validate(rows, qwen)
    (output / "plot_data.json").write_text(json.dumps({
        "toy_summary": source_name(args.toy_summary),
        "qwen_summary": source_name(args.qwen_summary),
        "toy_final": rows,
        "qwen_decision": {
            "predicted": qwen["predicted"]["ood_macro_accuracy"],
            "oracle": qwen["oracle"]["ood_macro_accuracy"],
            "reversed": qwen["mismatched"]["ood_macro_accuracy"],
            "reversed_decision_change": qwen["causal_control"][
                "decision_changed_under_mismatched_z"
            ],
        },
        "seed_count": len(toy["runs"]),
    }, indent=2) + "\n")
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        grouped_comparison(rows, output, temporary)
        causal_scatter(rows, output, temporary)
        safety_tradeoff(rows, output, temporary)
        seed_level(toy, output, temporary)
        scaffold_trajectory(toy, output, temporary)
        qwen_intervention(qwen, output, temporary)


if __name__ == "__main__":
    main()

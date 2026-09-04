"""Aggregate matched-method results and render publication-ready SVG plots."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path


LABELS = {
    "regular_sft": "Regular SFT",
    "multipass_z": "Multi-pass Z",
    "z_supervision": "Z supervision",
    "iit": "IIT",
    "full_scaffold": "Full scaffold",
    "staged_removal": "Scaffold removed",
}


def mean_sd(values):
    values = [float(value) for value in values]
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def aggregate(paths):
    reports = [json.loads(Path(path).read_text()) for path in paths]
    methods = [
        method for method in LABELS
        if any(method in report["methods"] for report in reports)
    ]
    runs = [run for report in reports for run in report["runs"]]
    result = {"methods": methods, "final": {}, "curves": {}}
    for method in methods:
        branches = [
            run["results"][method] for run in runs
            if method in run["results"]
        ]
        result["final"][method] = {
            "seed_count": len(branches),
            "heldout_accuracy": mean_sd([row["heldout"]["accuracy"] for row in branches]),
            "patch_accuracy": mean_sd([row["patch_accuracy"] for row in branches]),
            "atomic_accuracy": mean_sd([row["atomic_accuracy"] for row in branches]),
        }
        by_update = {}
        for branch in branches:
            for point in branch["history"]:
                by_update.setdefault(int(point["update"]), []).append(
                    float(point["heldout"]["accuracy"])
                )
        result["curves"][method] = [
            {"update": update, **mean_sd(values)}
            for update, values in sorted(by_update.items())
            if len(values) == len(branches)
        ]
    return result


def quote(path):
    return "'{}'".format(str(path).replace("'", "''"))


def run_gnuplot(script):
    executable = shutil.which("gnuplot")
    if executable is None:
        raise RuntimeError("gnuplot is required to render SVG files")
    subprocess.run([executable], input=script, text=True, check=True)


def metric_plot(aggregate_result, metric, title, ylabel, output, temporary):
    data = temporary / "{}.tsv".format(metric)
    rows = []
    for method in aggregate_result["methods"]:
        value = aggregate_result["final"][method][metric]
        rows.append("{}\t{:.8f}\t{:.8f}".format(
            LABELS[method], value["mean"], value["sd"]
        ))
    data.write_text("\n".join(rows) + "\n")
    run_gnuplot("""
set terminal svg size 920,520 dynamic enhanced font 'Arial,12' background rgb 'white'
set output {output}
set datafile separator '\t'
set title {title}
set ylabel {ylabel}
set yrange [0:1.08]
set ytics 0.1
set grid ytics lc rgb '#dddddd'
set boxwidth 0.65
set style fill solid 0.85 border rgb '#333333'
set xtics rotate by -22
unset key
plot {data} using 0:2:xtic(1) with boxes lc rgb '#4472c4', \
     '' using 0:2:3 with yerrorbars pt 0 lc rgb '#222222'
""".format(
        output=quote(output), data=quote(data), title=quote(title), ylabel=quote(ylabel)
    ))


def curve_plot(aggregate_result, output, temporary, task_label="permutation composition"):
    plots = []
    colors = ("#555555", "#70ad47", "#ed7d31", "#a64d79", "#5b9bd5", "#c00000")
    for index, method in enumerate(aggregate_result["methods"]):
        data = temporary / "curve_{}.tsv".format(method)
        data.write_text("\n".join(
            "{}\t{:.8f}\t{:.8f}".format(point["update"], point["mean"], point["sd"])
            for point in aggregate_result["curves"][method]
        ) + "\n")
        plots.append("{} using 1:2:3 with yerrorlines lw 2 pt 5 ps 0.5 lc rgb '{}' title {}".format(
            quote(data), colors[index], quote(LABELS[method])
        ))
    run_gnuplot("""
set terminal svg size 940,560 dynamic enhanced font 'Arial,12' background rgb 'white'
set output {output}
set datafile separator '\t'
set title 'Held-out task performance during fine-tuning'
set xlabel 'Fine-tuning updates'
set ylabel 'Held-out accuracy'
set xrange [0:*]
set yrange [0:1.08]
set ytics 0.1
set grid ytics lc rgb '#dddddd'
set key outside right center
plot {plots}
""".format(output=quote(output), plots=", \\\n     ".join(plots)))


def causal_scatter(aggregate_result, output, temporary):
    data = temporary / "causal_scatter.tsv"
    labels = temporary / "causal_scatter_labels.tsv"
    data.write_text("\n".join(
        "{:.8f}\t{:.8f}\t{}".format(
            aggregate_result["final"][method]["patch_accuracy"]["mean"],
            aggregate_result["final"][method]["heldout_accuracy"]["mean"],
            LABELS[method],
        )
        for method in aggregate_result["methods"]
    ) + "\n")
    labels.write_text("\n".join(
        "{:.8f}\t{:.8f}\t{}".format(
            aggregate_result["final"][method]["patch_accuracy"]["mean"],
            aggregate_result["final"][method]["heldout_accuracy"]["mean"],
            LABELS[method],
        )
        for method in aggregate_result["methods"]
        if method not in {"multipass_z", "full_scaffold"}
    ) + "\n")
    run_gnuplot("""
set terminal svg size 760,560 dynamic enhanced font 'Arial,12' background rgb 'white'
set output {output}
set datafile separator '\t'
set title 'Causal control versus held-out performance'
set xlabel 'Causal patching accuracy'
set ylabel 'Held-out accuracy'
set xrange [0:1.08]
set yrange [0:1.08]
set xtics 0.1
set ytics 0.1
set grid lc rgb '#dddddd'
unset key
set label 'Multi-pass Z' at 0.98,1.045 right
set label 'Full scaffold' at 0.98,0.965 right
plot {data} using 1:2 with points pt 7 ps 1.5 lc rgb '#4472c4', \
     {labels} using 1:2:3 with labels offset char 0.7,0.7 left
""".format(output=quote(output), data=quote(data), labels=quote(labels)))


def render(paths, output_dir, task_label="permutation composition"):
    result = aggregate(paths)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "aggregate.json").write_text(json.dumps(result, indent=2) + "\n")
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        metric_plot(result, "heldout_accuracy", "Held-out {}".format(task_label),
                    "Accuracy", output / "heldout_accuracy.svg", temporary)
        metric_plot(result, "patch_accuracy", "Causal interchange intervention",
                    "Crossed-target accuracy", output / "causal_patching.svg", temporary)
        metric_plot(result, "atomic_accuracy", "Retention of pretrained atomic operations",
                    "Atomic accuracy", output / "atomic_retention.svg", temporary)
        curve_plot(result, output / "learning_curves.svg", temporary)
        causal_scatter(result, output / "causal_control_vs_ood.svg", temporary)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+")
    parser.add_argument("--output", default="toy/arithmetic/within_task/figures/matched_methods")
    parser.add_argument("--task-label", default="permutation composition")
    args = parser.parse_args()
    render(args.summaries, args.output, args.task_label)


if __name__ == "__main__":
    main()

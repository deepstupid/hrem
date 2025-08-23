import os
import subprocess
import yaml
import json
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    smoke_test = args.smoke_test

    # --- Environment Setup ---
    print("Setting up the environment...")
    # The script will inherit the environment, so no need to set DEVICE explicitly

    # --- Data Preparation ---
    print("Preparing dataset...")
    data_dir = "data/arc-smoke" if smoke_test else "data/arc-full"
    if smoke_test:
        print("Running in smoke test mode. Using a small dataset.")
        subprocess.run([
            "python", "dataset/build_arc_dataset.py",
            f"--output-dir={data_dir}",
            "--num-aug=0"
        ], check=True)
    else:
        print("Running in full mode. Using the complete dataset.")
        subprocess.run([
            "python", "dataset/build_arc_dataset.py",
            f"--output-dir={data_dir}",
            "--num-aug=1000"
        ], check=True)

    # --- Training and Evaluation ---
    print("Starting training and evaluation...")

    epochs = 1 if smoke_test else 20000
    eval_interval = 1 if smoke_test else 2000

    # Create HREM config from HRM config
    hrem_config_path = "config/arch/hrem_v1.yaml"
    if not os.path.exists(hrem_config_path):
        with open("config/arch/hrm_v1.yaml", "r") as f:
            hrm_config = yaml.safe_load(f)

        hrem_config = hrm_config.copy()
        hrem_config["name"] = "hrm.hrem@HREM"
        hrem_config["use_memory"] = True
        hrem_config["m_loc"] = 128
        hrem_config["d_mem"] = 128
        hrem_config["top_k"] = 4
        hrem_config["sparse_addressing"] = True
        hrem_config["use_location_addressing"] = True

        with open(hrem_config_path, "w") as f:
            yaml.dump(hrem_config, f)

    models_to_run = {
        "HRM": "hrm_v1",
        "HREM": "hrem_v1",
    }

    for model_name, config_name in models_to_run.items():
        print(f"--- Running experiment for {model_name} ---")
        log_path = f"results_{model_name}.json"

        command = [
            "python", "pretrain.py",
            f"data_path={data_dir}",
            f"epochs={epochs}",
            f"eval_interval={eval_interval}",
            f"arch={config_name}",
            f"log_path={log_path}",
            "+project_name=HREM_vs_HRM",
            f"+run_name={model_name}_smoke_test_{smoke_test}"
        ]
        if smoke_test:
            command.append("smoke_test=True")

        subprocess.run(command, check=True)
        print(f"--- Finished experiment for {model_name} ---")

    # --- Reporting ---
    print("--- Comparison Report ---")
    report = []
    summary = []

    for model_name in models_to_run.keys():
        log_path = f"results_{model_name}.json"
        with open(log_path, "r") as f:
            data = json.load(f)

        report.append(f"## Results for {model_name}")

        final_metrics = data[-1]

        report.append("| Metric | Value |")
        report.append("|---|---|")
        for key, value in final_metrics.items():
            if key != "step":
                report.append(f"| {key} | {value} |")

        summary.append(f"**{model_name}**: Final loss = {final_metrics.get('test/all/total_loss', 'N/A')}")

    with open("comparison_report.md", "w") as f:
        f.write("# HREM vs HRM Performance Comparison\n\n")
        f.write("## Summary\n\n")
        f.write("\n".join(summary))
        f.write("\n\n")
        f.write("\n\n".join(report))

    print("Report generated: comparison_report.md")

if __name__ == "__main__":
    main()

import os
import subprocess
import yaml
import json
import argparse
from pathlib import Path

def get_hrem_config(hrm_config_path: str, hrem_config_path: str) -> Path:
    """Create HREM config from HRM config."""
    with open(hrm_config_path, "r") as f:
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

    return Path(hrem_config_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dataset", type=str, default="arc", choices=["arc", "sudoku", "maze", "synthetic"])
    parser.add_argument("--num-aug", type=int, default=0)
    parser.add_argument("--synthetic-task", type=str, default="copy", choices=["copy", "reverse"])
    args = parser.parse_args()

    smoke_test = args.smoke_test

    # --- Data Preparation ---
    print("Preparing dataset...")
    if smoke_test:
        args.dataset = "synthetic"
        data_dir = f"data/{args.dataset}-smoke"
        num_aug = 0
        epochs = 1
        eval_interval = 1
        print("Running in smoke test mode. Using a small synthetic dataset.")
    else:
        data_dir = f"data/{args.dataset}-full"
        num_aug = args.num_aug
        epochs = 20000
        eval_interval = 2000
        print("Running in full mode.")

    dataset_builder_script = f"dataset/build_{args.dataset}_dataset.py"
    if not os.path.exists(dataset_builder_script):
        raise FileNotFoundError(f"Dataset builder script not found: {dataset_builder_script}")

    build_command = [
        "python", dataset_builder_script,
        f"--output-dir={data_dir}",
        f"--num-aug={num_aug}"
    ]
    if args.dataset == "synthetic":
        build_command.append(f"--task-type={args.synthetic_task}")
        build_command.append("--num-samples=10") # smaller dataset for smoke test

    subprocess.run(build_command, check=True)

    # --- Training and Evaluation ---
    print("Starting training and evaluation...")

    hrem_config_path = Path("config/arch/hrem_v1_temp.yaml")
    try:
        get_hrem_config("config/arch/hrm_v1.yaml", str(hrem_config_path))

        models_to_run = {
            "HRM": "hrm_v1",
            "HREM": "hrem_v1_temp",
        }

        for model_name, config_name in models_to_run.items():
            print(f"--- Running experiment for {model_name} ---")
            log_path = f"results_{model_name}.json"

            command = [
                "torchrun", "--nproc-per-node", "1", "pretrain.py",
                f"data_path={data_dir}",
                f"epochs={epochs}",
                f"eval_interval={eval_interval}",
                f"arch={config_name}",
                f"+log_path={log_path}",
                "+project_name=HREM_vs_HRM",
                f"+run_name={model_name}_{args.dataset}_smoke_{smoke_test}"
            ]
            if smoke_test:
                command.append("+smoke_test=True")
                command.append("arch.hidden_size=16")
                command.append("arch.H_layers=1")
                command.append("arch.L_layers=1")
                command.append("arch.puzzle_emb_ndim=16")
                command.append("arch.num_heads=1")
                command.append("arch.expansion=1.0")
                command.append("global_batch_size=1")
                command.append("checkpoint_every_eval=True")


            subprocess.run(command, check=True)
            print(f"--- Finished experiment for {model_name} ---")

    finally:
        if hrem_config_path.exists():
            hrem_config_path.unlink()


    # --- Reporting ---
    print("--- Comparison Report ---")
    report = []
    summary = []

    for model_name in models_to_run.keys():
        log_path = f"results_{model_name}.json"
        if not os.path.exists(log_path):
            print(f"Log file not found for {model_name}, skipping report generation.")
            continue

        with open(log_path, "r") as f:
            data = json.load(f)

        if not data:
            print(f"No data found in log file for {model_name}, skipping report generation.")
            continue

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

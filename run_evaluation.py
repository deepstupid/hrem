import os
import subprocess
import json
import argparse
import sys

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
        sys.executable, dataset_builder_script,
        f"--output-dir={data_dir}",
        f"--num-aug={num_aug}"
    ]
    if args.dataset == "synthetic":
        build_command.append(f"--task-type={args.synthetic_task}")
        build_command.append("--num-samples=10")

    subprocess.run(build_command, check=True)

    # --- Training and Evaluation ---
    print("Starting training and evaluation...")

    models_to_run = {
        "HRM": {"config": "hrm_v1"},
        "HREM": {
            "config": "hrm_v1",
            "hparams": {
                "name": "hrm.hrem@HREM",
                "use_memory": True,
                "m_loc": 128,
                "d_mem": 128,
                "top_k": 4,
                "sparse_addressing": True,
                "use_location_addressing": True,
            },
        },
    }

    for model_name, model_info in models_to_run.items():
        print(f"--- Running experiment for {model_name} ---")
        log_path = f"results_{model_name}.json"

        command = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc-per-node", "1",
            "--rdzv-backend", "c10d",
            "--rdzv-endpoint", "localhost:0",
            "pretrain.py",
            f"data_path={data_dir}",
            f"epochs={epochs}",
            f"eval_interval={eval_interval}",
            f"arch={model_info['config']}",
            f"+log_path={log_path}",
            "+project_name=HREM_vs_HRM",
            f"+run_name={model_name}_{args.dataset}_smoke_{smoke_test}"
        ]

        if "hparams" in model_info:
            hparams = model_info["hparams"]

            command.append("arch.name=hrm.hrem@HREM")
            command.append("+arch.use_memory=True")

            new_hrem_params = ["m_loc", "d_mem", "top_k", "sparse_addressing", "use_location_addressing"]
            for key, value in hparams.items():
                if key in ["name", "use_memory"]:
                    continue
                if key in new_hrem_params:
                    command.append(f"+arch.{key}={value}")
                else:
                    command.append(f"arch.{key}={value}")

        if smoke_test:
            smoke_params = {
                "hidden_size": 16,
                "H_layers": 1,
                "L_layers": 1,
                "puzzle_emb_ndim": 16,
                "num_heads": 1,
                "expansion": 1.0,
            }
            for key, value in smoke_params.items():
                command.append(f"arch.{key}={value}")

            command.extend([
                "+smoke_test=True",
                "global_batch_size=1",
                "checkpoint_every_eval=True"
            ])

        subprocess.run(command, check=True)
        print(f"--- Finished experiment for {model_name} ---")


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

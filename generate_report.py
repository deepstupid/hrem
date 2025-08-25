import os
import sys
import json
import yaml
from collections import defaultdict

def parse_logs(log_dir):
    results = defaultdict(lambda: defaultdict(dict))
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.endswith("_metrics.jsonl"):
                model_run_name = file.removesuffix("_metrics.jsonl")
                parts = model_run_name.split("_")
                model_name = parts[0]
                run_type = parts[1]

                metrics_data = []
                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        metrics_data.append(json.loads(line))
                
                # Load config for this run
                config_file = os.path.join(root, f"{model_run_name}_config.yaml")
                config = {}
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)

                results[model_name][run_type]["metrics"] = metrics_data
                results[model_name][run_type]["config"] = config
                results[model_name][run_type]["path"] = root
    return results

def generate_report(results, output_file):
    with open(output_file, 'w') as f:
        f.write("### HREM vs HRM Evaluation Report ###\n\n")

        for model_name, run_types in results.items():
            f.write(f"## Model: {model_name}\n")
            for run_type, data in run_types.items():
                f.write(f"### Run Type: {run_type.capitalize()}\n")
                f.write(f"Output Path: {data["path"]}\n")
                
                config = data["config"]
                if config:
                    f.write("Configuration:\n")
                    for k, v in config.items():
                        if k not in ["arch", "data_path", "checkpoint_path", "run_name", "project_name"]:
                            f.write(f"  {k}: {v}\n")
                    f.write(f"  arch: {config.get("arch", {}).get("name", "N/A")}\n")
                    f.write(f"  loss: {config.get("arch", {}).get("loss", {}).get("name", "N/A")}\n")

                metrics = data["metrics"]
                if metrics:
                    # Extract latest training metrics and evaluation metrics
                    latest_train_metrics = {}
                    eval_metrics = {}
                    for entry in metrics:
                        if "train/loss" in entry["metrics"]:
                            latest_train_metrics = entry["metrics"]
                        if "eval_results" in entry["metrics"]:
                            eval_metrics = entry["metrics"]["eval_results"]
                    
                    f.write("\nLatest Training Metrics:\n")
                    if latest_train_metrics:
                        for k, v in latest_train_metrics.items():
                            if k.startswith("train/"):
                                f.write(f"  {k}: {v:.4f}\n")
                    else:
                        f.write("  No training metrics found.\n")

                    f.write("\nEvaluation Metrics:\n")
                    if eval_metrics:
                        for set_name, set_metrics in eval_metrics.items():
                            f.write(f"  Set: {set_name}\n")
                            for k, v in set_metrics.items():
                                f.write(f"    {k}: {v:.4f}\n")
                    else:
                        f.write("  No evaluation metrics found.\n")
                else:
                    f.write("No metrics logged for this run.\n")
                f.write("\n")

        f.write("\n### Summary of Performance Difference ###\n")
        # Add logic here to compare HRM and HREM, e.g., average eval loss
        hrm_eval_losses = []
        hrem_eval_losses = []

        for model_name in results:
            for run_type in results[model_name]:
                metrics = results[model_name][run_type]["metrics"]
                if metrics:
                    for entry in metrics:
                        if "eval_results" in entry["metrics"]:
                            for set_name, set_metrics in entry["metrics"]["eval_results"].items():
                                if "loss" in set_metrics:
                                    if model_name == "HRM":
                                        hrm_eval_losses.append(set_metrics["loss"])
                                    elif model_name == "HREM":
                                        hrem_eval_losses.append(set_metrics["loss"])
        
        if hrm_eval_losses and hrem_eval_losses:
            avg_hrm_loss = sum(hrm_eval_losses) / len(hrm_eval_losses)
            avg_hrem_loss = sum(hrem_eval_losses) / len(hrem_eval_losses)
            f.write(f"Average HRM Evaluation Loss: {avg_hrm_loss:.4f}\n")
            f.write(f"Average HREM Evaluation Loss: {avg_hrem_loss:.4f}\n")
            if avg_hrem_loss < avg_hrm_loss:
                f.write("HREM shows a potential improvement in average evaluation loss.\n")
            elif avg_hrem_loss > avg_hrm_loss:
                f.write("HRM shows better average evaluation loss.\n")
            else:
                f.write("HRM and HREM have similar average evaluation loss.\n")
        else:
            f.write("Insufficient data to compare average evaluation losses.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <path_to_output_directory>")
        sys.exit(1)

    output_directory = sys.argv[1]
    if not os.path.isdir(output_directory):
        print(f"Error: Directory {output_directory} not found.")
        sys.exit(1)

    results = parse_logs(output_directory)
    report_path = os.path.join(output_directory, "report.txt")
    generate_report(results, report_path)
    print(f"Report generated at {report_path}")

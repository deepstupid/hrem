import argparse
import json
from hrm_system.config import RunConfig, DataConfig, ModelConfig, TrainingConfig
from hrm_system.runner import run_single_model
from challenges import get_challenge_by_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--global_batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--eval_interval", type=int, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--smoke_test", action="store_true")
    # For simplicity, we'll handle arch overrides as a JSON string
    parser.add_argument("--arch_overrides", type=str, default="{}")
    # Allow specifying a challenge
    parser.add_argument("--challenge", type=str, default="copy_task_beginner")

    args = parser.parse_args()

    run_config = RunConfig(
        smoke_test=args.smoke_test,
        study_name="smoke_test",
        output_dir=args.checkpoint_path,
        log_path=args.log_path,
    )

    # Get data config from challenge system
    challenge = get_challenge_by_name(args.challenge)
    if challenge:
        data_config = challenge.data_config
        # Override the path if specified
        if args.data_path:
            data_config.path = args.data_path
    else:
        # Fallback to default synthetic dataset
        data_config = DataConfig(
            dataset="synthetic", # Smoke test uses synthetic dataset
            path=args.data_path,
        )

    arch_overrides = json.loads(args.arch_overrides)

    model_config = ModelConfig(
        name="HRM_smoke",
        algorithm_class="hrm_system.algorithms.hrm.HRMAlgorithm",
        base_arch_config=args.arch,
        arch_overrides=arch_overrides,
    )

    training_config = TrainingConfig(
        epochs=args.epochs,
        eval_interval=args.eval_interval,
        global_batch_size=args.global_batch_size,
        smoke_test=args.smoke_test,
    )

    run_single_model(
        run_config=run_config,
        data_config=data_config,
        model_config=model_config,
        training_config=training_config,
        run_identifier="smoke_test_run"
    )

if __name__ == "__main__":
    main()

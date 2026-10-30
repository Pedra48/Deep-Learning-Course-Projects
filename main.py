import argparse
import json
import random
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace


def load_config(config_path: Path) -> SimpleNamespace:
    """Read YAML parameters and expose them as args.lr, args.batch_size, etc."""
    import yaml

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or not config:
        raise ValueError("The YAML configuration must be a non-empty mapping.")

    return SimpleNamespace(**config)


def main():
    parser = argparse.ArgumentParser(description="Train a network to approximate sin(x).")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "exp1.yaml",
        help="Path to the YAML configuration file.",
    )
    cli_args = parser.parse_args()
    args = load_config(cli_args.config)

    import numpy as np
    import torch
    import yaml
    from src.exp import Exp

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    args.output_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(vars(args), stream, sort_keys=False)

    started = perf_counter()
    exp = Exp(args)
    exp.train()
    final_train_mse = exp._evaluate(exp.train_loader)
    final_val_mse = exp.eval()
    test_mse = exp.test()  # Only evaluate the test set once in this run.
    if exp.device.type == "cuda":
        torch.cuda.synchronize(exp.device)
    metrics = {
        "device": str(exp.device),
        "model_device": str(next(exp.model.parameters()).device),
        "gpu": torch.cuda.get_device_name(exp.device) if exp.device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "seed": args.seed,
        "epochs": args.epoch,
        "final_train_mse": final_train_mse,
        "final_val_mse": final_val_mse,
        "test_mse": test_mse,
        "test_evaluations": 1,
        "elapsed_seconds": perf_counter() - started,
        "peak_cuda_memory_mb": (
            torch.cuda.max_memory_allocated(exp.device) / 1024 ** 2
            if exp.device.type == "cuda" else 0.0
        ),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)
    print(f"Final train MSE: {final_train_mse:.10f}", flush=True)
    print(f"Final validation MSE: {final_val_mse:.10f}", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)
    

if __name__ == "__main__":
    main()

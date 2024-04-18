from pathlib import Path
import json
import os
import wandb

from eval_checker.eval_runner_helper import load_file


def process_data(score_path: Path, save_path: Path):
    for file in score_path.glob("**/*.json"):
        results = {}
        model_name = file.parent.name
        model_name = str(model_name).replace("-FC", "")
        model_name = model_name.replace("-Auto", "")

        predictions = load_file(file)

        for i, sample in enumerate(predictions):
            if i == 0:
                continue
            
            training_prompt = sample["training_prompt"]
            score = sample["score"]
            results["training_prompt"] = training_prompt
            results["score"] = score
            results[f"{i}"] = {
                "eval_details": {"prompt": training_prompt},
                "result": {"score": score}
            }
        if not (save_path / model_name).exists():
            os.makedirs(save_path / model_name)
        with open(save_path / model_name / file.name, "w") as f:
            json.dump(results, f)


def upload_to_wandb(data_path: Path):
    dataset = wandb.Artifact(name="gorilla-dataset", type="dataset")
    dataset.add_dir(data_path.resolve())
    with wandb.init(project="LLM Eval") as run:
        run.log_artifact(dataset)
    return True


if __name__ == "__main__":
    score_path = Path("./score") / "20240417_183023"
    save_path = Path("./training_data") / "20240417_183023"
    
    process_data(score_path, save_path)
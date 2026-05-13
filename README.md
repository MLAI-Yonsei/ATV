# ATV
Code repositories for __ATV (Adaptive Task Vectors).__

## Requirements

To run this code, create and activate a conda environment using the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml

conda activate ATV
```


## Run code
### Prepare datasets
   ```bash
   ./scripts/prepare_dataset.sh
   ```

### Train ATV model
   ```bash
   ./scripts/ATV_training.sh
   ```
Running the above script trains the model on all 20 in-domain datasets. After training, evaluation is performed on the test samples from all in-domain datasets.

#### Batch options (Training)
- `--batch_size N`: Groups N samples for GPT-2 forward.
- `--llama_batch`: Additionally batches LLaMA forward (N samples × 3 templates at once). FP16 numerical differences cause slightly different training trajectories.
- `--logits_to_keep`: Computes only final-token logits on compatible last-token inference paths. Adaptive training loss keeps full logits to preserve the original teacher-forcing objective.
- `--gradient_checkpointing`: Enables LLaMA activation checkpointing during training. This substantially reduces memory by recomputing LLaMA activations during backward, with extra compute cost.

```bash
# GPT-2 batch only
python ATV_training.py ... --batch_size 2

# GPT-2 + LLaMA batch
python ATV_training.py ... --batch_size 2 --llama_batch

# Memory-focused mode
python ATV_training.py ... --batch_size 2 --llama_batch --gradient_checkpointing
```

### Evaluate all datasets
   ```bash
   ./scripts/ATV_evaluate.sh
   ```
Running the above script enables evaluation of performance on each individual dataset within the full collection.

#### Batch options (Evaluation)
- `--batch_size N`: Batches N samples for GPT-2 and LLaMA forward simultaneously.
- `--logits_to_keep`: Computes only the final-token logits for last-token evaluation. This saves memory but can introduce tiny numerical differences.

```bash
python ATV_evaluate.py ... --batch_size 4
```

### Analyze results
   ```bash
   python ATV_analysis.py
   ```
This script enables evaluation of performance for each category.
Please make sure to modify the `result_dirs` variable in `ATV_analysis.py` to match the path to your result directory.

#### Analyze unseen task
   ```bash
   python ATV_analysis.py
   ```
For unseen data, run `ATV_unseen.py` to perform the evaluation.
As above, make sure to set the correct paths accordingly.

<br/>

## Acknowledge
This repository is built on top of the [ELICIT](https://github.com/LINs-lab/ELICIT) project. We thank the authors for sharing the source and their work itself.

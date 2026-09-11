# ATV
Code repositories for __ATV (Adaptive Task Vectors).__

This branch implements Llama-3-8B ATV with the paper's GPT-2 Last-Mean readout.
The last-token and masked-mean states are projected to 256 dimensions, combined
with mean weight 0.7, and mapped to the target layers through a shared projection.
No additional normalization or category subtraction is applied.

The injection follows the paper: it modifies only the last prompt token at each
target layer, which predicts the first answer token. Training and validation use
`prompt_len - 1` in the prompt-plus-answer sequence. Inference uses the final
prompt position, and subsequent generation proceeds without the hook.
This replaces the earlier sequence-wide broadcasting behavior. Existing
checkpoint weights can be loaded, but their scores under this injection rule
require reevaluation.

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
Running the above script trains the model on all 20 in-domain datasets using
seeds 42, 100, and 10. It uses GPT-2 LoRA with rank 16, alpha 32, and dropout 0.05;
AdamW with learning rates 8e-4 for LoRA and 1e-3 for the projections; and an
effective batch size of 16 for 15 epochs. The objective combines answer-token CE
with i-Mix on the last-token states using clean keys, temperature 0.1, mixup alpha
0.5, and loss weight 1.0. The intervention weight is 0.001.

`best_model_epoch.pt` is selected by validation token accuracy, with validation
loss breaking ties. Evaluation is run separately with the script below.

Both scripts default to GPU 0. Set `GPUS` and `SEEDS` to select GPUs and runs:

```bash
GPUS="0 1 2" SEEDS="42 100 10" ./scripts/ATV_training.sh
```

`ATV_LLAMA_MODEL` and `ATV_GPT2_MODEL` can point to local model directories.
The defaults are `meta-llama/Meta-Llama-3-8B` and `gpt2`.

### Evaluate all datasets
   ```bash
   ./scripts/ATV_evaluate.sh
   ```
Running the above script evaluates the best validation checkpoint on all 25
datasets. The readout configuration is loaded from the checkpoint, including
mean weight 0.7. Existing full GPT-2 checkpoints remain loadable.

The dataset preparation and evaluation splits retain the existing repository
behavior. Use the finalized evaluation data when comparing with the revised
paper, which uses corrected GLUE evaluation sets.

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

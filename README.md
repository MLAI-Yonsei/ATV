# ATV
Code repositories for __ATV (Adaptive Task Vectors).__

## Results

`Paper ATV` denotes the original Llama-3 ATV results reported in the paper, without i-mix or category subtraction.
`i-mix + Category subtraction` uses the integrated i-mix checkpoint and JSON-selected mean category subtraction settings.

### In-domain

| Row | NLU | Reasoning | Knowledge | Math | Safety | Avg |
|---|---:|---:|---:|---:|---:|---:|
| Paper ATV | 61.0 ± 5.0 | 76.1 ± 1.3 | 73.0 ± 1.6 | 25.8 ± 2.0 | 74.8 ± 0.4 | 62.1 ± 1.5 |
| i-mix + Category subtraction | 63.6 ± 5.8 | 78.6 ± 2.7 | 77.3 ± 2.0 | 28.4 ± 1.5 | 76.7 ± 1.5 | 64.9 ± 0.9 |

### OOD

| Row | GLUE CoLA | BBQ Religion | DeepMind | MMLU Psych | BBH Five Objects | Avg |
|---|---:|---:|---:|---:|---:|---:|
| Paper ATV | 77.6 ± 2.7 | 80.8 ± 2.6 | 26.4 ± 2.7 | 80.6 ± 2.3 | 51.7 ± 3.1 | 63.4 ± 2.5 |
| i-mix + Category subtraction | 82.7 ± 9.7 | 84.0 ± 0.7 | 31.0 ± 2.5 | 83.1 ± 2.3 | 52.0 ± 2.8 | 66.6 ± 2.7 |

## Updates from `dcfeb30`

This version keeps the `dcfeb30b4b7bd800034ed982e208b61a844d8f17` code path and adds the experiment-backed i-mix training objective plus JSON-selected mean category subtraction.
The category subtraction JSON selects the source datasets, coefficient, and layer mask for each dataset.

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

### Evaluate all datasets
   ```bash
   ./scripts/ATV_evaluate.sh
   ```
Running the above script enables evaluation of performance on each individual dataset within the full collection.

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

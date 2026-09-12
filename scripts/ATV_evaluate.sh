#!/bin/bash
# Datasets  
datasets=( "arc_challenge" "bbh_boolean_expressions" "bbh_date_understanding" "bbh_reasoning_about_colored_objects" "bbh_temporal_sequences" "bbq_age" "boolq" "crows_pairs" "commonsense_qa" "ethics_commonsense" "ethics_justice" "glue_sst2" "hellaswag" "math_qa" "mmlu_pro_math" "openbookqa" "superglue_rte" "superglue_wic" "glue_mnli" "glue_qnli" "glue_cola" "bbq_religion" "deepmind" "mmlu_high_school_psychology" "bbh_logical_deduction_five_objects" )

# GPUs to use (each GPU runs only one task)
read -r -a gpus <<< "${GPUS:-0}"
read -r -a seeds <<< "${SEEDS:-42 100 10}"
status=0
# Array to store the PID of currently running processes for each GPU
declare -A gpu_jobs

# Function to check if a GPU slot is available
wait_for_gpu() {
    while true; do
        for gpu in "${gpus[@]}"; do
            # If there is no task on the GPU, return immediately
            if [[ -z "${gpu_jobs[$gpu]}" ]]; then
                return 0
            else
                # If the process has ended, clear the slot and return
                if ! kill -0 "${gpu_jobs[$gpu]}" 2>/dev/null; then
                    wait "${gpu_jobs[$gpu]}" || status=1
                    gpu_jobs[$gpu]=""
                    return 0
                fi
            fi
        done
        sleep 1
    done
}

mkdir -p logs/evaluate_ATV

weight_decay=1e-5
# Task scheduling: run for each seed, weight_fv, and dataset combination
for epochs in 15; do
# for epochs in 10 15; do
    for lr in 1e-3; do
        for seed in "${seeds[@]}"; do
            for weight_fv in 0.001; do
                for d_name in "${datasets[@]}"; do
                    # Wait for an available GPU slot
                    wait_for_gpu
                    # Find an available GPU id
                    for gpu in "${gpus[@]}"; do
                        if [[ -z "${gpu_jobs[$gpu]}" ]]; then
                            free_gpu=$gpu
                            break
                        fi
                    done

                    # Log file name (create a separate log file for each task)
                    log_file="logs/evaluate_ATV/log_${weight_fv}_${seed}_${d_name}_${weight_decay}_${epochs}.log"

                    echo "Running ${d_name}, weight_fv: ${weight_fv}, seed: ${seed}, weight_decay: ${weight_decay}"
                    
                    # Actual command configuration (original cmd variable content)
                    cmd=(python ATV_evaluate.py \
                    --dataset_name ${d_name} \
                    --model_name "${ATV_LLAMA_MODEL:-meta-llama/Meta-Llama-3-8B}" \
                    --gpt2_model_name "${ATV_GPT2_MODEL:-gpt2}" \
                    --test_samples 100 \
                    --save_dir eval_results/evaluate_ATV/lastmean_0p7_${weight_fv}_${seed}_${weight_decay}_${epochs} \
                    --weight_ori 1.0 \
                    --weight_fv ${weight_fv} \
                    --dataset_split test \
                    --local \
                    --prompt_file dataset_files/natural_prompts_manual.json \
                    --seed ${seed} \
                    --soft \
                    --recollect_state \
                    --use_template \
                    --trained_model_path "eval_results/training_ATV/lastmean_0p7_${weight_fv}_${seed}_${weight_decay}/best_model_epoch.pt")
                    
                    # Assign the allocated GPU number to an environment variable and run in the background,
                    # Standard output and standard error are saved to the log file (no console output)
                    CUDA_VISIBLE_DEVICES=${free_gpu} "${cmd[@]}" > "$log_file" 2>&1 &
                    
                    # Save the background process PID
                    gpu_jobs[$free_gpu]=$!
                done
            done
        done
    done
done

# Wait for all background tasks to complete
for pid in "${gpu_jobs[@]}"; do
    [[ -z "$pid" ]] || wait "$pid" || status=1
done

# Final completion log (if desired, redirect to a file or remove)
echo "Evaluation workers finished with status ${status}"
exit "$status"

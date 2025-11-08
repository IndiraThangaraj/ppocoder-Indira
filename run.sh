    #!/bin/bash

    # --- Configuration ---
    DATA_DIR="data/mbpp/" 
    OUTPUT_DIR="saved_models/mbpp_ppo/"
    BASELINE_MODEL_DIR="baselines/saved_models/mbpp/"

    # --- Run PPOCoder RL Fine-Tuning ---
    echo "Starting PPOCoder RL fine-tuning for MBPP..."

    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:$(pwd)/code_prepro/lang_processors"

    python /Users/00143773indiraapthangaraj/PPOCoder/PPOCoder/rl_run.py \
        --run 1 \
        --l1 python \
        --l2 python \
        --asp 5 \
        --ns 10 \
        --data_path "$DATA_DIR" \
        --output_path "$OUTPUT_DIR" \
        --load_model_path "${BASELINE_MODEL_DIR}pytorch_model.bin" \
        --baseline_output_path "$BASELINE_MODEL_DIR" \
        --train_filename "${DATA_DIR}train.src,${DATA_DIR}train.tgt" \
        --dev_filename "${DATA_DIR}val.src,${DATA_DIR}val.tgt" \
        --test_filename "${DATA_DIR}test.src,${DATA_DIR}test.tgt" \
        --max_source_length 400 \
        --max_target_length 400 \
        --train_batch_size 32 \
        --test_batch_size 48 \
        --lr 1e-6 \
        --kl_coef 0.1 \
        --kl_target 1 \
        --vf_coef 1e-3
    ```

2.  **Execute the script:**

    ```bash
    bash run.sh
    

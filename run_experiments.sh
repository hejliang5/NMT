#!/bin/bash

set -euo pipefail

# Set common variables
DATASET="datasets/train_100k.jsonl"
VALID="datasets/valid.jsonl"
EPOCHS=20
DEVICE="cuda"

# H100-friendly knobs (tune if you hit OOM)
MAX_SRC_LEN=80
MAX_TGT_LEN=80

# Dataloader/tokenization
NUM_WORKERS=8

# Batch sizes
RNN_BS=256
TRANS_BS=256
T5_BS=64

# Logging
LOG_DIR="logs/h100_e${EPOCHS}_len${MAX_SRC_LEN}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Create directories
mkdir -p checkpoints/rnn_ablation
mkdir -p checkpoints/transformer_ablation
mkdir -p checkpoints/t5_finetune

echo "Starting Experiments..."
echo "Logs: $LOG_DIR"

run_logged () {
    local name="$1"; shift
    echo "=== RUN: $name ==="
    "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
}

# ==========================================
# 1. RNN Experiments
# ==========================================

echo "Running RNN Alignment Ablation..."
for align in dot general additive; do
    run_logged "rnn_align_${align}" python3 train_rnn.py \
        --train "$DATASET" \
        --valid "$VALID" \
        --save_dir "checkpoints/rnn_ablation/align_${align}" \
        --epochs "$EPOCHS" \
        --batch_size "$RNN_BS" \
        --alignment "$align" \
        --max_src_len "$MAX_SRC_LEN" \
        --max_tgt_len "$MAX_TGT_LEN" \
        --drop_long \
        --clean \
        --num_workers "$NUM_WORKERS" \
        --eval_decode beam \
        --beam_size 4 \
        --amp \
        --device "$DEVICE"
done

echo "Running RNN Teacher Forcing Ablation..."
for tf in 1.0 0.5 0.0; do
    run_logged "rnn_tf_${tf}" python3 train_rnn.py \
        --train "$DATASET" \
        --valid "$VALID" \
        --save_dir "checkpoints/rnn_ablation/tf_${tf}" \
        --epochs "$EPOCHS" \
        --batch_size "$RNN_BS" \
        --alignment dot \
        --teacher_forcing "$tf" \
        --max_src_len "$MAX_SRC_LEN" \
        --max_tgt_len "$MAX_TGT_LEN" \
        --drop_long \
        --clean \
        --num_workers "$NUM_WORKERS" \
        --eval_decode beam \
        --beam_size 4 \
        --amp \
        --device "$DEVICE"
done

# # 1.3 Decoding Strategy (Evaluate Baseline)
# echo "Evaluating RNN Decoding Strategies..."
# # Assuming baseline (dot, tf=0.5) is in checkpoints/seq2seq_100k/best_model.pt (from previous steps)
# # If not, use one of the above. Let's use align_dot.
# BASELINE_RNN="checkpoints/rnn_ablation/align_dot/best_model.pt"
# if [ -f "$BASELINE_RNN" ]; then
#     echo "Evaluating Greedy..."
#     python3 train_rnn.py --train $DATASET --valid $VALID --save_dir checkpoints/rnn_ablation/align_dot --epochs 0 --eval_decode greedy --device $DEVICE
#     echo "Evaluating Beam..."
#     python3 train_rnn.py --train $DATASET --valid $VALID --save_dir checkpoints/rnn_ablation/align_dot --epochs 0 --eval_decode beam --beam_size 4 --device $DEVICE
# else
#     echo "Baseline RNN checkpoint not found, skipping decoding eval."
# fi

echo "Running Transformer Ablations..."

run_logged "trans_base" python3 train_transformer.py \
    --train "$DATASET" \
    --valid "$VALID" \
    --save_dir checkpoints/transformer_ablation/base \
    --epochs "$EPOCHS" \
    --batch_size "$TRANS_BS" \
    --pos_encoding sinusoidal \
    --norm_type layernorm \
    --max_src_len "$MAX_SRC_LEN" \
    --max_tgt_len "$MAX_TGT_LEN" \
    --drop_long \
    --clean \
    --num_workers "$NUM_WORKERS" \
    --eval_decode beam \
    --beam_size 4 \
    --amp \
    --device "$DEVICE"

run_logged "trans_learned_pe" python3 train_transformer.py \
    --train "$DATASET" \
    --valid "$VALID" \
    --save_dir checkpoints/transformer_ablation/learned_pe \
    --epochs "$EPOCHS" \
    --batch_size "$TRANS_BS" \
    --pos_encoding learned \
    --norm_type layernorm \
    --max_src_len "$MAX_SRC_LEN" \
    --max_tgt_len "$MAX_TGT_LEN" \
    --drop_long \
    --clean \
    --num_workers "$NUM_WORKERS" \
    --eval_decode beam \
    --beam_size 4 \
    --amp \
    --device "$DEVICE"

run_logged "trans_rmsnorm" python3 train_transformer.py \
    --train "$DATASET" \
    --valid "$VALID" \
    --save_dir checkpoints/transformer_ablation/rmsnorm \
    --epochs "$EPOCHS" \
    --batch_size "$TRANS_BS" \
    --pos_encoding sinusoidal \
    --norm_type rmsnorm \
    --max_src_len "$MAX_SRC_LEN" \
    --max_tgt_len "$MAX_TGT_LEN" \
    --drop_long \
    --clean \
    --num_workers "$NUM_WORKERS" \
    --eval_decode beam \
    --beam_size 4 \
    --amp \
    --device "$DEVICE"

echo "Running Transformer Hyperparam Sensitivity..."

# Compare smaller batch (should be slower, but helps sensitivity study)
run_logged "trans_bs128" python3 train_transformer.py \
    --train "$DATASET" \
    --valid "$VALID" \
    --save_dir checkpoints/transformer_ablation/bs128 \
    --epochs "$EPOCHS" \
    --batch_size 128 \
    --max_src_len "$MAX_SRC_LEN" \
    --max_tgt_len "$MAX_TGT_LEN" \
    --drop_long \
    --clean \
    --num_workers "$NUM_WORKERS" \
    --eval_decode beam \
    --beam_size 4 \
    --amp \
    --device "$DEVICE"

# Lower learning rate
run_logged "trans_lr1e4" python3 train_transformer.py \
    --train "$DATASET" \
    --valid "$VALID" \
    --save_dir checkpoints/transformer_ablation/lr1e4 \
    --epochs "$EPOCHS" \
    --batch_size "$TRANS_BS" \
    --lr 1e-4 \
    --max_src_len "$MAX_SRC_LEN" \
    --max_tgt_len "$MAX_TGT_LEN" \
    --drop_long \
    --clean \
    --num_workers "$NUM_WORKERS" \
    --eval_decode beam \
    --beam_size 4 \
    --amp \
    --device "$DEVICE"

# ==========================================
# 3. T5 Fine-tuning
# ==========================================
echo "Fine-tuning T5..."
run_logged "t5_finetune" python3 train_t5.py \
    --train $DATASET \
    --valid $VALID \
    --model_path t5-base \
    --save_dir checkpoints/t5_finetune \
    --epochs $EPOCHS \
    --batch_size $T5_BS \
    --max_src_len 96 \
    --max_tgt_len 96 \
    --max_gen_len 96 \
    --num_beams 4 \
    --amp \
    --device $DEVICE

echo "Plotting training curves from logs..."
python3 plot_training_curves.py --log_dir "$LOG_DIR" --out_dir "${LOG_DIR}/plots" --csv "${LOG_DIR}/training_curves.csv"

echo "All experiments completed."

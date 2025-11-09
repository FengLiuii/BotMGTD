#!/bin/bash

echo "========================================"
echo "Attention-based BotMGTD Model Runner"
echo "========================================"
echo

echo "Step 1: Quick Test"
echo "-------------------"
python quick_test.py
if [ $? -ne 0 ]; then
    echo "Quick test failed! Please check the error messages."
    exit 1
fi

echo
echo "Step 2: Full Training"
echo "---------------------"
echo "Starting full training with attention aggregation..."
echo
#   --threshold 0.7
python run_attention_model.py \
    --data MGTAB \
    --epoch 200 \
    --lr 0.001 \
    --difflr 0.01 \
    --patience 10 \
    --batch 512 \
    --threshold 0.5 \
    --latdim 128 \
    --gcn_layer 1 \
    --uugt_layer 1 \
    --head 4 \
    --dropRate 0.3 \
    --decay 0.001 \
    --dims "[128]" \
    --d_emb_size 8 \
    --norm True \
    --steps 5 \
    --noise_scale 5e-5 \
    --noise_min 0.0001 \
    --noise_max 0.001 \
    --sampling_steps 0 \


echo
echo "========================================"
echo "Training completed!"
echo "Check the History/ folder for results"
echo "Check the current directory for visualizations"
echo "========================================"

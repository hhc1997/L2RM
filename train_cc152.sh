python -u main_L2RM.py \
  --comment _SGR_CC152K \
  --module_name SGR \
  --gpu 0 \
  --batch_size 128 \
  --seed 0 \
  --reg 0.07 \
  --num_epochs 40 \
  --lr 2e-4 \
  --lr_cost 2e-6 \
  --workers 0 \
  --lr_update 20 \
  --data_name cc152k_precomp \
  --noise_ratio 0.0 \
  --queue_length 128 \
  --warmup_epoch 10 \
  --data_path ./data \
  --vocab_path ./vocab


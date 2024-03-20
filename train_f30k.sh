python -u main_L2RM.py \
  --comment _MRate_0.2_SGR_f30k \
  --module_name SGR \
  --gpu 0 \
  --batch_size 128 \
  --seed 0 \
  --reg 0.01 \
  --num_epochs 35 \
  --lr 2e-4 \
  --lr_cost 2e-6 \
  --workers 0 \
  --lr_update 15 \
  --data_name f30k_precomp \
  --noise_ratio 0.2 \
  --queue_length 128 \
  --warmup_epoch 5 \
  --data_path ./data \
  --vocab_path ./vocab \
  --noise_file ./noise_index/f30k_precomp_0.2.npy



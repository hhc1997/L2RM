python -u main_L2RM.py \
  --comment _MRate_0.2_SGR_coco \
  --module_name SGR \
  --gpu 0 \
  --batch_size 128 \
  --seed 0 \
  --reg 0.07 \
  --num_epochs 20 \
  --lr 2e-4 \
  --lr_cost 2e-6 \
  --workers 0 \
  --lr_update 10 \
  --data_name coco_precomp \
  --noise_ratio 0.2 \
  --queue_length 128 \
  --warmup_epoch 10 \
  --data_path ./data \
  --vocab_path ./vocab \
  --noise_file ./noise_index/coco_precomp_0.2.npy


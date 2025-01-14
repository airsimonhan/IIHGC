#!/usr/bin/env bash

model_args='{"in_channels": 512, "n_target": 1, "k_threshold": 0.1, "k_nearest": 5, "hiddens": [128, 128, 128], "dropout": 0.3, "drop_max_ratio": 0.05}'
python train_intra_stage.py --config config/tcga/intra_tcga-lusc.yaml \
                        --work_dir work_dir/intra_stage/tcga_lusc/ \
                        --model_args "$model_args" \
                        --data_seed 1 \
                        --device 0

# model_args='{"in_channels": 512, "n_target": 1, "k_threshold": 0.1, "k_nearest": 5, "hiddens": [128, 128, 128], "dropout": 0.3, "drop_max_ratio": 0.05}'
# python train_intra_stage.py --config config/tcga/intra_tcga-luad.yaml \
#                         --work_dir work_dir/intra_stage/tcga_luad/ \
#                         --model_args "$model_args" \
#                         --data_seed 1 \
#                         --device 0

# model_args='{"in_channels": 512, "n_target": 1, "k_threshold": 0.1, "k_nearest": 5, "hiddens": [128, 128, 128], "dropout": 0.3, "drop_max_ratio": 0.05}'
# python train_intra_stage.py --config config/tcga/intra_tcga-kirc.yaml \
#                         --work_dir work_dir/intra_stage/tcga_kirc/ \
#                         --model_args "$model_args" \
#                         --data_seed 1 \
#                         --device 0
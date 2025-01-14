#!/usr/bin/env bash

inter_model_args='{"in_channels": 262, "n_target": 1, "k_threshold": 0.01, "hiddens": [128, 128, 128], "dropout": 0.3, "scale": 0.5,"label_hiddens": [16,16,16],"label_in": 1}'
python train_inter-intra.py --config config/tcga/inter-intra_tcga-lusc.yaml \
                        --work_dir work_dir/inter_intra/tcga_lusc/ \
                        --inter_model_args "$inter_model_args" \
                        --intra_weights work_dir/intra_stage/tcga_lusc/ \
                        --device 0 \
                        --WSI_patch_ft_dir /home2/zhouhuijian/codes/IIHGC/get_feature/WSI_patch_features/{}_2K_sample_threshold/patch_efficientnet_ft \
                        --WSI_patch_coor_dir /home2/zhouhuijian/codes/IIHGC/get_feature/WSI_patch_features/{}_2K_sample_threshold/patch_coor \
                        --num_epoch 10

# inter_model_args='{"in_channels": 262, "n_target": 1, "k_threshold": 0.01, "hiddens": [128, 128, 128], "dropout": 0.3, "scale": 0.5,"label_hiddens": [16,16,16],"label_in": 1}'
# python train_inter-intra.py --config config/tcga/inter-intra_tcga-luad.yaml \
#                         --work_dir work_dir/inter_intra/tcga_luad/ \
#                         --inter_model_args "$inter_model_args" \
#                         --intra_weights work_dir/intra_stage/tcga_luad/ \
#                         --device 0 

# inter_model_args='{"in_channels": 262, "n_target": 1, "k_threshold": 0.01, "hiddens": [128, 128, 128], "dropout": 0.3, "scale": 0.5,"label_hiddens": [16,16,16],"label_in": 1}'
# python train_inter-intra.py --config config/tcga/inter-intra_tcga-kirc.yaml \
#                         --work_dir work_dir/inter_intra/tcga_kirc/ \
#                         --inter_model_args "$inter_model_args" \
#                         --intra_weights work_dir/intra_stage/tcga_kirc/ \
#                         --device 0
'''
python wy_train_ssd_multiGPU_with_config_incremental_dataset_specifically_tune.py \
--dataset_type ecp \
--net mb2-ssd-lite \
--base_net models/mb2-imagenet-71_8.pth \
--batch_size 41 \
--num_epochs 100 \
--num_workers 13 \
--validation_epochs 5 \
--debug_steps 100 \
--config config/prague_combine_balance.yaml \
--sample_method uncertainty \
--checkpoint_folder ../experioments/uncertainty_prag_combine
'''
python wy_train_ssd_multiGPU_with_config_incremental_dataset_specifically_tune.py \
--dataset_type ecp \
--net mb2-ssd-lite \
--base_net models/mb2-imagenet-71_8.pth \
--batch_size 41 \
--num_epochs 100 \
--num_workers 13 \
--validation_epochs 5 \
--debug_steps 100 \
--config config/prague_combine_balance.yaml \
--sample_method uncertainty_modify \
--checkpoint_folder ../experioments/uncertainty_modify_prag_combine

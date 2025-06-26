1. 将数据集存入data/datasets/my_cool_dataset
2. 在configs/dataset_control_freq.json中加入my_cool_dataset的控制频率
3. 修改configs/finetune_datasets.json和finetune_sample_weights.json，加入my_cool_dataset数据集
4. 修改data/hdf5_vla_dataset.py中HDF5VLADataset类，修改HDF5_DIR = "data/datasets/my_cool_dataset"和self.DATASET_NAME="my_cool_dataset"，如果不是hdf5数据集需自行实现parse_hdf5_file()和parse_hdf5_file_state_only()
5. 执行python -m data.compute_dataset_stat_hdf5验证数据集是否成功导入以及计算min、max、mean等值
6. 修改finetune.sh中的相关训练参数，可能需要修改的参数包括train_batch_size,max_train_steps,checkpointing_period,checkpoints_total_limit,lr_scheduler,learning_rate
7. 执行source finetune.sh开始微调

其中2，3，4只需执行一次即可，若修改了数据集务必先执行一次5再进行微调

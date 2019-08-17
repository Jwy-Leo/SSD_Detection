'''
index='0.1'
model_path="../experiments/uncertainty_models_ecp_mb2-ssd-lite_2019-07-30-01-07-54/query_iter_0.1/mb2-ssd-lite-Epoch-95-Loss-6.463103634970529.pth"
python wy_eval_ssd_ecp.py --net mb2-ssd-lite --dataset_type ecp --label_file models/ecp_labels.txt --config config/prague_combine_debug.yaml --trained_model $model_path --eval_dir ../experiments/eval_iter_$index > ../experiments/logs/$index"_iter.log"
index='0.2'
model_path="../experiments/uncertainty_models_ecp_mb2-ssd-lite_2019-07-30-01-07-54/query_iter_0.2/mb2-ssd-lite-Epoch-55-Loss-6.087069988250732.pth"
python wy_eval_ssd_ecp.py --net mb2-ssd-lite --dataset_type ecp --label_file models/ecp_labels.txt --config config/prague_combine_debug.yaml --trained_model $model_path --eval_dir ../experiments/eval_iter_$index > ../experiments/logs/$index"_iter.log"
index='0.3'
model_path="../experiments/uncertainty_models_ecp_mb2-ssd-lite_2019-07-30-01-07-54/query_iter_0.3/mb2-ssd-lite-Epoch-85-Loss-5.93371259598505.pth"
python wy_eval_ssd_ecp.py --net mb2-ssd-lite --dataset_type ecp --label_file models/ecp_labels.txt --config config/prague_combine_debug.yaml --trained_model $model_path --eval_dir ../experiments/eval_iter_$index > ../experiments/logs/$index"_iter.log"
index='0.4'
model_path="../experiments/uncertainty_models_ecp_mb2-ssd-lite_2019-07-30-01-07-54/query_iter_0.4/mb2-ssd-lite-Epoch-65-Loss-5.488185859861828.pth"
python wy_eval_ssd_ecp.py --net mb2-ssd-lite --dataset_type ecp --label_file models/ecp_labels.txt --config config/prague_combine_debug.yaml --trained_model $model_path --eval_dir ../experiments/eval_iter_$index > ../experiments/logs/$index"_iter.log"
'''
index='0.5'
model_path="../experiments/uncertainty_models_ecp_mb2-ssd-lite_2019-07-30-01-07-54/query_iter_0.5/mb2-ssd-lite-Epoch-5-Loss-5.693613279433477.pth"
python wy_eval_ssd_ecp.py --net mb2-ssd-lite --dataset_type ecp --label_file models/ecp_labels.txt --config config/prague_combine_debug.yaml --trained_model $model_path --eval_dir ../experiments/eval_iter_$index > ../experiments/logs/$index"_iter.log"
index='0.6'
model_path="../experiments/uncertainty_models_ecp_mb2-ssd-lite_2019-07-30-01-07-54/query_iter_0.6/mb2-ssd-lite-Epoch-75-Loss-5.149186702001662.pth"
python wy_eval_ssd_ecp.py --net mb2-ssd-lite --dataset_type ecp --label_file models/ecp_labels.txt --config config/prague_combine_debug.yaml --trained_model $model_path --eval_dir ../experiments/eval_iter_$index > ../experiments/logs/$index"_iter.log"
index='0.7'
model_path="../experiments/uncertainty_models_ecp_mb2-ssd-lite_2019-07-30-01-07-54/query_iter_0.7/mb2-ssd-lite-Epoch-20-Loss-5.132543563842773.pth"
python wy_eval_ssd_ecp.py --net mb2-ssd-lite --dataset_type ecp --label_file models/ecp_labels.txt --config config/prague_combine_debug.yaml --trained_model $model_path --eval_dir ../experiments/eval_iter_$index > ../experiments/logs/$index"_iter.log"
index='0.8'
model_path="../experiments/uncertainty_models_ecp_mb2-ssd-lite_2019-07-30-01-07-54/query_iter_0.8/mb2-ssd-lite-Epoch-99-Loss-5.040320260184152.pth"
python wy_eval_ssd_ecp.py --net mb2-ssd-lite --dataset_type ecp --label_file models/ecp_labels.txt --config config/prague_combine_debug.yaml --trained_model $model_path --eval_dir ../experiments/eval_iter_$index > ../experiments/logs/$index"_iter.log"
index='0.9'
model_path="../experiments/uncertainty_models_ecp_mb2-ssd-lite_2019-07-30-01-07-54/query_iter_0.9/mb2-ssd-lite-Epoch-45-Loss-4.940915062313988.pth"
python wy_eval_ssd_ecp.py --net mb2-ssd-lite --dataset_type ecp --label_file models/ecp_labels.txt --config config/prague_combine_debug.yaml --trained_model $model_path --eval_dir ../experiments/eval_iter_$index > ../experiments/logs/$index"_iter.log"
index='1.0'
model_path="../experiments/uncertainty_models_ecp_mb2-ssd-lite_2019-07-30-01-07-54/query_iter_1.0/mb2-ssd-lite-Epoch-0-Loss-5.2300946826026555.pth"
python wy_eval_ssd_ecp.py --net mb2-ssd-lite --dataset_type ecp --label_file models/ecp_labels.txt --config config/prague_combine_debug.yaml --trained_model $model_path --eval_dir ../experiments/eval_iter_$index > ../experiments/logs/$index"_iter.log"

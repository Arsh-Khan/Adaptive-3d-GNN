#!/bin/bash
#SBATCH --job-name=AF_INFER
#SBATCH --out="jobs/slurm-%j_a3dGNN_cp30401_Nsample15.out"
#SBATCH --time=96:00:00
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# og
python a3dGNN/train.py -d ../../alphafold_finetune//datasets_alphafold_finetune/pmhc_finetune/train.labels -s ../../alphafold_finetune//datasets_alphafold_finetune/pmhc_finetune/val.labels --graph_path GNN_gen/alphafold_graph_add_peptide_feature/ --partition_ratio 0.8:0.2 -m GTN  -o a3dGNN_pytorch_test --best_model_criterion acc --mu_sigma_criterion ce --constrained-names ../../alphafold_finetune/datasets_alphafold_finetune/pmhc_finetune/train_classii.label

# Test
# lam=1 lam2=1
python train.py -d ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/train.labels -s ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/val.labels --graph_path ./GNN_gen/alphafold_graph_add_peptide_feature/ --partition_ratio 0.8:0.2 -m GTN  -o a3dGNN_pytorch_test --best_model_criterion acc --mu_sigma_criterion ce --constrained-names ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/train_classii.label  --constrained-names-val ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/val_classii.label --clustering y --modify_way remove --lam 1 --lam2 1 --weight-data y --loss_function ce
python train.py -d ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/train.labels -s ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/val.labels --graph_path ./GNN_gen/alphafold_graph_add_peptide_feature/ --partition_ratio 0.8:0.2 -m GTN  -o a3dGNN_pytorch_test --best_model_criterion acc --mu_sigma_criterion ce --constrained-names ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/train_classii.label  --constrained-names-val ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/val_classii.label --clustering y --modify_way remove --lam 1 --lam2 1 --weight-data y --loss_function fl

python train.py -d ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/train.labels -s ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/val.labels --graph_path ./GNN_gen/alphafold_graph_add_peptide_feature/ --partition_ratio 0.8:0.2 -m GTN  -o a3dGNN_pytorch_test --best_model_criterion acc --mu_sigma_criterion ce --constrained-names ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/train_classii.label  --constrained-names-val ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/val_classii.label --clustering y --modify_way remove --lam 1 --lam2 1 --weight-data y --loss_function ce --n_sample 30
python train.py -d ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/train.labels -s ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/val.labels --graph_path ./GNN_gen/alphafold_graph_add_peptide_feature/ --partition_ratio 0.8:0.2 -m GTN  -o a3dGNN_pytorch_test --best_model_criterion acc --mu_sigma_criterion ce --constrained-names ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/train_classii.label  --constrained-names-val ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/val_classii.label --clustering y --modify_way remove --lam 1 --lam2 1 --weight-data y --loss_function fl --n_sample 30

# lam=1  lam2=10
python train.py -d ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/train.labels -s ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/val.labels --graph_path ./GNN_gen/alphafold_graph_add_peptide_feature/ --partition_ratio 0.8:0.2 -m GTN  -o a3dGNN_pytorch_test --best_model_criterion acc --mu_sigma_criterion ce --constrained-names ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/train_classii.label  --constrained-names-val ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/val_classii.label --clustering y --modify_way remove --lam 1 --lam2 10 --weight-data y --loss_function ce
python train.py -d ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/train.labels -s ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/val.labels --graph_path ./GNN_gen/alphafold_graph_add_peptide_feature/ --partition_ratio 0.8:0.2 -m GTN  -o a3dGNN_pytorch_test --best_model_criterion acc --mu_sigma_criterion ce --constrained-names ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/train_classii.label  --constrained-names-val ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/val_classii.label --clustering y --modify_way remove --lam 1 --lam2 10 --weight-data y --loss_function fl

python train.py -d ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/train.labels -s ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/val.labels --graph_path ./GNN_gen/alphafold_graph_add_peptide_feature/ --partition_ratio 0.8:0.2 -m GTN  -o a3dGNN_pytorch_test --best_model_criterion acc --mu_sigma_criterion ce --constrained-names ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/train_classii.label  --constrained-names-val ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/val_classii.label --clustering y --modify_way remove --lam 1 --lam2 10 --weight-data y --loss_function ce --n_sample 30
python train.py -d ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/train.labels -s ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/val.labels --graph_path ./GNN_gen/alphafold_graph_add_peptide_feature/ --partition_ratio 0.8:0.2 -m GTN  -o a3dGNN_pytorch_test --best_model_criterion acc --mu_sigma_criterion ce --constrained-names ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/train_classii.label  --constrained-names-val ./datasets_alphafold_finetune/pmhc_finetune/pmhc_finetune/val_classii.label --clustering y --modify_way remove --lam 1 --lam2 10 --weight-data y --loss_function fl --n_sample 30





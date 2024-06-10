## Introduction

This is the code for NAACL 2024 findings paper "[Uncertainty Estimation on Sequential Labeling via Uncertainty Transmission](https://arxiv.org/pdf/2311.08726)". 



## Configure Enviroment

The enviroment requires you to install [Flair](https://github.com/flairNLP/flair) at first.

The other package versions

python=3.8

pytorch=1.10


## Running Command

Restaurant Dataset
```
python run_ner_unc.py --cal_unique_predict_scores True --dataset_name NER_ENGLISH_RESTAURANT --model_name_or_path roberta-base --batch_size 1 --learning_rate 5e-05 --num_epochs 50 --output_dir result/ood_NER_ENGLISH_RESTAURANT_056_arxiv --latent_dim 20 --device cuda:0 --maf_layers 0 --output_dim 25 --unc_method slpn --prior_mode local --use_uce 1 --normalize_dis 1 --neighbor_mode self_att_proval --self_att_dk_ratio 8 --self_att_droput 0.00 --only_test 0 --te_task ood --leave_out_labels "['Price', 'Hours']"
```

Simple Movie Dataset
```
python run_ner_unc.py --cal_unique_predict_scores True --dataset_name  NER_ENGLISH_MOVIE_SIMPLE --model_name_or_path roberta-base --batch_size 1 --learning_rate 5e-05 --num_epochs 50 --output_dir result/ood_NER_ENGLISH_MOVIE_SIMPLE_057 --latent_dim 20 --device cuda:2 --maf_layers 0 --output_dim 37 --unc_method slpn --prior_mode local --use_uce 1 --normalize_dis 1 --neighbor_mode self_att_proval --self_att_dk_ratio 4 --self_att_droput 0.00 --only_test 0 --te_task ood --leave_out_labels "['SONG', 'REVIEW', 'TRAILER']"
```


Complex Movie Dataset
```
python run_ner_unc.py --cal_unique_predict_scores True --dataset_name  NER_ENGLISH_MOVIE_COMPLEX --model_name_or_path roberta-base --batch_size 1 --learning_rate 5e-05 --num_epochs 50 --output_dir result/ood_NER_ENGLISH_MOVIE_COMPLEX_056 --latent_dim 20 --device cuda:2 --maf_layers 0 --output_dim 37 --unc_method slpn --prior_mode local --use_uce 1 --normalize_dis 1 --neighbor_mode self_att_proval --self_att_dk_ratio 8 --self_att_droput 0.00 --only_test 0 --te_task ood --leave_out_labels "['Award', 'Soundtrack', 'Quote']"
```

Once you have well-trained a model and only want to conduct evaluation process. 
You can set ```"--only_test 1"```.
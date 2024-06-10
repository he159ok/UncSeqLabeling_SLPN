import inspect
import json
import logging
import os
import sys
from dataclasses import dataclass, field

import torch
from transformers import HfArgumentParser

import flair
from flair import set_seed
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger, SequenceTaggerWiUnc, SequenceTagger_Dropout
from flair.trainers import ModelTrainer
from utility.ood_dataset import *

import utility
import utility.extract_labels as extract_labels

logger = logging.getLogger("flair")
logger.setLevel(level="INFO")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "The model checkpoint for weights initialization."},
    )
    layers: str = field(default="-1", metadata={"help": "Layers to be fine-tuned."})
    subtoken_pooling: str = field(
        default="first",
        metadata={"help": "Subtoken pooling strategy used for fine-tuned."},
    )
    hidden_size: int = field(default=256, metadata={"help": "Hidden size for NER model."})
    use_crf: bool = field(default=False, metadata={"help": "Whether to use a CRF on-top or not."})


@dataclass
class TrainingArguments:
    num_epochs: int = field(default=10, metadata={"help": "The number of training epochs."})
    batch_size: int = field(default=8, metadata={"help": "Batch size used for training."})
    mini_batch_chunk_size: int = field(
        default=1,
        metadata={"help": "If smaller than batch size, batches will be chunked."},
    )
    learning_rate: float = field(default=5e-05, metadata={"help": "Learning rate"})
    seed: int = field(default=42, metadata={"help": "Seed used for reproducible fine-tuning results."})
    device: str = field(default="cuda:0", metadata={"help": "CUDA device string."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for optimizer."})
    embeddings_storage_mode: str = field(default="none", metadata={"help": "Defines embedding storage method."})


@dataclass
class FlertArguments:
    context_size: int = field(default=0, metadata={"help": "Context size when using FLERT approach."})
    respect_document_boundaries: bool = field(
        default=False,
        metadata={"help": "Whether to respect document boundaries or not when using FLERT."},
    )



@dataclass
class DataArguments:
    dataset_name: str = field(
        metadata={"help": "Flair NER dataset name."}
    )
    dataset_arguments: str = field(default="", metadata={"help": "Dataset arguments for Flair NER dataset."})
    output_dir: str = field(
        default="resources/taggers/ner",
        metadata={"help": "Defines output directory for final fine-tuned model."},
    )
    use_small: int = field(default=0, metadata={"help": "if 1 use 0.05 for debug, else 0 use full dataset"})

### 0. design arguments for uncertainties
@dataclass
class UncertaintyArguments:
    unc_method: str = field(default=None, metadata={"help": "Flair NER dataset name. [slpn,dp]"})
    input_dims: str = field(default="[768]", metadata={"help": "the input dim to the latent embedding encoder"})
    hidden_dims: str = field(default="[100]", metadata={"help": "the hidden dim to the latent embedding encoder"})
    latent_dim: int = field(default=50, metadata={"help": "the output (latent) dim to the latent embedding encoder"})
    output_dim: int = field(default=17, metadata={"help": "the output (latent) dim to the latent embedding encoder"})
    k_lipschitz: float = field(default=None, metadata={"help": "Lipschitz constant. float or None (if no lipschitz)"})
    kernel_dim: str = field(default=None, metadata={"help": "kernal size for comnv archi"})
    no_density: bool = field(default=None, metadata={"help": "Use density estimation or not. boolean"})
    density_type: str = field(default='radial_flow', metadata={"help": "Density type. string"})
    n_density: int = field(default=8, metadata={"help": "# Number of density components. int"})
    budget_function: str = field(default='id', metadata={"help": "Budget function name applied on class count. name"})
    unc_seed: int = field(default=123, metadata={"help": "# seed of random among uncertainty quantification"})
    radial_layers: int = field(default=10, metadata={"help": "# number of radial_layers in normalize flow"})
    maf_layers: int = field(default=0, metadata={"help": "# number of maf_layers in normalize flow"})
    gaussian_layers: int = field(default=0, metadata={"help": "# number of gaussian_layers in normalize flow"})
    use_batched_flow: bool = field(default=True, metadata={"help": ""})
    alpha_evidence_scale: str = field(default='latent-new', metadata={"help": " ['latent-old', 'latent-new', 'latent-new-plus-classes', None]"})
    prior_mode: str = field(default='global', metadata={"help": " ['global', 'local', 'global_local']"})
    neighbor_mode: str = field(default=None, metadata={"help": " ['self_att', 'None', 'closest', 'simple_project']"})
    use_uce: int = field(default=None, metadata={"help": "force using ce loss if 0, and force using uce loss if 1"})
    only_test: int = field(default=0, metadata={"help": "1: only do test; 0: do both test and training"})
    normalize_dis: int = field(default=1, metadata={"help": "1: do global/local distribution normalization; 0: donot do distri normalization"})
    draw_tSNE: int = field(default=0, metadata={"help": "1: draw tSNE figure; 0: donot draw tSNE figure"})
    self_att_dk_ratio: int = field(default=8, metadata={"help": "dk = class_num / self_att_dk_ratio"})
    self_att_droput: float = field(default=0.05, metadata={"help": "dropout ratio in self_att"})
    cal_unique_predict_scores: bool= field(default=False, metadata={"help": "whether calculate and use the unique predicted scores in the testing process"})
    use_stable: bool= field(default=False, metadata={"help": "whether use stable skills to stable the training"})
    use_var_metric: bool= field(default=True, metadata={"help": "whether use stable skills to stable the training"})

    # below is related to ood detection task
    te_task: str = field(default='mis', metadata={"help": " ['mis', 'ood']"})
    oodset_name: str = field(default="", metadata={"help": "Dataset in OOD usage for Flair NER dataset."})
    ood_ratio: float = field(default=0.5, metadata={"help": "the ratio of OOD samples among the original samples"})
    leave_out_labels: str = field(default="['Price', 'Hours']", metadata={"help": "leave out labels"})
    exclude_split_ratio: str = field(default="[0.8, 0.9, 1.0]", metadata={"help": "the ratio used to split excluded split ratio"})
    ood_eval_mode: str = field(default='entity_ori', metadata={"help": "entity_ori: use the default entity_eval; entity_fp: take the ws as fp; token_ori: use the default token_eval"})
    ### when ood_eval_mode is in [entity_fp, token_ori], the wrong_span_based scores are uselessful, please ignore them.

    # below is related to sequential training
    use_seq_training: bool = field(default=False, metadata={"help": "whether use and the sequential loss"})
    pretr_ep_num: int = field(default=20, metadata={"help": "init epoch used for the training pre_trained model only"})
    load_pretr_bert_latentenc: bool = field(default=False, metadata={"help": "whether load the pretr_bert_latentenc or not"})
    well_fined_model_path: str = field(default=None, metadata={"help": "the path to load pretr_bert_latentecn_emb"})
    pretr_lr: float = field(default=0.001, metadata={"help": "the lr rate in the pre-training"})
    pretr_weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for optimizer in pretraining."})

    # below is related to combine training
    use_multitask_training: bool= field(default=False, metadata={"help": "whether use and the multi_task loss"})
    bert_loss_w: float=field(default=0.1, metadata={"help": "weight of using the bert loss"})

    # below is baseline - dropout related parameters
    main_dropout: float = field(default=0.00, metadata={"help": "dropout ratio in framework"}) # 0.15
    test_dropout_num:  int = field(default=10, metadata={"help": "dropout time in the testing process"})


def get_flair_corpus(data_args):
    ner_task_mapping = {}

    for name, obj in inspect.getmembers(flair.datasets.sequence_labeling):
        if inspect.isclass(obj):
            if name.startswith("NER") or name.startswith("CONLL") or name.startswith("WNUT"):
                ner_task_mapping[name] = obj

    dataset_args = {}
    dataset_name = data_args.dataset_name

    if data_args.dataset_arguments:
        dataset_args = json.loads(data_args.dataset_arguments)

    if dataset_name not in ner_task_mapping:
        raise ValueError(f"Dataset name {dataset_name} is not a valid Flair datasets name!")

    return ner_task_mapping[dataset_name](**dataset_args)


def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments, FlertArguments, DataArguments, UncertaintyArguments))


    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (
            model_args,
            training_args,
            flert_args,
            data_args,
            unc_args,
        ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (
            model_args,
            training_args,
            flert_args,
            data_args,
            unc_args,
        ) = parser.parse_args_into_dataclasses()


    unc_args.num_epochs = training_args.num_epochs 
    unc_args.batch_size = training_args.batch_size 


    set_seed(training_args.seed)

    flair.device = training_args.device

    if training_args.device == "cuda:0":
        torch.cuda.set_device(0)
    elif training_args.device == "cuda:2":
        torch.cuda.set_device(2)
    elif training_args.device == "cuda:3":
        torch.cuda.set_device(3)
    else:
        raise ValueError

    # detect



    if data_args.use_small == 1 and unc_args.te_task != 'ood': # used for debug
        corpus = get_flair_corpus(data_args).downsample(0.05)
    else:
        corpus = get_flair_corpus(data_args)


    logger.info(corpus)

    if unc_args.use_seq_training==True and unc_args.use_multitask_training==True:
        raise ValueError("cannot unc_args.use_seq_training==True and unc_args.use_multitask_training==True")

    # avoid the previous results are overlapped
    if "debug" not in data_args.output_dir and unc_args.only_test==0:
        res_path = os.path.join(os.getcwd(), data_args.output_dir)
        if os.path.exists(res_path):
            raise ValueError(f"the save path is repeated in current: {res_path}")

    if data_args.dataset_name == "NER_ENGLISH_RESTAURANT":
        number_span_class = 9


    tag_type: str = "ner"
    # tag_dictionary = corpus.make_label_dictionary(tag_type, add_unk=False)
    tag_dictionary = corpus.make_label_dictionary(tag_type)
    logger.info(tag_dictionary)
    leave_out_label_list = None

    # below is for OOD testing dataset preparing
    if unc_args.te_task == 'ood':
        ##### use leave-out-method for the OOD construction

        # count the label word list
        tag_list = list(corpus.get_label_distribution().keys())

        # count the distribution of the remove-one word
        for ele_tag in tag_list:
            if '<unk>' == ele_tag:
                continue
            # remained_num, new_tr, new_dev, new_te = count_removed_class_dataset(corpus, remove_word_list=[])
            exclued_num, inclued_num = count_removed_class_dataset(corpus, remove_word_list=[ele_tag]) # exclued_num: potential training; inclued_num: potential testing
            print(f'{ele_tag} case: exclued_num-(train)-{exclued_num}, included_num-(test)-{inclued_num}')
        # assert 1==0

        # count the distribution of the remove-multitple word
        leave_out_label_list = eval(unc_args.leave_out_labels)
        exclude_split_ratio = eval(unc_args.exclude_split_ratio)
        exclued_num, inclued_num, excluded_set, included_set = count_removed_class_dataset(corpus, remove_word_list=leave_out_label_list, return_new_set=True)
        print(f'use leave out label list of {leave_out_label_list}: the exclued_num-(train)-{exclued_num}, included_num-(test)-{inclued_num}')
        # replace the leaveout_labels into OOD in the included_set
        included_set = replace_lo_label_ood(included_set, leave_out_label_list)
        corpus = merge_ood_corpus(corpus, excluded_set, included_set, exclude_split_ratio)

        tag_dictionary = corpus.make_label_dictionary(tag_type)
        logger.info(tag_dictionary)

        if data_args.use_small == 1:  # used for debug
            corpus.downsample(0.05)


        if unc_args.well_fined_model_path is not None:
            assert unc_args.load_pretr_bert_latentenc==True
        if unc_args.use_seq_training and unc_args.well_fined_model_path==None:
            unc_args.well_fined_model_path = data_args.output_dir + '_partfine'

    # above is for OOD testing dataset preparing



    embeddings = TransformerWordEmbeddings(
        model=model_args.model_name_or_path,
        layers=model_args.layers,
        subtoken_pooling=model_args.subtoken_pooling,
        fine_tune=True,
        # fine_tune=False,
        use_context=flert_args.context_size,
        respect_document_boundaries=flert_args.respect_document_boundaries,
    )


    ### 1. design a new SequenceTagger with uncertianty function
    if unc_args.unc_method is None or (unc_args.unc_method == 'slpn' and unc_args.use_uce!=2):
        tagger = SequenceTaggerWiUnc(
            hidden_size=model_args.hidden_size,
            embeddings=embeddings,
            tag_dictionary=tag_dictionary,
            tag_type=tag_type,
            dropout=unc_args.main_dropout, # newly added after dropout baseline
            use_crf=model_args.use_crf,
            use_rnn=False,
            reproject_embeddings=False,



            unc_args=unc_args
        )
    elif unc_args.unc_method == 'dp': # baseline - dropout
        tagger = SequenceTagger_Dropout(
            hidden_size=model_args.hidden_size,
            embeddings=embeddings,
            tag_dictionary=tag_dictionary,
            tag_type=tag_type,
            dropout=unc_args.main_dropout,
            use_crf=False,
            use_rnn=False,
            reproject_embeddings=False,

            unc_args=unc_args
        )



    if unc_args.unc_method == 'slpn':
        ### Calculate category distribution
        gl_cat_list, gl_cat_dict, gl_cat_dis_list, token_cat_list_dict, token_cat_dis_list_dict = tagger.extract_cat_distribution(corpus)
        tagger.__setattr__('N_gl_cat_list', gl_cat_list)
        tagger.__setattr__('N_gl_cat_dict', gl_cat_dict)
        tagger.__setattr__('P_gl_cat_list', gl_cat_dis_list)
        tagger.__setattr__('N_token_cat_list_dict', token_cat_list_dict)
        tagger.__setattr__('P_token_cat_list', token_cat_dis_list_dict)



    trainer = ModelTrainer(tagger, corpus)

    ## this is drawed before the loading
    if unc_args.draw_tSNE == 1:
        print('start draw tSNE figures')
        trainer.draw_tSNE(
            base_path=data_args.output_dir,
            eval_mini_batch_size=training_args.batch_size,
            mode='latent',
            save_name_pre='pre_train_',
            use_data='te', # ['tr', 'val', 'te']
        )

    if unc_args.only_test == 0:

        if unc_args.use_seq_training==True: # use seq training mode
            tagger.unc_args.finished_pretr_process = False
            if unc_args.load_pretr_bert_latentenc==True:
                # load well-fine tuned transformer
                if os.path.exists(os.path.join(unc_args.well_fined_model_path, "best-model.pt")):
                    print('use the best model for well-fine tunned partial model')
                    trainer.model.load_state_dict(
                        trainer.model.load(os.path.join(unc_args.well_fined_model_path, "best-model.pt")).state_dict())
                elif os.path.exists(os.path.join(unc_args.well_fined_model_path, "final-model.pt")):
                    print('use the final model for well-fine tunned partial model')
                    trainer.model.load_state_dict(
                        trainer.model.load(os.path.join(unc_args.well_fined_model_path, "final-model.pt")).state_dict())
                else:
                    raise ValueError(f'a pre-trained pt file for well-fine tunned partial model is not found under {unc_args.well_fined_model_path}')
            else:
                # fine-tune the pre_trained transformer
                trainer.fine_tune(
                    unc_args.well_fined_model_path,
                    learning_rate=unc_args.pretr_lr,
                    mini_batch_size=training_args.batch_size,
                    mini_batch_chunk_size=training_args.mini_batch_chunk_size,
                    max_epochs=unc_args.pretr_ep_num,
                    embeddings_storage_mode=training_args.embeddings_storage_mode,
                    weight_decay=training_args.weight_decay,
                    use_small=data_args.use_small,
                    use_final_model_for_eval=False,
                    partial_pretrain=unc_args.use_seq_training
                )

            tagger.unc_args.finished_pretr_process = True # (very important) indicate the model has finished the learning of seq_training
            trainer = ModelTrainer(tagger, corpus)


        trainer.fine_tune(
            data_args.output_dir,
            learning_rate=training_args.learning_rate,
            mini_batch_size=training_args.batch_size,
            mini_batch_chunk_size=training_args.mini_batch_chunk_size,
            max_epochs=training_args.num_epochs,
            embeddings_storage_mode=training_args.embeddings_storage_mode,
            weight_decay=training_args.weight_decay,
            use_small=data_args.use_small,
            use_final_model_for_eval=False,
        )

        torch.save(model_args, os.path.join(data_args.output_dir, "model_args.bin"))
        torch.save(training_args, os.path.join(data_args.output_dir, "training_args.bin"))


        # below starts the testing process
        if os.path.exists(os.path.join(data_args.output_dir,  "best-model.pt")):
            print('use the best model for eval')
            trainer.model.load_state_dict(trainer.model.load(os.path.join(data_args.output_dir,  "best-model.pt")).state_dict())
        elif os.path.exists(os.path.join(data_args.output_dir, "final-model.pt")):
            print('use the final model for eval')
            trainer.model.load_state_dict(trainer.model.load(os.path.join(data_args.output_dir,  "final-model.pt")).state_dict())
        else:
            raise ValueError(f'a pre-trained pt file is not found under {data_args.output_dir}')



    elif unc_args.only_test == 1:
        # add model load operation below


        if os.path.exists(os.path.join(data_args.output_dir,  "best-model.pt")):
            print('use the best model for eval')
            trainer.model.load_state_dict(trainer.model.load(os.path.join(data_args.output_dir,  "best-model.pt")).state_dict())
        elif os.path.exists(os.path.join(data_args.output_dir, "final-model.pt")):
            print('use the final model for eval')
            trainer.model.load_state_dict(trainer.model.load(os.path.join(data_args.output_dir,  "final-model.pt")).state_dict())
        else:
            raise ValueError(f'a pre-trained pt file is not found under {data_args.output_dir}')


    ## this is drawed after the loading
    if unc_args.draw_tSNE == 1:
        print('start draw tSNE figures')
        trainer.draw_tSNE(
            base_path=data_args.output_dir,
            eval_mini_batch_size=training_args.batch_size,
            mode='latent',
            save_name_pre='post_train_',
            use_data = 'te',  # ['tr', 'val', 'te']
        )
        raise ValueError('the draw_tSNE mode is defaultly incontinuous')

    ## do default test in any process
    write_file_name = os.path.join(data_args.output_dir, 'unc_cal.txt')
    if os.path.exists(write_file_name):
        os.remove(write_file_name)
    main_evaluation_metric = ('micro avg', 'f1-score')

    if unc_args.unc_method in ['slpn']:

        print('start aleatoric for test in sfmx') # should be correct for traditional probability calculation
        final_score_sfmx1 = trainer.final_test(
            base_path=data_args.output_dir,
            eval_mini_batch_size=training_args.batch_size,
            num_workers=None,
            main_evaluation_metric=main_evaluation_metric,
            gold_label_dictionary_for_eval=None, # for OOD test, should definitely keep as None
            exclude_labels=[],
            use_alpha=False,
            sfmx_mode='sfmx',
            uncertainty_metrics=['alea'],
            show_classification_res=False,
            te_task=unc_args.te_task,
            leave_out_labels=leave_out_label_list,
            cal_unique_predict_scores=unc_args.cal_unique_predict_scores,
            use_var_metric=unc_args.use_var_metric,
            ood_eval_mode=unc_args.ood_eval_mode,
        )
        #


        print('start aleatoric for test in mean') # should be correct for alpha-based probability
        final_score_sfmx2 = trainer.final_test(
            base_path=data_args.output_dir,
            eval_mini_batch_size=training_args.batch_size,
            num_workers=None,
            main_evaluation_metric=main_evaluation_metric,
            gold_label_dictionary_for_eval=None,
            exclude_labels=[],
            use_alpha=False,
            sfmx_mode='mean',
            uncertainty_metrics=['alea'],
            show_classification_res=False,
            te_task=unc_args.te_task,
            leave_out_labels=leave_out_label_list,
            cal_unique_predict_scores=unc_args.cal_unique_predict_scores,
            use_var_metric=unc_args.use_var_metric,
            ood_eval_mode=unc_args.ood_eval_mode,
        )

        print('start diss or vacu or epis or entr test by alpha')
        final_score_alpha = trainer.final_test(
            base_path=data_args.output_dir,
            eval_mini_batch_size=training_args.batch_size,
            num_workers=None,
            main_evaluation_metric=main_evaluation_metric,
            gold_label_dictionary_for_eval=None,
            exclude_labels=[],
            use_alpha=True,
            sfmx_mode=None,
            uncertainty_metrics=['diss', 'vacu', 'entr_mean', 'epis'],
            show_classification_res=True,
            te_task=unc_args.te_task,
            leave_out_labels=leave_out_label_list,
            cal_unique_predict_scores=unc_args.cal_unique_predict_scores,
            use_var_metric=unc_args.use_var_metric,
            ood_eval_mode=unc_args.ood_eval_mode,
        )

    elif unc_args.unc_method == 'dp':
        print('start aleatoric, epis, entropy, for test in sfmx') # should be correct for traditional probability calculation
        final_score_sfmx1 = trainer.final_test(
            base_path=data_args.output_dir,
            eval_mini_batch_size=training_args.batch_size,
            num_workers=None,
            main_evaluation_metric=main_evaluation_metric,
            gold_label_dictionary_for_eval=None,
            exclude_labels=[],
            use_alpha=False,
            sfmx_mode='sfmx',
            uncertainty_metrics=['alea', 'epis', 'entr_sfmx'],
            show_classification_res=False,
            test_dropout_num=unc_args.test_dropout_num,
            te_task=unc_args.te_task,
            leave_out_labels=leave_out_label_list,
            cal_unique_predict_scores=unc_args.cal_unique_predict_scores,
            use_var_metric=unc_args.use_var_metric,
            ood_eval_mode=unc_args.ood_eval_mode,
        )


if __name__ == "__main__":
    main()

import sys
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from DELRec.SR_models.SR_models import SASRec, Caser, GRU
import os
from DELRec.distill_pattern_from_conventional_SR_models.train import training_of_first_stage
from DELRec.llms_based_sr.train import training_of_second_stage
from test_DELRec import test


def main(args):
    pl.seed_everything(args.seed)
    logger = TensorBoardLogger(save_dir='./log/', name=args.log_dir)
    args.logger = logger
    if not os.path.exists(args.second_model_path):
        os.makedirs(args.second_model_path)

    if args.mode == 'train':
        learned_soft_prompt = training_of_first_stage(args)
        training_of_second_stage(args, learned_soft_prompt)
    else:
        test(args)

    sys.exit()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = ArgumentParser()
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], type=str)
    parser.add_argument('--parallelize', default=True, type=bool)
    parser.add_argument('--llm_path', default="./flan-t5-xl", type=str)
    parser.add_argument('--llm', default="t5", choices=['roberta', 'bert', 'albert', 'gpt', 'gpt2', 'opt', 'llama'],
                        type=str)
    parser.add_argument('--SR_model', default="SASRec", choices=['SASRec', 'Caser', 'GRU'], type=str)
    parser.add_argument('--ICL_length', default=4, choices=[4, 6], type=int)
    parser.add_argument('--ICL_back', default=3, choices=[3, 5], type=int)
    parser.add_argument('--candidate_size', default=20, choices=[15, 20, 25], type=int)
    parser.add_argument('--load_soft_prompt_log', default=False, type=bool)
    parser.add_argument('--log_dir', default='record_logs', type=str)
    parser.add_argument('--seed', default=1234, type=int)
    # parser.add_argument('--predicted_items', default=SASRec, choices=[SASRec, Caser, GRU])

    parser.add_argument('--first_shuffle', default=False, type=bool)
    parser.add_argument('--first_teacher_forcing', default=False, type=bool)
    parser.add_argument('--first_predict_eos_token', default=False, type=bool)
    parser.add_argument('--first_batch_size', default=20, type=int)
    parser.add_argument('--first_decoder_max_length', default=20, type=int)
    parser.add_argument('--first_truncate_method', default="tail", choices=['head', 'tail'], type=str)
    parser.add_argument('--first_max_seq_length', default=1065, type=int)
    parser.add_argument('--first_learned_soft_prompt_path', default="./learned_soft_prompt.ckpt", type=str)
    parser.add_argument('--first_total_epoch', default=1000, type=int)
    parser.add_argument('--first_lr', default=5e-3, type=float)
    parser.add_argument('--first_weight_decay', default=1e-5, type=float)
    parser.add_argument('--first_num_warmup_steps', default=400, type=int)
    parser.add_argument('--first_num_training_steps', default=800, type=int)
    parser.add_argument('--first_gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--first_eval_every_steps', default=2, type=int)

    parser.add_argument('--TA_prompt_id', default=0, choices=[0, 1, 2], type=int)  # different datasets
    parser.add_argument('--TA_load', default=False, type=bool)
    parser.add_argument('--TA_load_log', default=False, type=bool)
    parser.add_argument('--TA_all_item_titles_path', default='./title_set.csv', type=str)  # with both titles and ids
    parser.add_argument('--TA_log_train_dataset_path', default='TA_dataset1.pkl', type=str)
    parser.add_argument('--TA_log_test_dataset_path', default='TA_dataset2.pkl', type=str)
    parser.add_argument('--TA_seq_with_recommended_size_h', default=-10, choices=[-10, -15], type=int)
    parser.add_argument('--TA_log_validation_dataset_path', default='TA_dataset3.pkl', type=str)
    parser.add_argument('--TA_truncation_seq', default=14, choices=[14, 19], type=int)
    sr_model_paths = {
        'SASRec': './user_interactions_with_text_title_and_predicted_items_by_SASRec',  # from SASRec
        'Caser': './user_interactions_with_text_title_and_predicted_items_by_Caser',  # from Caser
        'GRU': './user_interactions_with_text_title_and_predicted_items_by_GRU'  # from GRU
    }
    path = sr_model_paths.get(parser.parse_args().SR_model, sr_model_paths['SASRec'])
    parser.add_argument('--TA_user_interactions_with_text_title_predicted_by_SR_path', default=path, type=str)

    parser.add_argument('--RPS_prompt_id', default=0, choices=[0, 1, 2], type=int)  # different datasets
    parser.add_argument('--RPS_load', default=False, type=bool)
    parser.add_argument('--RPS_load_log', default=False, type=bool)
    parser.add_argument('--RPS_SR_pre_forw', default=-9, choices=[-9, -14], type=int)
    parser.add_argument('--RPS_all_item_titles_path', default='./title_set.csv', type=str)  # with both titles and ids
    parser.add_argument('--RPS_log_train_dataset_path', default='RPS_dataset1.pkl', type=str)
    parser.add_argument('--RPS_log_test_dataset_path', default='RPS_dataset2.pkl', type=str)
    parser.add_argument('--RPS_seq_with_recommended_size_h', default=-10, choices=[-10, -15], type=int)
    parser.add_argument('--RPS_log_validation_dataset_path', default='RPS_dataset3.pkl', type=str)
    parser.add_argument('--RPS_truncation_seq', default=14, choices=[14, 19], type=int)
    parser.add_argument('--RPS_user_interactions_with_text_title_predicted_by_SR_path', default=path, type=str)

    parser.add_argument('--second_shuffle', default=False, type=bool)
    parser.add_argument('--second_teacher_forcing', default=False, type=bool)
    parser.add_argument('--second_predict_eos_token', default=False, type=bool)
    parser.add_argument('--second_batch_size', default=20, type=int)
    parser.add_argument('--second_decoder_max_length', default=20, type=int)
    parser.add_argument('--second_truncate_method', default="tail", choices=['head', 'tail'], type=str)
    parser.add_argument('--second_max_seq_length', default=1065, type=int)
    parser.add_argument('--second_model_path', default="./model.ckpt", type=str)
    parser.add_argument('--second_total_epoch', default=2000, type=int)
    parser.add_argument('--second_lr', default=1e-4, type=float)
    parser.add_argument('--second_weight_decay', default=1e-6, type=float)
    parser.add_argument('--second_num_warmup_steps', default=800, type=int)
    parser.add_argument('--second_num_training_steps', default=1400, type=int)
    parser.add_argument('--second_gradient_accumulation_steps', default=6, type=int)
    parser.add_argument('--second_eval_every_steps', default=2, type=int)
    parser.add_argument('--second_if_peft', default=True, type=bool)
    parser.add_argument('--second_peft_type', default='ADALORA', type=str)
    parser.add_argument('--second_init_r', default=320, type=int)
    parser.add_argument('--second_lora_alpha', default=32, type=int)
    parser.add_argument('--second_lora_dropout', default=0.05, type=float)
    parser.add_argument('--second_target_modules', default=['q', 'v'], type=str, nargs='+')

    parser.add_argument('--LSR_prompt_id', default=0, choices=[0, 1, 2], type=int)
    parser.add_argument('--LSR_load', default=False, type=bool)
    parser.add_argument('--LSR_load_log', default=False, type=bool)
    parser.add_argument('--LSR_SR_pre_forw', default=-9, choices=[-9, -14], type=int)
    parser.add_argument('--LSR_all_item_titles_path', default='./title_set.csv', type=str)
    parser.add_argument('--LSR_log_train_dataset_path', default='LSR_dataset1.pkl', type=str)
    parser.add_argument('--LSR_log_test_dataset_path', default='LSR_dataset2.pkl', type=str)
    parser.add_argument('--LSR_seq_with_recommended_size_h', default=-10, choices=[-10, -15], type=int)
    parser.add_argument('--LSR_log_validation_dataset_path', default='LSR_dataset3.pkl', type=str)
    parser.add_argument('--LSR_truncation_seq', default=5, type=int)
    parser.add_argument('--LSR_user_interactions_with_text_title_ground_truth_path',
                        default='./user_interactions_with_text_title_and_ground_truth', type=str) # txt version of ./data user interactions.csv

    args = parser.parse_args()
    main(args)

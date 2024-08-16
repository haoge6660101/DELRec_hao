from ..utils import check_suffix, rearrange_string, create_prompt, check_seq_spell
from ..dataload import create_sas_dataset, read_file_portions_based_rows, create_TA_dataset, \
    read_file_portions_based_ratio
from collections import defaultdict
import pandas as pd
import pickle
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from tqdm import tqdm
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np
from openprompt import PromptDataLoader


def creat_TA_datalines(load_log, args):
    if load_log == False:
        with open(args.TA_log_train_dataset_path, 'rb') as f:
            dataset1 = pickle.load(f)

        with open(args.TA_log_test_dataset_path, 'rb') as f:
            dataset2 = pickle.load(f)

        with open(args.TA_log_validation_dataset_path, 'rb') as f:
            dataset3 = pickle.load(f)

        dataset = {
            'train': create_TA_dataset(dataset1, 'train'),
            'validation': create_TA_dataset(dataset2, 'validation'),
            'test': create_TA_dataset(dataset3, 'test')

        }

        pre_dataset = {}
        for split in ['train', 'test', 'validation']:
            pre_dataset[split] = []
            for data in dataset[split]:

                check_seq_spell(data['ICL'])
                check_seq_spell(data['user_can'])
                check_seq_spell(data['user_TA'])
                check_seq_spell(data['m'])
                check_seq_spell(data['m-1'])
                check_seq_spell(data['next'])

                ICL = ', '.join(data['ICL'])
                user_can = ', '.join(data['user_can'])
                user_TA = ', '.join(data['user_TA'])

                if len(data['user_label']) > 0:
                    input_example = InputExample(label=(data['user_label'][0] - 1), guid=int(data['user_id']),
                                                 meta={'ICL': ICL, 'm': data['m'], 'm_1': data['m_1'],
                                                       'user_TA': user_TA, 'next': data['next'], 'user_can': user_can,
                                                       'model_name': data['model_name']})
                    pre_dataset[split].append(input_example)

        train_list_lines = []
        test_list_lines = []
        val_list_lines = []

        for i in pre_dataset['train']:
            train_list_lines.append(i.guid)
        for i in pre_dataset['test']:
            test_list_lines.append(i.guid)
        for i in pre_dataset['validation']:
            val_list_lines.append(i.guid)

        f1, f2, f3 = read_file_portions_based_rows(args.TA_user_interactions_with_text_title_predicted_by_SR_path,
                                                   train_list_lines, test_list_lines, val_list_lines)
    else:
        f1, f2, f3 = read_file_portions_based_ratio(args.TA_user_interactions_with_text_title_predicted_by_SR_path)

    return f1, f2, f3


def load_TA_prompt(args):
    plm, tokenizer, model_config, WrapperClass = load_plm(args.llm, args.llm_path)

    mytemplate_TA = create_prompt('TA', plm, tokenizer, prompt_id=args.TA_prompt_id)
    return mytemplate_TA


def TA_data_partition(ff, args):
    df = pd.read_csv(args.TA_all_item_titles)
    id_set = df.set_index('Title')['ID'].to_dict()
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    ICL = {}
    user_can = {}
    user_label = {}
    m = {}
    m_1 = {}
    user_TA = {}
    next = {}
    model_name = {}

    for line in ff:
        u = line.rstrip().split(' ')[0]
        i = ' '.join(line.rstrip().split(' ')[1:])
        u = int(u)
        i = str(i)
        usernum = max(u, usernum)
        User[u].append(i)

    for user in tqdm(User):
        if len(User[user]) < args.TA_truncation_seq:
            user_can[user] = []
            user_label[user] = []
            user_seq = User[user]
            ICL_forw = user_seq[:args.ICL_length]
            ICL[user] = ICL_forw[:-2]
            m[user] = ICL_forw[-2]
            m_1[user] = ICL_forw[-1]
            ICL_back = user_seq[args.ICL_back:]
            user_TA[user] = ICL_back[:-2]
            next[user] = ICL_back[-1]
            model_name[user] = args.SR_model

        else:
            model_name[user] = args.SR_model
            user_seq = User[user][:args.TA_seq_with_recommended_size_h]
            ICL_forw = user_seq[:args.ICL_length]
            ICL[user] = ICL_forw[:-2]
            m[user] = ICL_forw[-2]
            m_1[user] = ICL_forw[-1]
            ICL_back = user_seq[args.ICL_back:]
            user_TA[user] = ICL_back[:-2]
            next[user] = ICL_back[-1]
            it = ICL_back[-2]
            user_can[user] = []
            user_can[user].append(it)
            user_label[user] = []
            user_label[user].append(id_set[it])

            for _ in range(args.candidate_size - 1):
                can_item = np.random.choice(list(id_set.keys()))
                while can_item in user_can[user] and can_item in User[user]:
                    can_item = np.random.choice(list(id_set.keys()))
                user_can[user].append(can_item)
            np.random.shuffle(user_can[user])
    return [ICL, m, m_1, user_TA, next, user_can, model_name, user_label]


def load_TA_dataset(args):
    TA_load = args.TA_load

    if TA_load == False:

        ta1, ta2, ta3 = creat_TA_datalines(args.TA_load_log, args)
        TA_dataset1 = TA_data_partition(ta1, args)
        TA_dataset2 = TA_data_partition(ta2, args)
        TA_dataset3 = TA_data_partition(ta3, args)

        with open(args.TA_log_train_dataset_path, 'wb') as f:
            pickle.dump(TA_dataset1, f)

        with open(args.TA_log_test_dataset_path, 'wb') as f:
            pickle.dump(TA_dataset2, f)

        with open(args.TA_log_validation_dataset_path, 'wb') as f:
            pickle.dump(TA_dataset3, f)

    else:
        with open(args.TA_log_train_dataset_path, 'rb') as f:
            TA_dataset1 = pickle.load(f)

        with open(args.TA_log_test_dataset_path, 'rb') as f:
            TA_dataset2 = pickle.load(f)

        with open(args.TA_log_validation_dataset_path, 'rb') as f:
            TA_dataset3 = pickle.load(f)

    TA_ori_dataset = {
        'train': create_TA_dataset(TA_dataset1, 'train'),
        'test': create_TA_dataset(TA_dataset2, 'test'),
        'validation': create_TA_dataset(TA_dataset3, 'validation')
    }

    TA_dataset = {}
    for split in ['train', 'test', 'validation']:
        TA_dataset[split] = []
        for data in TA_ori_dataset[split]:

            check_seq_spell(data['ICL'])
            check_seq_spell(data['user_can'])
            check_seq_spell(data['user_TA'])
            check_seq_spell(data['m'])
            check_seq_spell(data['m-1'])
            check_seq_spell(data['next'])

            ICL = ', '.join(data['ICL'])
            user_can = ', '.join(data['user_can'])
            user_TA = ', '.join(data['user_TA'])

            if len(data['user_label']) > 0:
                input_example = InputExample(label=(data['user_label'][0] - 1), guid=int(data['user_id']),
                                             meta={'ICL': ICL, 'm': data['m'], 'm_1': data['m_1'], 'user_TA': user_TA,
                                                   'next': data['next'], 'user_can': user_can,
                                                   'model_name': data['model_name']})
                TA_dataset[split].append(input_example)

    plm, tokenizer, model_config, WrapperClass = load_plm(args.llm, args.llm_path)

    mytemplate = create_prompt('TA', plm, tokenizer, prompt_id=args.TA_prompt_id)
    train_dataloader = PromptDataLoader(dataset=TA_dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass, max_seq_length=args.first_max_seq_length,
                                        decoder_max_length=args.first_decoder_max_length,
                                        batch_size=args.first_batch_size, shuffle=args.first_shuffle,
                                        teacher_forcing=args.first_teacher_forcing,
                                        predict_eos_token=args.first_predict_eos_token,
                                        truncate_method=args.first_truncate_method)

    test_dataloader = PromptDataLoader(dataset=TA_dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                       tokenizer_wrapper_class=WrapperClass, max_seq_length=args.first_max_seq_length,
                                       decoder_max_length=args.first_decoder_max_length,
                                       batch_size=args.first_batch_size, shuffle=args.first_shuffle,
                                       teacher_forcing=args.first_teacher_forcing,
                                       predict_eos_token=args.first_predict_eos_token,
                                       truncate_method=args.first_truncate_method)

    validation_dataloader = PromptDataLoader(dataset=TA_dataset["validation"], template=mytemplate, tokenizer=tokenizer,
                                             tokenizer_wrapper_class=WrapperClass,
                                             max_seq_length=args.first_max_seq_length,
                                             decoder_max_length=args.first_decoder_max_length,
                                             batch_size=args.first_batch_size, shuffle=args.first_shuffle,
                                             teacher_forcing=args.first_teacher_forcing,
                                             predict_eos_token=args.first_predict_eos_token,
                                             truncate_method=args.first_truncate_method)
    return train_dataloader, test_dataloader, validation_dataloader

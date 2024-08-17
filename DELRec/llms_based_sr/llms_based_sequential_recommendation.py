from ..utils import check_suffix, rearrange_string, create_prompt, check_seq_spell
from ..dataload import create_sas_dataset, read_file_portions_based_rows, create_LSR_dataset, \
    read_file_portions_based_ratio
from collections import defaultdict
import pandas as pd
import pickle
from openprompt.plms import load_plm
from tqdm import tqdm
from openprompt.data_utils.utils import InputExample
import numpy as np
from openprompt import PromptDataLoader


def creat_LSR_datalines(load_datalines, args):
    if load_datalines == True:
        with open(args.LSR_log_train_dataset_path, 'rb') as f:
            dataset1 = pickle.load(f)

        with open(args.LSR_log_test_dataset_path, 'rb') as f:
            dataset2 = pickle.load(f)

        with open(args.LSR_log_validation_dataset_path, 'rb') as f:
            dataset3 = pickle.load(f)

        dataset = {
            'train': create_LSR_dataset(dataset1, 'train'),
            'validation': create_LSR_dataset(dataset2, 'validation'),
            'test': create_LSR_dataset(dataset3, 'test')
        }

        pre_dataset = {}
        for split in ['train', 'test', 'validation']:
            pre_dataset[split] = []
            for data in dataset[split]:

                check_seq_spell(data['user_seq'])
                check_seq_spell(data['user_can'])

                user_seq = ', '.join(data['user_seq'])
                user_can = ', '.join(data['user_can'])

                if len(data['user_label']) > 0:
                    input_example = InputExample(label=(data['user_label'][0] - 1), guid=int(data['user_id']),
                                                 meta={'user_seq': user_seq, 'user_can': user_can,
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

        f1, f2, f3 = read_file_portions_based_rows(args.LSR_user_interactions_with_text_title_ground_truth_path,
                                                   train_list_lines, test_list_lines, val_list_lines)
    else:
        f1, f2, f3 = read_file_portions_based_ratio(args.LSR_user_interactions_with_text_title_ground_truth_path)

    return f1, f2, f3


def LSR_data_partition(ff, args):
    df = pd.read_csv(args.TA_all_item_titles)
    id_set = df.set_index('Title')['ID'].to_dict()
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_seq = {}
    user_can = {}
    user_label = {}
    model_name = {}

    for line in ff:
        u = line.rstrip().split(' ')[0]
        i = ' '.join(line.rstrip().split(' ')[1:])
        u = int(u)
        i = str(i)
        usernum = max(u, usernum)
        User[u].append(i)

    for user in tqdm(User):
        if len(User[user]) < args.LSR_truncation_seq:
            user_seq[user] = User[user]
            user_can[user] = []
            user_label[user] = []
            model_name[user] = args.SR_model
        else:
            model_name[user] = args.SR_model
            user_seq[user] = User[user][:-1]
            user_can[user] = []
            it = User[user][-1]
            user_can[user].append(it)
            user_label[user] = []
            user_label[user].append(id_set[it])

            for _ in range(args.candidate_size - 1):
                can_item = np.random.choice(list(id_set.keys()))
                while can_item in user_can[user] and can_item in User[user]:
                    can_item = np.random.choice(list(id_set.keys()))
                user_can[user].append(can_item)
            np.random.shuffle(user_can[user])

    return [user_seq, user_can, user_label, model_name]


def load_LSR_prompt(args):
    plm, tokenizer, model_config, WrapperClass = load_plm(args.llm, args.llm_path)
    mytemplate_LSR = create_prompt('LSR', plm, tokenizer, prompt_id=args.LSR_prompt_id)
    return mytemplate_LSR


def load_LSR_dataset(args):
    load_sas = args.LSR_load
    if load_sas == False:

        lsrf1, lsrf2, lsrf3 = creat_LSR_datalines(False, args)
        LSR_dataset1 = LSR_data_partition(lsrf1, args)
        LSR_dataset2 = LSR_data_partition(lsrf2, args)
        LSR_dataset3 = LSR_data_partition(lsrf3, args)

        with open(args.LSR_log_train_dataset_path, 'wb') as f:
            pickle.dump(LSR_dataset1, f)

        with open(args.LSR_log_test_dataset_path, 'wb') as f:
            pickle.dump(LSR_dataset2, f)

        with open(args.LSR_log_validation_dataset_path, 'wb') as f:
            pickle.dump(LSR_dataset3, f)

    else:

        with open(args.LSR_log_train_dataset_path, 'rb') as f:
            LSR_dataset1 = pickle.load(f)

        with open(args.LSR_log_test_dataset_path, 'rb') as f:
            LSR_dataset2 = pickle.load(f)

        with open(args.LSR_log_validation_dataset_path, 'rb') as f:
            LSR_dataset3 = pickle.load(f)

    LSR_ori_dataset = {
        'train': create_LSR_dataset(LSR_dataset1, 'train'),
        'test': create_LSR_dataset(LSR_dataset2, 'test'),
        'validation': create_LSR_dataset(LSR_dataset3, 'validation')
    }

    LSR_dataset = {}
    for split in ['train', 'test', 'validation']:
        LSR_dataset[split] = []
        for data in LSR_ori_dataset[split]:

            check_seq_spell(data['user_seq'])
            check_seq_spell(data['user_can'])

            user_seq = ', '.join(data['user_seq'])
            user_can = ', '.join(data['user_can'])

            if len(data['user_label']) > 0:
                input_example = InputExample(label=(data['user_label'][0] - 1), guid=int(data['user_id']),
                                             meta={'user_seq': user_seq, 'user_can': user_can,
                                                   'model_name': data['model_name']})
                LSR_dataset[split].append(input_example)

    plm, tokenizer, model_config, WrapperClass = load_plm(args.llm, args.llm_path)
    mytemplate = create_prompt('LSR', plm, tokenizer, prompt_id=args.LSR_prompt_id)

    train_dataloader = PromptDataLoader(dataset=LSR_dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass, max_seq_length=args.first_max_seq_length,
                                        decoder_max_length=args.first_decoder_max_length,
                                        batch_size=args.first_batch_size, shuffle=args.first_shuffle,
                                        teacher_forcing=args.first_teacher_forcing,
                                        predict_eos_token=args.first_predict_eos_token,
                                        truncate_method=args.first_truncate_method)

    test_dataloader = PromptDataLoader(dataset=LSR_dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                       tokenizer_wrapper_class=WrapperClass, max_seq_length=args.first_max_seq_length,
                                       decoder_max_length=args.first_decoder_max_length,
                                       batch_size=args.first_batch_size, shuffle=args.first_shuffle,
                                       teacher_forcing=args.first_teacher_forcing,
                                       predict_eos_token=args.first_predict_eos_token,
                                       truncate_method=args.first_truncate_method)

    validation_dataloader = PromptDataLoader(dataset=LSR_dataset["validation"], template=mytemplate,
                                             tokenizer=tokenizer,
                                             tokenizer_wrapper_class=WrapperClass,
                                             max_seq_length=args.first_max_seq_length,
                                             decoder_max_length=args.first_decoder_max_length,
                                             batch_size=args.first_batch_size, shuffle=args.first_shuffle,
                                             teacher_forcing=args.first_teacher_forcing,
                                             predict_eos_token=args.first_predict_eos_token,
                                             truncate_method=args.first_truncate_method)
    return train_dataloader, test_dataloader, validation_dataloader

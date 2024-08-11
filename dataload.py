from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer
from openprompt.prompts import SoftTemplate
from openprompt import PromptForClassification
import time
import os
import torch
from collections import defaultdict
import random
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
import pandas as pd
import numpy as np
import pickle
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from transformers import T5Tokenizer
from openprompt.prompts import MixedTemplate, ManualTemplate
import csv
import sys
from tqdm import tqdm
from openprompt.data_utils import PROCESSORS
import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer
from openprompt.prompts import SoftTemplate
from openprompt import PromptForClassification
import time
import os
from transformers import AutoTokenizer,AutoModelForMultipleChoice,AutoConfig,AutoModelWithLMHead

from openprompt.plms.seq2seq import T5LMTokenizerWrapper, T5TokenizerWrapper

def read_file_portions_based_ratio(fname, ratios=[0.8, 0.1, 0.1]):
    with open(f'{fname}.txt', 'r') as f:
        lines = f.readlines()


    user_data = defaultdict(list)
    for line in lines:
        u = line.rstrip().split(' ')[0]
        user_data[u].append(line)

    user_ids = list(user_data.keys())
    random.shuffle(user_ids)

    split1 = int(ratios[0] * len(user_ids))
    split2 = split1 + int(ratios[1] * len(user_ids))


    f1_lines = [line for uid in user_ids[:split1] for line in user_data[uid]]
    f2_lines = [line for uid in user_ids[split1:split2] for line in user_data[uid]]
    f3_lines = [line for uid in user_ids[split2:] for line in user_data[uid]]

    return f1_lines, f2_lines, f3_lines

f1, f2, f3 = read_file_portions_based_ratio('../../SASRec_pytorch/data/ori_index2title')


sasf1, sasf2, sasf3 = read_file_portions_based_ratio('../../SASRec_pytorch/data/saspre_index2title_10')



with open('sas_dataset1.pkl', 'rb') as f:
    sas_dataset1 = pickle.load(f)

with open('sas_dataset2.pkl', 'rb') as f:
    sas_dataset2 = pickle.load(f)

with open('sas_dataset3.pkl', 'rb') as f:
    sas_dataset3 = pickle.load(f)


def create_sas_dataset(org_dataset, dataset_name):
    user_seq, user_can, user_label, user_saspre = org_dataset
    dataset = []
    for user_id in user_seq.keys():

        dataset.append({
            'user_seq': user_seq[user_id],
            'user_can': user_can[user_id],
            'user_id': user_id,
            'user_label': user_label[user_id],
            'user_saspre':user_saspre[user_id]
        })
    return dataset

sas_dataset = {
    'train': create_sas_dataset(sas_dataset1, 'train'),
    'validation': create_sas_dataset(sas_dataset2, 'validation'),
    'test': create_sas_dataset(sas_dataset3, 'test')
}

def create_RPS_dataset(dataset, dataset_name):
    user_seq, user_can, user_label, model_name, SR_pre = dataset
    dataset = []
    for user_id in user_seq.keys():

        dataset.append({
            'user_seq': user_seq[user_id],
            'user_can': user_can[user_id],
            'user_id': user_id,
            'user_label': user_label[user_id],
            'model_name': model_name[user_id],
            'SR_pre':SR_pre[user_id]
        })
    return dataset

def create_TA_dataset(dataset, dataset_name):
    ICL, movie_m, movie_m_1, user_TA, movie_next, user_can, model_name,user_label = dataset
    proc_dataset = []
    for user_id in user_can.keys():
        proc_dataset.append({
            'ICL': ICL[user_id],
            'm': movie_m[user_id],
            'm_1': movie_m_1[user_id],
            'user_TA': user_TA[user_id],
            'next': movie_next[user_id],
            'user_can':user_can[user_id],
            'model_name':model_name[user_id],
            'user_id': user_id,
            'user_label':user_label[user_id]
        })
    return dataset

saspre_dataset = {}
for split in ['train', 'test', 'validation']:
    saspre_dataset[split] = []
    for data in sas_dataset[split]:

        user_seq = ', '.join(data['user_seq'])
        user_can = ', '.join(data['user_can'])
        user_saspre = ','.join(data['user_saspre'])
        if len(data['user_label']) > 0 and data['user_label'][0] is not None:
            input_example = InputExample(label=(data['user_label'][0] - 1), guid=int(data['user_id']),
                                         meta={'user_saspre': user_saspre, 'user_seq': user_seq, 'user_can': user_can})
            saspre_dataset[split].append(input_example)
sas_train_list = []
sas_test_list = []
sas_val_list = []
for i in saspre_dataset['train']:
    sas_train_list.append(i.guid)
for i in saspre_dataset['test']:
    sas_test_list.append(i.guid)
for i in saspre_dataset['validation']:
    sas_val_list.append(i.guid)


def read_file_portions_based_rows(fname, train_id, test_id, val_id):
    with open(f'{fname}.txt', 'r') as f:
        lines = f.readlines()
    user_data = defaultdict(list)
    for line in lines:
        u = line.rstrip().split(' ')[0]
        user_data[u].append(line)

    user_ids = list(user_data.keys())
    ud1 =[]
    ud2 = []
    ud3 = []
    for i in train_id:
        ud1.append(user_ids[i-1])
    for i in test_id:
        ud2.append(user_ids[i-1])
    for i in val_id:
        ud3.append(user_ids[i-1])

    f1_lines = [line for uid in ud1 for line in user_data[uid]]
    f2_lines = [line for uid in ud2 for line in user_data[uid]]
    f3_lines = [line for uid in ud3 for line in user_data[uid]]

    return f1_lines, f2_lines, f3_lines

def topk(input, k):
    sorted_values_list = []
    sorted_indices_list = []
    for i in range(input.shape[0]):
        sorted_values, sorted_indices = torch.sort(input[i], descending=True)
        sorted_values_list.append(sorted_values[:k])
        sorted_indices_list.append(sorted_indices[:k])
    return torch.stack(sorted_values_list), torch.stack(sorted_indices_list)


def topk2(input, k):
    sorted_indices = sorted(range(len(input)), key=lambda x: input[x], reverse=True)
    sorted_values = sorted(input, reverse=True)


    return sorted_values[:k], sorted_indices[:k]

import random
from collections import defaultdict
import pickle
import torch
from openprompt.data_utils.utils import InputExample


def read_file_portions_based_ratio(fname, ratios=None):
    if ratios is None:
        ratios = [0.8, 0.1, 0.1]

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


def create_sas_dataset(org_dataset, dataset_name):
    user_seq, user_can, user_label, user_saspre = org_dataset
    dataset = []
    for user_id in user_seq.keys():
        dataset.append({
            'user_seq': user_seq[user_id],
            'user_can': user_can[user_id],
            'user_id': user_id,
            'user_label': user_label[user_id],
            'user_saspre': user_saspre[user_id]
        })
    return dataset


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
            'SR_pre': SR_pre[user_id]
        })
    return dataset


def create_LSR_dataset(dataset, dataset_name):
    user_seq, user_can, user_label, model_name = dataset
    dataset = []
    for user_id in user_seq.keys():
        dataset.append({
            'user_seq': user_seq[user_id],
            'user_can': user_can[user_id],
            'user_id': user_id,
            'user_label': user_label[user_id],
            'model_name': model_name[user_id],
        })
    return dataset


def create_TA_dataset(dataset, dataset_name):
    ICL, movie_m, movie_m_1, user_TA, movie_next, user_can, model_name, user_label = dataset
    proc_dataset = []
    for user_id in user_can.keys():
        proc_dataset.append({
            'ICL': ICL[user_id],
            'm': movie_m[user_id],
            'm_1': movie_m_1[user_id],
            'user_TA': user_TA[user_id],
            'next': movie_next[user_id],
            'user_can': user_can[user_id],
            'model_name': model_name[user_id],
            'user_id': user_id,
            'user_label': user_label[user_id]
        })
    return dataset


def read_file_portions_based_rows(fname, train_id, test_id, val_id):
    with open(f'{fname}.txt', 'r') as f:
        lines = f.readlines()
    user_data = defaultdict(list)
    for line in lines:
        u = line.rstrip().split(' ')[0]
        user_data[u].append(line)

    user_ids = list(user_data.keys())
    ud1 = []
    ud2 = []
    ud3 = []
    for i in train_id:
        ud1.append(user_ids[i - 1])
    for i in test_id:
        ud2.append(user_ids[i - 1])
    for i in val_id:
        ud3.append(user_ids[i - 1])

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

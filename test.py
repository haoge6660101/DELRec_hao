import os
import torch
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

#to = AutoTokenizer.from_pretrained()

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
print(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("我们有", torch.cuda.device_count(), "个GPU!")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# def read_file_portions(fname, ratios=[0.7, 0.1, 0.2]):
#     with open(f'{fname}.txt', 'r') as f:
#         lines = f.readlines()
#
#     # 将数据按用户ID分组
#     user_data = defaultdict(list)
#     for line in lines:
#         u = line.rstrip().split(' ')[0]
#         user_data[u].append(line)
#
#     # 将用户ID打乱
#     user_ids = list(user_data.keys())
#     random.shuffle(user_ids)
#
#     # 根据比例划分用户ID
#     split1 = int(ratios[0] * len(user_ids))
#     split2 = split1 + int(ratios[1] * len(user_ids))
#
#     # 根据划分的用户ID，获取对应的数据
#     f1_lines = [line for uid in user_ids[:split1] for line in user_data[uid]]
#     f2_lines = [line for uid in user_ids[split1:split2] for line in user_data[uid]]
#     f3_lines = [line for uid in user_ids[split2:] for line in user_data[uid]]
#
#     return f1_lines, f2_lines, f3_lines

# f1, f2, f3 = read_file_portions('../../SASRec_pytorch/data/ori_index2title')
#
# print(len(f1))
# print(len(f2))
# print(len(f3))
#
# sasf1, sasf2, sasf3 = read_file_portions('../../SASRec_pytorch/data/saspre_index2title_10',ratios=[0.7,0.01,0.299])
#
# print(len(sasf1))
# print(len(sasf2))
# print(len(sasf3))
# print("Loading Datasets.......")
# # 加载数据集
# with open('sas_dataset1.pkl', 'rb') as f:
#     sas_dataset1 = pickle.load(f)
#
# with open('sas_dataset2.pkl', 'rb') as f:
#     sas_dataset2 = pickle.load(f)
#
# with open('sas_dataset3.pkl', 'rb') as f:
#     sas_dataset3 = pickle.load(f)
# print("Loading Datasets Success!")
#
#
# def create_sas_dataset(org_dataset, dataset_name):
#     user_seq, user_can, user_label, user_saspre = org_dataset
#     dataset = []
#     for user_id in user_seq.keys():
#         # 检查用户是否有标签
#         # if user_id not in user_label:
#         #     continue  # 如果没有，就跳过这个用户
#         dataset.append({
#             'user_seq': user_seq[user_id],
#             'user_can': user_can[user_id],
#             'user_id': user_id,
#             'user_label': user_label[user_id],
#             'user_saspre':user_saspre[user_id]
#         })
#     return dataset
#
# sas_dataset = {
#     'train': create_sas_dataset(sas_dataset1, 'train'),
#     'validation': create_sas_dataset(sas_dataset2, 'validation'),
#     'test': create_sas_dataset(sas_dataset3, 'test')
# }
# print(len(sas_dataset['train']))
# print(' ')
# print(len(sas_dataset['test']))
# print(' ')
# print(len(sas_dataset['validation']))
#
#
#
# saspre_dataset = {}
# for split in ['train', 'test', 'validation']:
#     saspre_dataset[split] = []
#     for data in sas_dataset[split]:
#         # 将列表转换为字符串a
#
#         user_seq = ', '.join(data['user_seq'])
#         user_can = ', '.join(data['user_can'])
#         user_saspre = ','.join(data['user_saspre'])
#         if len(data['user_label']) > 0 and data['user_label'][0] is not None:
#             input_example = InputExample(label=(data['user_label'][0] - 1), guid=int(data['user_id']),
#                                          meta={'user_saspre': user_saspre, 'user_seq': user_seq, 'user_can': user_can})
#             saspre_dataset[split].append(input_example)
# sas_train_list = []
# sas_test_list = []
# sas_val_list = []
# for i in saspre_dataset['train']:
#     sas_train_list.append(i.guid)
# for i in saspre_dataset['test']:
#     sas_test_list.append(i.guid)
# for i in saspre_dataset['validation']:
#     sas_val_list.append(i.guid)
#
# print("1:",len(sas_train_list))
# print("2:",len(sas_test_list))
# print("3:",len(sas_val_list))
# #sys.exit()
# def read_file_portions(fname, train_id, test_id, val_id):
#     with open(f'{fname}.txt', 'r') as f:
#         lines = f.readlines()
#     #train_id = []
#     # 将数据按用户ID分组
#     user_data = defaultdict(list)
#     for line in lines:
#         u = line.rstrip().split(' ')[0]
#         user_data[u].append(line)
#     #print(user_data[11511])
#     user_ids = list(user_data.keys())
#     ud1 =[]
#     ud2=[]
#     ud3 = []
#     #print(train_id[0])
#     #ii = train_id[0]
#     #print(user_ids[ii-1])
#     for i in train_id:
#         ud1.append(user_ids[i-1])
#     for i in test_id:
#         ud2.append(user_ids[i-1])
#     for i in val_id:
#         ud3.append(user_ids[i-1])
#     #print(len(ud1),ud1[:5])
#
#     # 根据划分的用户ID，获取对应的数据
#     f1_lines = [line for uid in ud1 for line in user_data[uid]]
#     f2_lines = [line for uid in ud2 for line in user_data[uid]]
#     f3_lines = [line for uid in ud3 for line in user_data[uid]]
#
#     return f1_lines, f2_lines, f3_lines
# # f1, f2, f3 = read_file_portions('../../SASRec_pytorch/data/ori_index2title')
#
# # print(len(f1))
# # print(len(f2))
# # print(len(f3))
#
# sasf1, sasf2, sasf3 = read_file_portions('../../SASRec_pytorch/data/saspre_index2title_10',sas_train_list,sas_test_list,sas_val_list)
#
# print(len(sasf1))
# print(len(sasf2))
# print(len(sasf3))

load_sas = True
# 读取movie.csv文件
if load_sas == False:
    print("Distributing Datasets.......")
    # df = pd.read_csv('../../transform/TFRS/smovies.csv')
    # movies = df.set_index('Title')['MovieID'].to_dict()
    #
    #
    # # train/val/test data generation
    # def sas_data_partition(ff):
    #     usernum = 0
    #     itemnum = 0
    #     User = defaultdict(list)
    #     user_seq = {}
    #     user_can = {}
    #     user_label = {}
    #     user_saspre = {}
    #     # assume user/item index starting from 1
    #     # f = open('%s.txt' % fname, 'r')
    #     for line in ff:
    #         u = line.rstrip().split(' ')[0]
    #         i = ' '.join(line.rstrip().split(' ')[1:])
    #         u = int(u)
    #         i = str(i)
    #         usernum = max(u, usernum)
    #         # itemnum = max(i, itemnum)
    #         User[u].append(i)
    #
    #     for user in tqdm(User):
    #         if len(User[user]) < 10:
    #             user_seq[user] = User[user]
    #             user_can[user] = []
    #             user_label[user] = []
    #             user_saspre[user] = []
    #         else:
    #             user_saspre[user] = []
    #             user_saspre[user] = User[user][-9:]
    #             user_seq[user] = User[user][:-10]
    #             # user_valid[user].append(User[user][-2])
    #             user_can[user] = []
    #             it = User[user][-10]
    #             # print(it)
    #             user_can[user].append(it)
    #             user_label[user] = []
    #             user_label[user].append(movies[it])
    #             print(it)
    #             print(movies[it])
    #             if movies[it] > 3883:
    #                 print("ops!")
    #             # print(user_pre[user][0])
    #
    #             for ii in User[user][-9:]:
    #                 user_can[user].append(ii)
    #             for _ in range(10):
    #                 movie = np.random.choice(list(movies.keys()))
    #                 while movie in user_can[user] and movie in User[user]:
    #                     movie = np.random.choice(list(movies.keys()))
    #                 user_can[user].append(movie)
    #             np.random.shuffle(user_can[user])
    #     return [user_seq, user_can, user_label, user_saspre]
    #
    #
    # sas_dataset1 = sas_data_partition(sasf1)
    # sas_dataset2 = sas_data_partition(sasf2)
    # sas_dataset3 = sas_data_partition(sasf3)
    # [org_user_seq1,  org_user_pre1,] = org_dataset1
    # [org_user_seq2,  org_user_pre2,] = org_dataset2
    # [org_user_seq3,  org_user_pre3] = org_dataset3
    # print(len(org_user_seq1))
    # print(len(org_user_seq2))
    # print(len(org_user_seq3))

    # 保存数据集
    # with open('./sas_dataset1.pkl', 'wb') as f:
    #     pickle.dump(sas_dataset1, f)
    #     print("save suc")
    #
    # with open('./sas_dataset2.pkl', 'wb') as f:
    #     pickle.dump(sas_dataset2, f)
    #     print("save suc")
    #
    # with open('./sas_dataset3.pkl', 'wb') as f:
    #     pickle.dump(sas_dataset3, f)
    #     print("save suc")
    # print("Save Success!")

else:
    print("Loading Datasets.......")
    # # 加载数据集
    with open('sas_dataset1.pkl', 'rb') as f:
        sas_dataset1 = pickle.load(f)

    with open('sas_dataset2.pkl', 'rb') as f:
        sas_dataset2 = pickle.load(f)

    with open('sas_dataset3.pkl', 'rb') as f:
        sas_dataset3 = pickle.load(f)
    print("hhhh Loading Datasets Success!")

print("Loading Datasets.......")
#加载数据集
with open('tori_dataset1.pkl', 'rb') as f:
    ori_dataset1 = pickle.load(f)

with open('tori_dataset3.pkl', 'rb') as f:
    ori_dataset2 = pickle.load(f)

with open('tori_dataset2.pkl', 'rb') as f:
    ori_dataset3 = pickle.load(f)
print("Loading Datasets Success!")
def create_sas_dataset(org_dataset, dataset_name):
    user_seq, user_can, user_label, user_saspre = org_dataset
    dataset = []
    for user_id in user_seq.keys():
        # 检查用户是否有标签
        # if user_id not in user_label:
        #     continue  # 如果没有，就跳过这个用户
        dataset.append({
            'user_seq': user_seq[user_id],
            'user_can': user_can[user_id],
            'user_id': user_id,
            'user_label': user_label[user_id],
            'user_saspre':user_saspre[user_id]
        })
    return dataset
sas_list = []
ori_list = []
sas_dataset = {
    'train': create_sas_dataset(sas_dataset1, 'train'),
    'validation': create_sas_dataset(sas_dataset2, 'validation'),
    'test': create_sas_dataset(sas_dataset3, 'test')
}

ori_dataset = {
    'train': create_sas_dataset(ori_dataset1, 'train'),
    'validation': create_sas_dataset(ori_dataset2, 'validation'),
    'test': create_sas_dataset(ori_dataset3, 'test')
}

print(sas_dataset['train'][0])
print(' ')
print(sas_dataset['test'][0])
print(' ')
print(sas_dataset['validation'][0])

import re
def check_suffix(s):
    # 正则表达式匹配模式
    pattern = r",\s(The|A|An)\s\(\d{4}\)$"
    # 使用search方法检查字符串是否符合模式
    if re.search(pattern, s):
        return True
    else:
        return False
def rearrange_string(s):
    # 正则表达式匹配模式
    pattern = r"(.*),\s(The|A|An)\s\((\d{4})\)"
    # 使用sub方法替换和重新排列字符串
    return re.sub(pattern, r"\2 \1 (\3)", s)
# 读取文件内容
with open('new_saspre_classes.txt', 'r') as file:
    lines = file.readlines()

saspre_dataset = {}
DUDU = {}
caca = {}
for split in ['train', 'test', 'validation']:
    saspre_dataset[split] = []
    DUDU[split] = []
    caca[split] = []
    for data in sas_dataset[split]:
        # 将列表转换为字符串a

        for index, i in enumerate(data['user_seq']):
            if check_suffix(i):
                data['user_seq'][index] = rearrange_string(i)
        for index, i in enumerate(data['user_can']):
            if check_suffix(i):
                data['user_can'][index] = rearrange_string(i)
        for index, i in enumerate(data['user_saspre']):
            if check_suffix(i):
                data['user_saspre'][index] = rearrange_string(i)


        user_seq = ', '.join(data['user_seq'])
        user_can = ', '.join(data['user_can'])
        user_saspre = ','.join(data['user_saspre'])
        if len(data['user_label']) > 0: #and data['user_label'][0] is not None:
            input_example = InputExample(label=(data['user_label'][0] - 1), guid=int(data['user_id']),
                                         meta={'user_saspre': user_saspre, 'user_seq': user_seq, 'user_can': user_can})
            saspre_dataset[split].append(input_example)
            DUDU[split].append(user_seq)
            caca[split].append(int(data['user_id']))


#ori_tes_dataset = {}
DUDU2 = {}
caca2 = {}
for split in ['train', 'test', 'validation']:
    #ori_tes_dataset[split] = []
    DUDU2[split] = []
    caca2[split] = []
    for data in ori_dataset[split]:
        # 将列表转换为字符串a

        user_seq = ', '.join(data['user_seq'])
        user_can = ', '.join(data['user_can'])
        #user_saspre = ','.join(data['user_saspre'])
        if len(data['user_label']) > 0: #and data['user_label'][0] is not None:
           # aa += 1
            input_example = InputExample(label=(data['user_label'][0]-1), guid=int(data['user_id']),
                                         meta={'user_saspre': "",'user_seq': user_seq, 'user_can': user_can})
            #ori_tes_dataset[split].append(input_example)
            DUDU2[split].append(user_seq)
            caca2[split].append(int(data['user_id']))
print("sas")
print(saspre_dataset["train"][:3])
print(saspre_dataset["test"][:3])
print(saspre_dataset["validation"][:3])
print(len(saspre_dataset["train"]))
print(len(saspre_dataset["test"]))
print(len(saspre_dataset["validation"]))
print("ori")
#sys.exit()
#print(ori_tes_dataset["train"][:3])
#print(ori_tes_dataset["test"][:3])
#print(ori_tes_dataset["validation"][:3])
#sys.exit()
# for i in DUDU["train"]:
#     sas_list.append(i)
# for i in DUDU2["train"]:
#     ori_list.append(i)
#     print("sota")
# print(len(saspre_dataset["train"]))
# print(len(ori_tes_dataset["train"]))
# print(len(sas_list))
# print(len(ori_list))
# aoa = []
# for i in tqdm(sas_list):
#     for j in ori_list:
#         if i == j:
#             aoa.append(i)
# print(len(aoa))
# sys.exit()
# for i in saspre_dataset['train']:
#     sas_list.append(i.guid)
# for i in ori_tes_dataset['validation']:
#     ori_list.append(i.guid)
# print(len(sas_list))
# print(len(ori_list))
# aoa = []
# for i in tqdm(sas_list):
#     for j in ori_list:
#         if i == j:
#             aoa.append(i)
# print(len(aoa))

# print(saspre_dataset['train'].guid)
# print(saspre_dataset['test'][:3])
# print(saspre_dataset['validation'][:3])
#sys.exit()


# for i in gong:
#     print(i)
# print(len(ori_dataset['train']))
# print(' ')
# print(len(ori_dataset['test']))
# print(' ')
# print(len(ori_dataset['validation']))

# model_path = '../../gemma-2b-it'
# model_dir = model_path
# model_config = GemmaConfig.from_pretrained(model_dir)
# plm = GemmaForCausalLM.from_pretrained(model_dir,config=model_config)
# tokenizer = GemmaTokenizer.from_pretrained(model_dir)
# WrapperClass = LMTokenizerWrapper
# plm, tokenizer = add_special_tokens(plm, tokenizer, specials_to_add=["<pad>"])

plm, tokenizer, model_config, WrapperClass = load_plm("t5","../../flan-t5-xl")

#mytemplate1 = MixedTemplate(model=plm, tokenizer=tokenizer, text='Use the recommendation mode of the SASRec model to predict next movie that the user might watch from candidate set, recommendation mode of the SASRec model is:{"soft":None,"duplicate":400}, movie viewing record of the user is:{"meta":"user_seq"}, candidate set is:{"meta":"user_can"}, next movie the user will watch might be:{"mask"},{"meta":"user_saspre"}.')

mytemplate2 = MixedTemplate(model=plm, tokenizer=tokenizer, text='Please simulate the recommendation pattern of the SASRec model to predict next movie that the user might watch from candidate set, the information and recommendation pattern of the SASRec model is:{"soft":None,"duplicate":500}, movie viewing history of the user is:{"meta":"user_seq"}, candidate set is:{"meta":"user_can"}, the next movie the user will watch (predicted by SASRec model) might be:{"mask"},{"meta":"user_saspre"}.')

plm.parallelize()


with open('saspre_classes.txt', 'r') as f:
    lines = f.readlines()

# 将每一行转换为一个列表元素
cla = [line.strip()for line in lines]
saspre_label = {item: [item, item[:-7]] for item in cla}
#print(saspre_label)
myverbalizer = ManualVerbalizer(tokenizer=tokenizer,classes=cla,label_words=saspre_label)#.from_file(f"saspre_classes.txt")

from torch import nn
load_model = True
prompt_model = PromptForClassification(plm=plm,template=mytemplate2, verbalizer=myverbalizer, freeze_plm=True)
params_before = {name: p.clone().cpu() for name, p in prompt_model.named_parameters()}
if load_model == True:
     #prompt_model.load_state_dict(torch.load("./ht1_184.0000_ht10_804.0000_nd1_0.000000_nd10_0.000000_saspre4.ckpt"))

     prompt_model.load_state_dict(torch.load("./last/500_ht1_6846.0000_ht10_16128.0000_epoch_64_lr0.007_wd5e-08_t5xl.ckpt"))
     print("load")
     print("./7e-3_last_5e-8/500_ht1_6846.0000_ht10_16128.0000_epoch_64_lr0.007_wd5e-08_t5xl.ckpt")

     print("Load  Model Success!")
params_after = {name: p.clone().cpu() for name, p in prompt_model.named_parameters()}
aaa = 0
# 计算参数的变化，并打印出发生变化的参数
for name in params_before.keys():
 diff = params_after[name] - params_before[name]
 # if 'lora' in name:
 #     print(f"{name}diff:",diff)
 if torch.sum(diff).item() != 0:  # 如果参数发生了变化
     # print(diff)
     print(f"{name} has changed during training.")
     # print(pd1.template.soft_embedding.weight)
     # pvar = pd.template.soft_embedding.weight
     aaa = aaa + 1

print(aaa)
prompt_model = prompt_model.cuda()

prompt_model.parallelize()
#prompt_model.train()

from bigmodelvis import Visualization
Visualization(prompt_model).structure_graph()
#sys.exit()
# sas_list = []
# sas_list2 = []+
# ori_list = []
# ori_list2 = []
train_dataloader = PromptDataLoader(dataset=saspre_dataset["train"], template=mytemplate2, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=1065, decoder_max_length=10,
    batch_size=7,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")
# for i,j in zip(DUDU["validation"],caca['validation']):
#     sas_list.append(i)
#     sas_list2.append(j)
# for i,j in zip(DUDU2["validation"],caca2["validation"]):
#     ori_list.append(i)
#     ori_list2.append(j)
# print(len(saspre_dataset["validation"]))
# print(len(ori_tes_dataset["validation"]))
# print(len(sas_list))
# print(len(ori_list))
# aoa = []
# aoa2 = []
# for i,o in tqdm(zip(sas_list,sas_list2)):
#     for j,k in zip(ori_list,ori_list2):
#         if i == j and o == k:
#             aoa.append(i)
#             aoa2.append(o)
#             #print(o,' ',i)
# print(len(aoa))
# print(len(aoa2))

# print(saspre_dataset['train'].guid)
# print(saspre_dataset['test'][:3])
# print(saspre_dataset['validation'][:3])
# test_dataloader = PromptDataLoader(dataset=saspre_dataset["test"], template=mytemplate1, tokenizer=tokenizer,
#     tokenizer_wrapper_class=WrapperClass, max_seq_length=1000, decoder_max_length=5,
#     batch_size=15,shuffle=True, teacher_forcing=False, predict_eos_token=False,
#     truncate_method="tail")

# validation_dataloader = PromptDataLoader(dataset=saspre_dataset["validation"], template=mytemplate1, tokenizer=tokenizer,
#     tokenizer_wrapper_class=WrapperClass, max_seq_length=1000, decoder_max_length=5,
#     batch_size=15,shuffle=True, teacher_forcing=False, predict_eos_token=False,
#     truncate_method="tail")


def topk(input, k):
    sorted_values_list = []
    sorted_indices_list = []
    for i in range(input.shape[0]):
        sorted_values, sorted_indices = torch.sort(input[i], descending=True)
        sorted_values_list.append(sorted_values[:k])
        sorted_indices_list.append(sorted_indices[:k])
    return torch.stack(sorted_values_list), torch.stack(sorted_indices_list)


def topk2(input, k):
    # 对输入列表进行排序，返回排序后的元素及其对应的索引
    sorted_indices = sorted(range(len(input)), key=lambda x: input[x], reverse=True)
    sorted_values = sorted(input, reverse=True)

    # 返回最大的k个元素及其对应的索引
    return sorted_values[:k], sorted_indices[:k]





use_cuda =True
def evaluate(prompt_model, dataloader, desc):
    prompt_model.eval()
    allpreds = []
    alllabels = []

    for step, inputs in enumerate(dataloader):
        if use_cuda:
            inputs = inputs#.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    return acc



loss_func = torch.nn.CrossEntropyLoss()
optimizer1 = None
scheduler1 = None
# Now the training is standard
from transformers import  AdamW, get_linear_schedule_with_warmup,Adafactor
from lion_pytorch import Lion
from bitsandbytes.optim import PagedLion,PagedLion8bit,Lion8bit
optimizer_grouped_parameters2 = [{'params': [p for name, p in prompt_model.template.named_parameters() if 'raw_embedding' not in name]}] # note that you have to remove the raw_embedding manually from the optimization
tot_step = 400
lr = 7e-4
we_de = 5e-8
#500_ht1_6846.0000_ht10_16128.0000_epoch_64_lr0.007_wd5e-08_t5xl
#optimizer2 = Adafactor(optimizer_grouped_parameters2,lr=1e-3,relative_step=False)
optimizer2 = PagedLion8bit(optimizer_grouped_parameters2, lr=lr,weight_decay=we_de) # usually lr = 0.5
#scheduler2 = get_linear_schedule_with_warmup(optimizer2,num_warmup_steps=200, num_training_steps=400) # usually num_warmup_steps is 500
print("8-it ",lr,' ',we_de,"paged")
#23.44 52 59
best_ht1 = 0.00000
best_ht10 = 0.00000
best_nd1 = 0.00000
best_nd10 = 0.00000
ht1 = 0.00000
ht10 = 0.00000
nd1 = 0.00000
nd10 = 0.00000
gradient_accumulation_steps = 1
prompt_model.train()
params_before = {name: p.clone().cpu() for name, p in prompt_model.named_parameters()}
#torch.cuda.empty_cache()
for epoch in tqdm(range(400)):
    tot_loss = 0
    ht1 = 0.00000
    ht10 = 0.00000
    nd1 = 0.0000000
    nd10 = 0.0000000

    #torch.cuda.empty_cache()
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        #logits2 = logits
        labels = inputs['label']
        probabilities = F.softmax(logits, dim=-1)
        # probabilities2 = F.softmax(logits2, dim=-1)
        #_, topk_indices = topk(probabilities, 10)
        # print("Top 10 indices:", topk_indices)
        # for i in range(15):
        #     # 获取前 ss[i] 个最大概率的索引
        #     _, topk_indices = topk2(probabilities[i], 10)
        #     _, topk_indices2 = topk2(probabilities2[i], 1)
        #     # 获取排名第 ss[i] 的概率的索引
        #     index = topk_indices[-1]
        #     ind = topk_indices2[-1]
        #
        #     print(f"在第 {i + 1} 个 probabilities 中，排名第 {ind} 的概率的索引是：{index}")
        #     print(topk_indices2)
        #     print(topk_indices)
        # 计算每个标签的概率在所有概率中的排名
        sorted_indices = torch.argsort(probabilities, descending=True)
        rankss = (sorted_indices == labels.unsqueeze(-1)).nonzero()[:, -1] + 1
        for j in range(0,len(rankss)):
            i =rankss[j]
            #print(i.cpu())
            if i.cpu() <=  10:
                nd10 =nd10#+= 1 / np.log2(i.cpu() + 2)
                ht10 += 1
            if i.cpu() <= 1:
                nd1 = nd1#+= 1 / np.log2(i.cpu() + 2)
                ht1 += 1
        # print("Rank of each label:", rankss)
        # print("Label:",labels)
        # for i in range(len(rankss)):
        #     # 获取前 ss[i] 个最大概率的索引
        #     _, topk_indices = topk2(probabilities[i], rankss[i])
        #     # 获取排名第 ss[i] 的概率的索引
        #     index = topk_indices[-1]
            #print(f"在第 {i + 1} 个 probabilities 中，排名第 {rankss[i]} 的概率的索引是：{index}")
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer2.step()
        optimizer2.zero_grad()
        #scheduler2.step()
        if step %100 ==1:
            print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)
        if step % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 2.0)
    if ht10 > best_ht10 or ht1 >best_ht1:
        best_ht10 = ht10
        best_ht1 = ht1
        best_nd1 = nd1
        best_nd10 = nd10
        torch.save(prompt_model.state_dict(), f"./last/500_ht1_{ht1:.4f}_ht10_{ht10:.4f}_epoch_{epoch+65}_lr{lr}_wd{we_de}_t5xl.ckpt")
        print("Save Success!")
    print(f"/ht1_{ht1:.4f}_ht10_{ht10:.4f}_nd1_{nd1:.4f}_nd10_{nd10:.4f}")

params_after = {name: p.clone().cpu() for name, p in prompt_model.named_parameters()}

aaa = 0
# 计算参数的变化，并打印出发生变化的参数
for name in params_before.keys():
    diff = params_after[name] - params_before[name]
    if torch.sum(diff).item() != 0:  # 如果参数发生了变化
        print(f"{name} has changed during training.")
        #print(pd1.template.soft_embedding.weight)
        #pvar = pd.template.soft_embedding.weight
        aaa = aaa+1

print(aaa)
sys.exit()

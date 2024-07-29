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
from bigmodelvis import Visualization
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
from torch import nn
import os
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig,AdaLoraConfig
from transformers import AutoTokenizer,AutoModelForMultipleChoice,AutoConfig,AutoModelForCausalLM
from openprompt.plms.seq2seq import T5LMTokenizerWrapper, T5TokenizerWrapper
from openprompt.plms.lm import LMTokenizerWrapper
from bigmodelvis import Visualization


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
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
#     #print(user_data[:10])
#
#     # 将用户ID打乱
#     user_ids = list(user_data.keys())
#     #random.shuffle(user_ids)
#
#     # 根据比例划分用户ID
#     split1 = int(ratios[0] * len(user_ids))
#     split2 = split1 + int(ratios[1] * len(user_ids))
#     for uid in user_ids[:split1]:
#         print(uid)
#
#
#     # 根据划分的用户ID，获取对应的数据
#     f1_lines = [line for uid in user_ids[:split1] for line in user_data[uid]]
#     f2_lines = [line for uid in user_ids[split1:split2] for line in user_data[uid]]
#     f3_lines = [line for uid in user_ids[split2:] for line in user_data[uid]]
#
#     return f1_lines, f2_lines, f3_lines
print("Loading Datasets.......")
# 加载数据集
with open('sas_dataset1.pkl', 'rb') as f:
    sas_dataset1 = pickle.load(f)

with open('sas_dataset2.pkl', 'rb') as f:
    sas_dataset2 = pickle.load(f)

with open('sas_dataset3.pkl', 'rb') as f:
    sas_dataset3 = pickle.load(f)
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

sas_dataset = {
    'train': create_sas_dataset(sas_dataset1, 'train'),
    'validation': create_sas_dataset(sas_dataset2, 'validation'),
    'test': create_sas_dataset(sas_dataset3, 'test')
}
print(len(sas_dataset['train']))
print(' ')
print(len(sas_dataset['test']))
print(' ')
print(len(sas_dataset['validation']))



saspre_dataset = {}
for split in ['train', 'test', 'validation']:
    saspre_dataset[split] = []
    for data in sas_dataset[split]:
        # 将列表转换为字符串a

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

print("1:",len(sas_train_list))
print("2:",len(sas_test_list))
print("3:",len(sas_val_list))
#sys.exit()
def read_file_portions(fname, train_id, test_id, val_id):
    with open(f'{fname}.txt', 'r') as f:
        lines = f.readlines()
    #train_id = []
    # 将数据按用户ID分组
    user_data = defaultdict(list)
    for line in lines:
        u = line.rstrip().split(' ')[0]
        user_data[u].append(line)
    #print(user_data[11511])
    user_ids = list(user_data.keys())
    ud1 =[]
    ud2=[]
    ud3 = []
    #print(train_id[0])
    #ii = train_id[0]
    #print(user_ids[ii-1])
    for i in train_id:
        ud1.append(user_ids[i-1])
    for i in test_id:
        ud2.append(user_ids[i-1])
    for i in val_id:
        ud3.append(user_ids[i-1])
    #print(len(ud1),ud1[:5])

    # 根据划分的用户ID，获取对应的数据
    f1_lines = [line for uid in ud1 for line in user_data[uid]]
    f2_lines = [line for uid in ud2 for line in user_data[uid]]
    f3_lines = [line for uid in ud3 for line in user_data[uid]]

    return f1_lines, f2_lines, f3_lines
# f1, f2, f3 = read_file_portions('../../SASRec_pytorch/data/ori_index2title')

# print(len(f1))
# print(len(f2))
# print(len(f3))

orif1, orif2, orif3 = read_file_portions('../../SASRec_pytorch/data/ori_index2title',sas_train_list,sas_test_list,sas_val_list)

print(len(orif1))
print(len(orif2))
print(len(orif3))
#sys.exit()
load_sas = True
df = pd.read_csv('../../transform/TFRS/smovies.csv')
movies = df.set_index('Title')['MovieID'].to_dict()
# 读取movie.csv文件
if load_sas == False:
    print("Distributing Datasets.......")
    df = pd.read_csv('../../transform/TFRS/smovies.csv')
    movies = df.set_index('Title')['MovieID'].to_dict()


    # train/val/test data generation
    def ori_data_partition(ff):
        usernum = 0
        itemnum = 0
        User = defaultdict(list)
        user_seq = {}
        user_can = {}
        user_label = {}
        user_saspre = {}
        user_label_name = {}
        user_idcan = {}
        # assume user/item index starting from 1
        # f = open('%s.txt' % fname, 'r')
        for line in ff:
            u = line.rstrip().split(' ')[0]
            i = ' '.join(line.rstrip().split(' ')[1:])
            u = int(u)
            i = str(i)
            usernum = max(u, usernum)
            # itemnum = max(i, itemnum)
            User[u].append(i)

        for user in tqdm(User):
            if len(User[user]) < 10:
                user_seq[user] = User[user]
                user_can[user] = []
                user_label[user] = []
                user_saspre[user] = []
                user_label_name[user] = []
                user_idcan[user] = []
            else:
                user_idcan[user] = []
                user_label_name[user] = []
                user_saspre[user] = [-1]
                #user_saspre[user] = User[user][-9:]
                user_seq[user] = User[user][:-1]
                # user_valid[user].append(User[user][-2])
                user_can[user] = []
                it = User[user][-1]
                # print(it)
                user_can[user].append(it)
                user_label[user] = []
                user_label[user].append(movies[it])
                user_idcan[user].append(movies[it])
                user_label_name[user].append(it)

                print("it",it)
                print("movies_it",movies[it])
                if movies[it] > 3883:
                    print("ops!")
                # print(user_pre[user][0])
                for _ in range(4):
                    movie = np.random.choice(list(movies.keys()))
                    while movie in User[user]:
                        movie = np.random.choice(list(movies.keys()))
                    user_can[user].append(movie)
                    user_idcan[user].append(movies[movie])
                # for ii in User[user][-9:]:
                #     user_can[user].append(ii)
                np.random.shuffle(user_can[user])
        return [user_seq, user_can, user_label,user_saspre,user_idcan]


    ori_dataset1 = ori_data_partition(orif1)
    ori_dataset2 = ori_data_partition(orif2)
    ori_dataset3 = ori_data_partition(orif3)
    # [org_user_seq1,  org_user_pre1,] = org_dataset1
    # [org_user_seq2,  org_user_pre2,] = org_dataset2
    # [org_user_seq3,  org_user_pre3] = org_dataset3
    # print(len(org_user_seq1))
    # print(len(org_user_seq2))
    # print(len(org_user_seq3))

    # 保存数据集
    with open('./5_tori_dataset1.pkl', 'wb') as f:
        pickle.dump(ori_dataset1, f)

    with open('./5_tori_dataset2.pkl', 'wb') as f:
        pickle.dump(ori_dataset3, f)

    with open('./5_tori_dataset3.pkl', 'wb') as f:
        pickle.dump(ori_dataset2, f)
    print("Loading  5 Datasets Success!")
else:
    print("Loading Datasets.......")
    # 加载数据集
    with open('5_tori_dataset1.pkl', 'rb') as f:
        ori_dataset1 = pickle.load(f)

    with open('5_tori_dataset3.pkl', 'rb') as f:
        ori_dataset3 = pickle.load(f)

    with open('5_tori_dataset2.pkl', 'rb') as f:
        ori_dataset2 = pickle.load(f)
    print("Loading Datasets Success!")

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
# with open('new_saspre_classes.txt', 'r') as file:
#     lines = file.readlines()


def create_sas_dataset(org_dataset, dataset_name):
    user_seq, user_can, user_label,user_saspre,user_idcan = org_dataset
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
            #'user_label_name': user_label_name[user_id],
            'user_saspre':user_saspre[user_id],
            'user_idcan':user_idcan[user_id]
        })
    return dataset
sas_list = []
ori_list = []
ori_dataset = {
    'train': create_sas_dataset(ori_dataset1, 'train'),
    'validation': create_sas_dataset(ori_dataset2, 'validation'),
    'test': create_sas_dataset(ori_dataset3, 'test')
}
print(len(ori_dataset['train']))
print(' ')
print(len(ori_dataset['test']))
print(' ')
print(len(ori_dataset['validation']))

#sys.exit()
aa = 0
bb = 0
ori_tes_dataset = {}

for split in ['train', 'test', 'validation']:
    ori_tes_dataset[split] = []
    aa=0
    for data in ori_dataset[split]:
        # 将列表转换为字符串a
        for index, i in enumerate(data['user_seq']):
            if check_suffix(i):
                data['user_seq'][index] = rearrange_string(i)
        for index, i in enumerate(data['user_can']):
            if check_suffix(i):
                data['user_can'][index] = rearrange_string(i)
        user_idcan = []
        for i in data['user_idcan']:
            user_idcan.append(i-1)
        user_seq = ', '.join(data['user_seq'])
        user_can = ', '.join(data['user_can'])
        #user_saspre = ','.join(data['user_saspre'])
        if len(data['user_label']) > 0: #and data['user_label'][0] is not None:
           # aa += 1
            input_example = InputExample(label=(data['user_label'][0]-1), guid=user_idcan,
                                         meta={'user_saspre': "",'user_seq': user_seq, 'user_can': user_can,'user_idcan':user_idcan})
            # print("len:",data['user_idcan'])
            # if len(data['user_idcan']) != 5:
            #        print("yes")
            #        aa += 1
            ori_tes_dataset[split].append(input_example)
    #     bb+=1
    # print("aa:",aa)
    # print("bb:",bb)

print(ori_tes_dataset['train'][:3])
print(ori_tes_dataset['test'][:3])
print(ori_tes_dataset['validation'][:3])
print(len(ori_tes_dataset['train']))
print(len(ori_tes_dataset['test']))
print(len(ori_tes_dataset['validation']))
# aa = 0
# bb =0
# for i in ori_tes_dataset['validation']:
#     print(i.meta['user_idcan'])
#     if len(i.meta['user_idcan']) < 5:
#         aa+=1
#     bb+=1
# print(aa)
# print(bb)
# sys.exit()
# for i in ori_tes_dataset['train']:
#     sas_list.append(i.guid)
# for i in ori_tes_dataset['test']:
#     ori_list.append(i.guid)
# print(len(sas_list))
# print(len(ori_list))
# print(sas_list[:10])
# print(ori_list[:10])
# aa2 = []
# for i in tqdm(sas_test_list):
#     for j in ori_list:
#         if i == j:
#             aa2.append(i)
# print("len:",len(aa2))
# sys.exit()
#print(aa)
#print(bb)
# print(ori_tes_dataset['train'][-2:])
#print(ori_tes_dataset['test'][-3:])
# print(ori_tes_dataset['validation'][-2:])
# lpl = 0
# for i in ori_tes_dataset['test']:
#     print( i.guid)
#     lpl += 1
# print(lpl)
from bigmodelvis import Visualization
#sys.exit()
# model_path = '../../gemma-2b-it'
# model_dir = model_path
# model_config = GemmaConfig.from_pretrained(model_dir)
# plm = GemmaForCausalLM.from_pretrained(model_dir,config=model_config)
# tokenizer = GemmaTokenizer.from_pretrained(model_dir)
# WrapperClass = LMTokenizerWrapper
# plm, tokenizer = add_special_tokens(plm, tokenizer, specials_to_add=["<pad>"])
model_dir = "../../flan-t5-xl"
plm, tokenizer, model_config, WrapperClass = load_plm("t5",model_dir)
#model_config = AutoConfig.from_pretrained("../../glm-2b",trust_remote_code=True)
#tokenizer = AutoTokenizer.from_pretrained("../../glm-2b",trust_remote_code=True)
#plm = AutoModelForMultipleChoice.from_pretrained("../../glm-2b",trust_remote_code=True)
#WrapperClass = T5LMTokenizerWrapper

Visualization(plm).structure_graph()
text_tem ='Please refer to the recommendation pattern of the SASRec model to predict next movie that the user might watch from candidate set, the information and recommendation pattern of the SASRec model is:{"soft":None,"duplicate":500}, movie viewing history of the user is:{"meta":"user_seq"}, candidate set is:{"meta":"user_can"}, the next movie the user will watch might be:{"mask"}.'
text_tem2_2='Please refer to the recommendation pattern of the SASRec model to predict next movie that the user might watch from candidate set, the information and recommendation pattern of the SASRec model is:{"soft":None,"duplicate":400}, movie viewing history of the user is:{"meta":"user_seq"}, candidate set is:{"meta":"user_can"}, the next movie the user will watch might be:{"mask"}.'

#text_tem2 ='Please predict next movie that the user might watch from candidate set, movie viewing record of the user is:{"meta":"user_seq"}, candidate set is:{"meta":"user_can"}, the next movie the user will watch might be:{"mask"}{"meta":"user_saspre"}.'
text_tem3 = 'Please simulate the recommendation pattern of the SASRec model to predict next movie that the user might watch from candidate set, the information and recommendation pattern of the SASRec model is:{"soft":None,"duplicate":475}, movie viewing history of the user is:{"meta":"user_seq"}, candidate set is:{"meta":"user_can"}, the next movie the user will watch (predicted by SASRec model) might be:{"mask"},{"meta":"user_saspre"}.'
#google-bert/bert-large-uncased
mytemplate1 = MixedTemplate(model=plm, tokenizer=tokenizer, text=text_tem)
#plm.cuda()
#plm.parallelize()

with open('saspre_classes.txt', 'r') as f:
    lines = f.readlines()

# 将每一行转换为一个列表元素
cla = [line.strip()for line in lines]
saspre_label = {item: [item, item[:-7]] for item in cla}
#print(saspre_label)
myverbalizer = ManualVerbalizer(tokenizer=tokenizer,classes=cla,label_words=saspre_label)#.from_file(f"saspre_classes.txt")

from torch import nn
#print("验证模板一样的y_groundtruth")
load_model = True
prompt_model = PromptForClassification(plm=plm,template=mytemplate1, verbalizer=myverbalizer, freeze_plm=True)
params_before = {name: p.clone().cpu() for name, p in prompt_model.named_parameters()}
load_dir = "load none"
if load_model == True:
     #prompt_model.load_state_dict(torch.load("./ht1_184.0000_ht10_804.0000_nd1_0.000000_nd10_0.000000_saspre4.ckpt"))
     #ht1_6448.0000_ht10_17718.0000_nd1_0.000000_nd10_0.000000_saspre4
     # adalora_config = AdaLoraConfig(peft_type="ADALORA", init_r=108, lora_alpha=32, lora_dropout=0.01,
     #                                target_modules=["q",
     #                                                "v"])  # ["o"]+[f"block.{i}.layer.1.EncDecAttention.q" for i in range(15, 24)] + [f"block.{i}.layer.0.SelfAttention.q" for i in range(15, 24)] +[f"block.{j}.layer.0.SelfAttention.v" for j in range(15, 24)] + [f"block.{i}.layer.1.EncDecAttention.v" for i in range(15, 24)])#[f"block.{i}.layer.0.SelfAttention.o" for i in range(15, 24)] + [f"block.{i}.layer.1.EncDecAttention.o" for i in range(15, 24)]+
     # prompt_model = get_peft_model(prompt_model, peft_config=adalora_config)
     # adalora_config = AdaLoraConfig(peft_type="ADALORA", init_r=320, lora_alpha=32, lora_dropout=0.01,
     #                                target_modules=["q",
     #                                                "v"])  # ["o"]+[f"block.{i}.layer.1.EncDecAttention.q" for i in range(15, 24)] + [f"block.{i}.layer.0.SelfAttention.q" for i in range(15, 24)] +[f"block.{j}.layer.0.SelfAttention.v" for j in range(15, 24)] + [f"block.{i}.layer.1.EncDecAttention.v" for i in range(15, 24)])#[f"block.{i}.layer.0.SelfAttention.o" for i in range(15, 24)] + [f"block.{i}.layer.1.EncDecAttention.o" for i in range(15, 24)]+
     # prompt_model = get_peft_model(prompt_model, peft_config=adalora_config)
     adalora_config = AdaLoraConfig(peft_type="ADALORA", init_r=320, lora_alpha=32, lora_dropout=0.01,
                                    target_modules=["q",
                                                    "v"])  # ["o"]+[f"block.{i}.layer.1.EncDecAttention.q" for i in range(15, 24)] + [f"block.{i}.layer.0.SelfAttention.q" for i in range(15, 24)] +[f"block.{j}.layer.0.SelfAttention.v" for j in range(15, 24)] + [f"block.{i}.layer.1.EncDecAttention.v" for i in range(15, 24)])#[f"block.{i}.layer.0.SelfAttention.o" for i in range(15, 24)] + [f"block.{i}.layer.1.EncDecAttention.o" for i in range(15, 24)]+
     prompt_model = get_peft_model(prompt_model, peft_config=adalora_config)
     load_dir = "./last/less_epoch_12/ht1_7586.0000_ht10_16496.0000_nd1_0.000000_nd10_8512.105311_epoch_51_lr5e-06_wd1e-07.ckpt"
     prompt_model.load_state_dict(torch.load(load_dir,map_location='cuda:0'))#ht1_6448.0000_ht10_17718.0000_nd1_0.000000_nd10_0.000000_saspre4

     print(f"loading {load_dir}")

     print("Load  Model Success!")
# params_after = {name: p.clone().cpu() for name, p in prompt_model.named_parameters()}
# aaa = 0
# # 计算参数的变化，并打印出发生变化的参数
# for name in params_before.keys():
#  diff = params_after[name] - params_before[name]
#  # if 'lora' in name:
#  #     print(f"{name}diff:",diff)
#  if torch.sum(diff).item() != 0:  # 如果参数发生了变化
#      # print(diff)
#      print(f"{name} has changed during training.")
#      # print(pd1.template.soft_embedding.weight)
#      # pvar = pd.template.soft_embedding.weight
#      aaa = aaa + 1
#
# print(aaa)

device = torch.device("cuda:0")
prompt_model = prompt_model.to(device)
#prompt_model = prompt_model.cuda()
prompt_model.parallelize()
#prompt_model = torch.nn.DataParallel(prompt_model,device_ids=[0,1])
#prompt_model.to(device)
#prompt_model.train()

# from bigmodelvis import Visualization
Visualization(prompt_model).structure_graph()

print("len:",len(ori_dataset["test"]))
train_dataloader = PromptDataLoader(dataset=ori_tes_dataset["test"], template=mytemplate1, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=920, decoder_max_length=5,
    batch_size=1,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")




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
            inputs = inputs.cuda()
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
optimizer_grouped_parameters2 = [{'params': [p for name, p in prompt_model.template.named_parameters() if 'raw_embedding' not in name]}] # note that you have to remove the raw_embedding manually from the optimization
tot_step = 400
#optimizer2 = Adafactor(optimizer_grouped_parameters2,lr=1e-3,relative_step=False)
optimizer2 = Lion(optimizer_grouped_parameters2, lr=1e-3) # usually lr = 0.5
#scheduler2 = get_linear_schedule_with_warmup(optimizer2,num_warmup_steps=200, num_training_steps=400) # usually num_warmup_steps is 500


best_ht1 = 0.00000
best_ht10 = 0.00000
best_nd1 = 0.00000
best_nd10 = 0.00000
ht1 = 0.00000
ht10 = 0.00000
nd1 = 0.00000
nd10 = 0.00000
gradient_accumulation_steps = 1

# adalora_config = AdaLoraConfig(peft_type="ADALORA",init_r=108,lora_alpha=32,lora_dropout=0.01,target_modules=["q","v"]) # ["o"]+[f"block.{i}.layer.1.EncDecAttention.q" for i in range(15, 24)] + [f"block.{i}.layer.0.SelfAttention.q" for i in range(15, 24)] +[f"block.{j}.layer.0.SelfAttention.v" for j in range(15, 24)] + [f"block.{i}.layer.1.EncDecAttention.v" for i in range(15, 24)])#[f"block.{i}.layer.0.SelfAttention.o" for i in range(15, 24)] + [f"block.{i}.layer.1.EncDecAttention.o" for i in range(15, 24)]+
# prompt_model = get_peft_model(prompt_model, peft_config=adalora_config)
params_before = {name: p.clone().cpu() for name, p in prompt_model.named_parameters()}
prompt_model.eval()
Visualization(prompt_model).structure_graph()
# ppp = nn.DataParallel(prompt_model)
# ppp.cuda()
#torch.cuda.empty_cache()
for epoch in tqdm(range(1)):
    prompt_model.eval()
    tot_loss = 0
    ht1 = 0.00000
    ht10 = 0.00000
    nd1 = 0.0000000
    nd10 = 0.0000000
    validate_num = 0
    unva_num = 0
    with torch.no_grad():
        #torch.cuda.empty_cache()
        for step, inputs in enumerate(train_dataloader):

            inputs = inputs.cuda()#device="cuda:1")
            #inputs=inputs
            logits = prompt_model(inputs)
            #logits2 = logits
            labels = inputs['label']
            # can_labels = inputs['meta']['user_idcan']
            # print("can:",can_labels)
            can_labels = []
            for i in inputs['guid']:
                can_labels.append(i[0])
                print(i)
            print("len: ",len(can_labels))
            # if len(can_labels) < 5:
            #     sys.exit()
            # if len(inputs['guid'])<5:
            #     sys.exit()

            labels=labels
            probabilities = F.softmax(logits, dim=-1)
            sorted_indices = torch.argsort(probabilities, descending=True)
            rankss = (sorted_indices == labels.unsqueeze(-1)).nonzero()[:, -1] + 1
            # ai = 0
            # valisum = 0
            # unvalid = 0
            # for ii in can_labels:
            #     iia =ii.cuda()
            #     ai = ai+1
            #     print(f"num{ai}:",iia)
            #     rank_can = (sorted_indices == iia.unsqueeze(-1)).nonzero()[:, -1] + 1
            #     vali = 0
            #     for j in range(0, len(rank_can)):
            #         i = rank_can[j]
            #         print(f"ran{ai}:",i)
            #         if i <= 10:
            #             vali = 1
            #         else:
            #             vali = 0
            #     if vali == 1:
            #         valisum += 1
            #     else:
            #         unvalid += 1
            # if valisum != 0:
            #     validate_num+=1
            # # else:
            # #     unva_num += 1
            # if unvalid == 5:
            #     unva_num +=1


                    # print(i.cpu())


            for j in range(0,len(rankss)):
                i =rankss[j]

                #print("icpu:",i.cpu())
                if i.cpu() <=  10:
                    nd10 / np.log2(i.cpu() + 2)
                    ht10 += 1
                if i.cpu() <= 1:
                    nd1 / np.log2(i.cpu() + 2)
                    ht1 += 1
            print(f"/ht1_{ht1:.4f}_ht10_{ht10:.4f}")
            # print("va:",valisum)
            # print("unva:",unvalid)


        print(f"/ht1_{ht1:.4f}_ht10_{ht10:.4f}_nd1_{nd1:.4f}_nd10_{nd10:.4f}")
        print(validate_num)
        print(unva_num)
        print(model_dir)
        # print("last 1062")
        print(load_dir)
        os.system("pause")



"""start lora_tes!!!!"""
target_modules = []
for name, param in prompt_model.named_parameters():
     if ('soft' in name):
         param.requires_grad = False
         print(param)
     if ("SelfAttention.q" in name) or ("SelfAttention.v" in name) or ("EncDecAttention.q" in name) or ("EncDecAttention.v" in name) or ("SelfAttention.o" in name) or ("EncDecAttention.o" in name):
         #print(name)
         block_number = int(name.split(".")[4])
         #print(block_number)
         if block_number >= 12:
             target_modules.append("block."+name.split(".")[4]+"."+name.split(".")[5]+"."+name.split(".")[6]+"."+name.split(".")[7]+"."+name.split(".")[8])
adalora_config = AdaLoraConfig(peft_type="ADALORA", r=16, lora_alpha=32, lora_dropout=0.01,
                                  target_modules=target_modules)  # ["o"]+[f"block.{i}.layer.1.EncDecAttention.q" for i in range(15, 24)] + [f"block.{i}.layer.0.SelfAttention.q" for i in range(15, 24)] +[f"block.{j}.layer.0.SelfAttention.v" for j in range(15, 24)] + [f"block.{i}.layer.1.EncDecAttention.v" for i in range(15, 24)])#[f"block.{i}.layer.0.SelfAttention.o" for i in range(15, 24)] + [f"block.{i}.layer.1.EncDecAttention.o" for i in range(15, 24)]+
prompt_model2 = get_peft_model(prompt_model, peft_config=adalora_config)
prompt_model2 = prompt_model2.cuda()

prompt_model2.parallelize()
#prompt_model.train()

# from bigmodelvis import Visualization
Visualization(prompt_model2).structure_graph()
for epoch in tqdm(range(1)):
    tot_loss = 0
    aht1 = 0.00000
    aht10 = 0.00000
    and1 = 0.0000000
    and10 = 0.0000000
    with torch.no_grad():
        #torch.cuda.empty_cache()
        for step, inputs in enumerate(train_dataloader):

            inputs = inputs.cuda()
            #inputs=inputs
            logits = prompt_model2(inputs)
            #logits2 = logits
            labels = inputs['label']
            labels=labels
            probabilities = F.softmax(logits, dim=-1)
            sorted_indices = torch.argsort(probabilities, descending=True)
            rankss = (sorted_indices == labels.unsqueeze(-1)).nonzero()[:, -1] + 1
            for j in range(0,len(rankss)):
                i =rankss[j]
                #print(i.cpu())
                if i.cpu() <=  10:
                    and10 / np.log2(i.cpu() + 2)
                    aht10 += 1
                if i.cpu() <= 1:
                    and1 / np.log2(i.cpu() + 2)
                    aht1 += 1
            print(f"/ht1_{aht1:.4f}_ht10_{aht10:.4f}")

        print("load")
        print(f"LORA /ht1_{aht1:.4f}_ht10_{aht10:.4f}_nd1_{and1:.4f}_nd10_{and10:.4f}")
        print(f"SIMPLE /ht1_{ht1:.4f}_ht10_{ht10:.4f}_nd1_{nd1:.4f}_nd10_{nd10:.4f}")
        print(model_dir)
        #print("last 1062")
        print(load_dir)


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
#print("false!simple")
'''
ht1_6448.0000_ht10_17718.0000_flan_t5_large  ===>  ht1_1754.0000_ht10_5241

500_ht1_3961.0000_ht10_12183.0000_epoch_5_ ===> ht1_1301.0000_ht10_4306
500_ht1_3909.0000_ht10_12249.0000_epoch_6 ===> ht1_1322.0000_ht10_4329
500_ht1_3886.0000_ht10_12283.0000_epoch_7 ===> ht1_1344.0000_ht10_4419
500_ht1_4208.0000_ht10_12690.0000_epoch_8 ===> ht1_1432.0000_ht10_4485
500_ht1_4468.0000_ht10_13088.0000_epoch_9 ===> ht1_1431.0000_ht10_4501
500_ht1_4575.0000_ht10_13479.0000_epoch_10 ===> ht1_1363.0000_ht10_4319 y
500_ht1_4752.0000_ht10_13612.0000_epoch_11 ===> ht1_1535.0000_ht10_4698 y
500_ht1_4832.0000_ht10_13615.0000_epoch_12 ===> ht1_1501.0000_ht10_4731 y
500_ht1_4856.0000_ht10_13802.0000_epoch_13 ===> ht1_1642.0000_ht10_4864 y
500_ht1_5081.0000_ht10_14063.0000_epoch_14 ===> ht1_1619.0000_ht10_4858 y
500_ht1_5278.0000_ht10_14248.0000_epoch_15 ===> ht1_1573.0000_ht10_4752 y
500_ht1_5342.0000_ht10_14322.0000_epoch_16 ===> ht1_1575.0000_ht10_4800 y
500_ht1_5446.0000_ht10_14606.0000_epoch_17 ===> ht1_1748.0000_ht10_4966 y 2183 5769
500_ht1_5533.0000_ht10_14677.0000_epoch_18 ===> ht1_1332.0000_ht10_4157 y
500_ht1_5687.0000_ht10_14781.0000_epoch_19 ===> ht1_1587.0000_ht10_4762 y
500_ht1_5751.0000_ht10_14892.0000_epoch_20_lr0.02_wd9e-11 ===> ht1_1636.0000_ht10_4818 y ht1_2120.0000_ht10_5660
(500_ht1_5797.0000_ht10_14898.0000_epoch_21_lr0.02_wd9e-11 ===> ht1_1592.0000_ht10_4707)
500_ht1_5973.0000_ht10_15130.0000_epoch_21_lr0.002_wd2e-09 ===> ht1_1698.0000_ht10_4881 y
500_ht1_6075.0000_ht10_15256.0000_epoch_22_lr0.002_wd2e-09 ===> ht1_1688.0000_ht10_4871 y
500_ht1_6141.0000_ht10_15270.0000_epoch_23_lr0.002_wd2e-09 ===> ht1_2108.0000_ht10_5574 y 1647 4769
500_ht1_6124.0000_ht10_15337.0000_epoch_24_lr0.002_wd2e-09 ===> ht1_2155.0000_ht10_5670 y ht1_1694.0000_ht10_4878 y
500_ht1_6096.0000_ht10_15303.0000_epoch_25_lr0.005_wd5e-09 ===> ht1_2106.0000_ht10_5576 y ht1_1660.0000_ht10_4786
500_ht1_6192.0000_ht10_15443.0000_epoch_26_lr0.005_wd5e-09 ===> ht1_2187.0000_ht10_5687 y ht1_1748.0000_ht10_4959 y
500_ht1_6193.0000_ht10_15447.0000_epoch_27_lr0.005_wd5e-09 ===> ht1_2237.0000_ht10_5768 y 1770 5022 y
500_ht1_6254.0000_ht10_15411.0000_epoch_28_lr0.005_wd6e-09 ===> ht1_2164.0000_ht10_5698 y ht1_1739.0000_ht10_4957
500_ht1_6236.0000_ht10_15490.0000_epoch_29_lr0.005_wd6e-09 ===> ht1_2174.0000_ht10_5705 y ht1_1741.0000_ht10_4909
500_ht1_6315.0000_ht10_15568.0000_epoch_30_lr0.005_wd6e-09 ===> ht1_2240.0000_ht10_5807 y ht1_1794.0000_ht10_5091 y ht1_2608.0000_ht10_6273
(500_ht1_6321.0000_ht10_15509.0000_epoch_31_lr0.005_wd6e-09 ===> ht1_2111.0000_ht10_5611 y ht1_1644.0000_ht10_4770  )
(500_ht1_6406.0000_ht10_15606.0000_epoch_31_lr0.001_wd3e-09 ===> ht1_2175.0000_ht10_5681 y ht1_1743.0000_ht10_4950 )
(500_ht1_6427.0000_ht10_15649.0000_epoch_32_lr0.001_wd3e-09 ===> ht1_2154.0000_ht10_5632                           )
(500_ht1_6444.0000_ht10_15615.0000_epoch_33_lr0.001_wd3e-09 ===> ht1_2151.0000_ht10_5606 y ht1_1727.0000_ht10_4895 )
(500_ht1_6343.0000_ht10_15569.0000_epoch_31_lr0.0008_wd8e-09 ===>                         y ht1_1750.0000_ht10_4997)
(500_ht1_6393.0000_ht10_15646.0000_epoch_32_lr0.0008_wd8e-09 ===>                         y ht1_1759.0000_ht10_5006)
500_ht1_6401.0000_ht10_15643.0000_epoch_33_lr0.0008_wd8e-09 ===> ht1_2221.0000_ht10_5719 y ht1_1798.0000_ht10_5023 y ht1_2537.0000_ht10_6189

500_ht1_6392.0000_ht10_15578.0000_epoch_31_lr0.0002_wd6e-09 ===> 5:ht1_2574.0000_ht10_6279-8644-987
500_ht1_6377.0000_ht10_15614.0000_epoch_32_lr0.0002_wd6e-09 ===> 5:ht1_2556.0000_ht10_6264-8643-988

500_ht1_6378.0000_ht10_15568.0000_epoch_31_lr0.0005_wd5e-09 ===> 5:ht1_2562.0000_ht10_6264-8643-988
500_ht1_6388.0000_ht10_15594.0000_epoch_32_lr0.0005_wd5e-09 ===> 5:ht1_2541.0000_ht10_6227-8643-988
500_ht1_6400.0000_ht10_15590.0000_epoch_33_lr0.0005_wd5e-09 ===> 5:ht1_2536.0000_ht10_6224-8628-1003
500_ht1_6406.0000_ht10_15676.0000_epoch_34_lr0.0005_wd5e-09 ===> 5:ht1_2530.0000_ht10_6206-8627-1004
500_ht1_6488.0000_ht10_15644.0000_epoch_35_lr0.0005_wd5e-09 ===> 5:ht1_2505.0000_ht10_6162-8630-1001


500_ht1_6353.0000_ht10_15576.0000_epoch_31_lr0.005_wd1e-08 ===> 5:ht1_2514.0000_ht10_6203-8724-907
500_ht1_6348.0000_ht10_15614.0000_epoch_32_lr0.005_wd1e-08 ===> 5:ht1_2475.0000_ht10_6126-8609-1022
500_ht1_6386.0000_ht10_15613.0000_epoch_33_lr0.005_wd1e-08 ===> 5:ht1_2432.0000_ht10_6067-8629-1002
500_ht1_6348.0000_ht10_15616.0000_epoch_34_lr0.005_wd1e-08 ===> 5:ht1_2492.0000_ht10_6179-8535-1096
500_ht1_6388.0000_ht10_15639.0000_epoch_35_lr0.005_wd1e-08 ===> 5:ht1_2481.0000_ht10_6153-8540-1091 
500_ht1_6440.0000_ht10_15705.0000_epoch_36_lr0.005_wd1e-08 ===> 5:ht1_2573.0000_ht10_6221-8457-1147

500_ht1_6450.0000_ht10_15716.0000_epoch_38_lr0.005_wd1e-08 ===> 5:ht1_2504.0000_ht10_6177-8506-1125
500_ht1_6513.0000_ht10_15795.0000_epoch_39_lr0.005_wd1e-08 ===> 5:ht1_2627.0000_ht10_6330-8633-998
500_ht1_6516.0000_ht10_15836.0000_epoch_41_lr0.005_wd1e-08 ===> 5:ht1_2589.0000_ht10_6283-8544-1087
500_ht1_6571.0000_ht10_15791.0000_epoch_42_lr0.005_wd1e-08 ===> 5:ht1_2609.0000_ht10_6314-8584-1047
500_ht1_6563.0000_ht10_15810.0000_epoch_43_lr0.005_wd1e-08 ===> 5:ht1_2635.0000_ht10_6316-8497-1134
500_ht1_6570.0000_ht10_15847.0000_epoch_44_lr0.005_wd1e-08 ===> 5:ht1_2584.0000_ht10_6280-8539-1092
500_ht1_6608.0000_ht10_15839.0000_epoch_45_lr0.005_wd1e-08 ===> 5:ht1_2593.0000_ht10_6276-8492-1139
500_ht1_6600.0000_ht10_15895.0000_epoch_46_lr0.005_wd1e-08 ===> 5:ht1_2614.0000_ht10_6265-8447-1184
500_ht1_6590.0000_ht10_15903.0000_epoch_47_lr0.005_wd1e-08 ===> 5:ht1_2569.0000_ht10_6257-8551-1080
500_ht1_6621.0000_ht10_15940.0000_epoch_48_lr0.005_wd1e-08 ===> 5:ht1_2636.0000_ht10_6322-8513-1118
500_ht1_6687.0000_ht10_15956.0000_epoch_49_lr0.005_wd1e-08 ===> 5:ht1_2620.0000_ht10_6329-8571-1060
500_ht1_6697.0000_ht10_15933.0000_epoch_50_lr0.005_wd1e-08 ===> 5:ht1_2620.0000_ht10_6311-8534-1097
500_ht1_6721.0000_ht10_15975.0000_epoch_51_lr0.005_wd1e-08 ===> 5:ht1_2578.0000_ht10_6249-8504-1127
500_ht1_6691.0000_ht10_15995.0000_epoch_52_lr0.005_wd1e-08 ===> 5:ht1_2651.0000_ht10_6353-8523-1108

500_ht1_6764.0000_ht10_15994.0000_epoch_54_lr0.005_wd1e-08 ===> 5:ht1_2636.0000_ht10_6337-8493-1138
500_ht1_6767.0000_ht10_16007.0000_epoch_55_lr0.005_wd1e-08 ===> 5:ht1_2581.0000_ht10_6265-8478-1153
500_ht1_6753.0000_ht10_16034.0000_epoch_56_lr0.005_wd1e-08 ===> 5:ht1_2611.0000_ht10_6274-8414-1217
500_ht1_6757.0000_ht10_16025.0000_epoch_57_lr0.005_wd1e-08 ===> 5:ht1_2667.0000_ht10_6301-8437-1194
500_ht1_6795.0000_ht10_16076.0000_epoch_58_lr0.005_wd1e-08 ===> 5:ht1_2620.0000_ht10_6311-8534-1097
500_ht1_6799.0000_ht10_16050.0000_epoch_59_lr0.005_wd1e-08 ===> 5:ht1_2632.0000_ht10_6276-8416-1215
500_ht1_6801.0000_ht10_16066.0000_epoch_60_lr0.005_wd1e-08 ===> 5:ht1_2545.0000_ht10_6187-8484-1147
500_ht1_6855.0000_ht10_16148.0000_epoch_61_lr0.005_wd1e-08 ===> 5:ht1_2623.0000_ht10_6234-8407-1224


500_ht1_6768.0000_ht10_16050.0000_epoch_58_lr0.007_wd5e-08 ===> 5:ht1_2616.0000_ht10_6230-8407-1224
500_ht1_6794.0000_ht10_16044.0000_epoch_60_lr0.007_wd5e-08 ===> 5:ht1_2559.0000_ht10_6181-8439-1192
500_ht1_6768.0000_ht10_16063.0000_epoch_62_lr0.007_wd5e-08 ===> 5:ht1_2615.0000_ht10_6243-8433-1198
500_ht1_6779.0000_ht10_16063.0000_epoch_63_lr0.007_wd5e-08 ===> 5:ht1_2648.0000_ht10_6328-8446-1185
* 500_ht1_6846.0000_ht10_16128.0000_epoch_64_lr0.007_wd5e-08 ===> 5:ht1_2701.0000_ht10_6345-8353-1278
500_ht1_6849.0000_ht10_16137.0000_epoch_66_lr0.007_wd5e-08 ===> 5:ht1_2617.0000_ht10_6279-8409-1222

500_ht1_6944.0000_ht10_16192.0000_epoch_65_lr0.0007_wd5e-08 ===> 5:ht1_2696.0000_ht10_6353-8381-1250
500_ht1_6882.0000_ht10_16193.0000_epoch_67_lr0.0007_wd5e-08 ===> 5:ht1_2676.0000_ht10_6323-8384-1247
500_ht1_6873.0000_ht10_16205.0000_epoch_68_lr0.0007_wd5e-08 ===> 5:ht1_2673.0000_ht10_6315-8362-1269
500_ht1_6933.0000_ht10_16239.0000_epoch_69_lr0.0007_wd5e-08 ===> 5:ht1_2679.0000_ht10_6315-8355-1276

ht1_6242.0000_ht10_15243.0000_nd1_0.000000_nd10_7605.790727_epoch_0_lr1e-05_wd5e-08===ht1_2783.0000_ht10_6579
ht1_6271.0000_ht10_15443.0000_nd1_0.000000_nd10_7683.252236_epoch_1_lr1e-05_wd5e-08===ht1_2790.0000_ht10_6585
ht1_6398.0000_ht10_15536.0000_nd1_0.000000_nd10_7759.800905_epoch_2_lr1e-05_wd5e-08===ht1_2799.0000_ht10_6593
ht1_6376.0000_ht10_15552.0000_nd1_0.000000_nd10_7762.673095_epoch_3_lr1e-05_wd5e-08===ht1_2810.0000_ht10_6600
ht1_6459.0000_ht10_15610.0000_nd1_0.000000_nd10_7816.537234_epoch_4_lr1e-05_wd5e-08===ht1_2824.0000_ht10_6603
ht1_6529.0000_ht10_15656.0000_nd1_0.000000_nd10_7841.533354_epoch_5_lr1e-05_wd5e-08===ht1_2842.0000_ht10_6624
ht1_6548.0000_ht10_15705.0000_nd1_0.000000_nd10_7878.112668_epoch_6_lr1e-05_wd5e-08===ht1_2844.0000_ht10_6619
ht1_6569.0000_ht10_15739.0000_nd1_0.000000_nd10_7896.355325_epoch_7_lr1e-05_wd5e-08===ht1_2840.0000_ht10_6628
ht1_6596.0000_ht10_15799.0000_nd1_0.000000_nd10_7925.095647_epoch_8_lr1e-05_wd5e-08===ht1_2837.0000_ht10_6633
ht1_6652.0000_ht10_15768.0000_nd1_0.000000_nd10_7927.034045_epoch_9_lr1e-05_wd5e-08===ht1_2845.0000_ht10_6636
ht1_6691.0000_ht10_15858.0000_nd1_0.000000_nd10_7977.152602_epoch_10_lr1e-05_wd5e-08===ht1_2847.0000_ht10_6637
ht1_6712.0000_ht10_15824.0000_nd1_0.000000_nd10_7976.887523_epoch_11_lr1e-05_wd5e-08===ht1_2846.0000_ht10_6626
ht1_6739.0000_ht10_15894.0000_nd1_0.000000_nd10_8012.383625_epoch_12_lr1e-05_wd5e-08===ht1_2838.0000_ht10_6630

ht1_6726.0000_ht10_15845.0000_nd1_0.000000_nd10_7984.898258_epoch_13_lr1e-05_wd7e-07===ht1_2839.0000_ht10_6624
ht1_6821.0000_ht10_15853.0000_nd1_0.000000_nd10_8014.661979_epoch_14_lr1e-05_wd7e-07===ht1_2844.0000_ht10_6624
ht1_6811.0000_ht10_15911.0000_nd1_0.000000_nd10_8040.504760_epoch_15_lr1e-05_wd7e-07===ht1_2847.0000_ht10_6627
ht1_6828.0000_ht10_15980.0000_nd1_0.000000_nd10_8072.281406_epoch_17_lr1e-05_wd7e-07===ht1_2837.0000_ht10_6620
ht1_6842.0000_ht10_15970.0000_nd1_0.000000_nd10_8071.097959_epoch_18_lr1e-05_wd7e-07===ht1_2845.0000_ht10_6629

ht1_6474.0000_ht10_15584.0000_nd1_0.000000_nd10_7804.283601_epoch_13_lr0.0002_wd8e-07===ht1_2801.0000_ht10_6615
ht1_6548.0000_ht10_15714.0000_nd1_0.000000_nd10_7883.053028_epoch_14_lr0.0002_wd8e-07===ht1_2792.0000_ht10_6605
ht1_6943.0000_ht10_16063.0000_nd1_0.000000_nd10_8135.156467_epoch_17_lr0.0002_wd8e-07===ht1_2782.0000_ht10_6579

ht1_6576.0000_ht10_15747.0000_nd1_0.000000_nd10_7899.253833_epoch_11_lr5e-05_wd1e-08===ht1_2828.0000_ht10_6616
ht1_6669.0000_ht10_15841.0000_nd1_0.000000_nd10_7969.913640_epoch_12_lr5e-05_wd1e-08===ht1_2831.0000_ht10_6617
ht1_6789.0000_ht10_15885.0000_nd1_0.000000_nd10_8023.763017_epoch_13_lr5e-05_wd1e-08===ht1_2849.0000_ht10_6627

ht1_6947.0000_ht10_16005.0000_nd1_0.000000_nd10_8117.076105_epoch_15_lr5e-05_wd1e-08===ht1_2836.0000_ht10_6604
ht1_7056.0000_ht10_16105.0000_nd1_0.000000_nd10_8197.428503_epoch_16_lr5e-05_wd1e-08===ht1_2839.0000_ht10_6620
ht1_7121.0000_ht10_16153.0000_nd1_0.000000_nd10_8230.318855_epoch_17_lr5e-05_wd1e-08===ht1_2819.0000_ht10_6607
ht1_7212.0000_ht10_16233.0000_nd1_0.000000_nd10_8290.987315_epoch_18_lr5e-05_wd1e-08===ht1_2806.0000_ht10_6600
ht1_7343.0000_ht10_16292.0000_nd1_0.000000_nd10_8352.848463_epoch_19_lr5e-05_wd1e-08===ht1_2798.0000_ht10_6580

ht1_6957.0000_ht10_15994.0000_nd1_0.000000_nd10_8116.320096_epoch_14_lr5e-06_wd1e-07===ht1_2842.0000_ht10_6630
ht1_6988.0000_ht10_16042.0000_nd1_0.000000_nd10_8141.695873_epoch_15_lr5e-06_wd1e-07===ht1_2832.0000_ht10_6621
ht1_6992.0000_ht10_16073.0000_nd1_0.000000_nd10_8155.166387_epoch_16_lr5e-06_wd1e-07===ht1_2834.0000_ht10_6620
ht1_7016.0000_ht10_16120.0000_nd1_0.000000_nd10_8184.061218_epoch_17_lr5e-06_wd1e-07===ht1_2840.0000_ht10_6620
ht1_7051.0000_ht10_16125.0000_nd1_0.000000_nd10_8204.393974_epoch_18_lr5e-06_wd1e-07===ht1_2840.0000_ht10_6626
ht1_7096.0000_ht10_16107.0000_nd1_0.000000_nd10_8200.929422_epoch_19_lr5e-06_wd1e-07===ht1_2844.0000_ht10_6619
ht1_7099.0000_ht10_16166.0000_nd1_0.000000_nd10_8228.298400_epoch_20_lr5e-06_wd1e-07===ht1_2846.0000_ht10_6623
ht1_7133.0000_ht10_16177.0000_nd1_0.000000_nd10_8236.283898_epoch_22_lr5e-06_wd1e-07===ht1_2834.0000_ht10_6628
ht1_7107.0000_ht10_16182.0000_nd1_0.000000_nd10_8235.704574_epoch_23_lr5e-06_wd1e-07===ht1_2833.0000_ht10_6619
ht1_7130.0000_ht10_16211.0000_nd1_0.000000_nd10_8260.639648_epoch_24_lr5e-06_wd1e-07===ht1_2839.0000_ht10_6626
ht1_7225.0000_ht10_16198.0000_nd1_0.000000_nd10_8277.028377_epoch_25_lr5e-06_wd1e-07===ht1_2833.0000_ht10_6630
ht1_7215.0000_ht10_16219.0000_nd1_0.000000_nd10_8278.491825_epoch_26_lr5e-06_wd1e-07===ht1_2828.0000_ht10_6623
ht1_7200.0000_ht10_16242.0000_nd1_0.000000_nd10_8288.652014_epoch_27_lr5e-06_wd1e-07===ht1_2845.0000_ht10_6633
ht1_7240.0000_ht10_16219.0000_nd1_0.000000_nd10_8292.359821_epoch_28_lr5e-06_wd1e-07===ht1_2837.0000_ht10_6618

ht1_7265.0000_ht10_16276.0000_nd1_0.000000_nd10_8319.458049_epoch_30_lr5e-06_wd1e-07===ht1_2853.0000_ht10_6610
ht1_7306.0000_ht10_16283.0000_nd1_0.000000_nd10_8338.041776_epoch_31_lr5e-06_wd1e-07===ht1_2850.0000_ht10_6613
ht1_7311.0000_ht10_16258.0000_nd1_0.000000_nd10_8334.524485_epoch_32_lr5e-06_wd1e-07===ht1_2849.0000_ht10_6622
ht1_7312.0000_ht10_16325.0000_nd1_0.000000_nd10_8359.969727_epoch_33_lr5e-06_wd1e-07===ht1_2847.0000_ht10_6616
ht1_7283.0000_ht10_16328.0000_nd1_0.000000_nd10_8354.906141_epoch_34_lr5e-06_wd1e-07===ht1_2842.0000_ht10_6619
ht1_7345.0000_ht10_16330.0000_nd1_0.000000_nd10_8369.977894_epoch_35_lr5e-06_wd1e-07===ht1_2853.0000_ht10_6612
ht1_7326.0000_ht10_16350.0000_nd1_0.000000_nd10_8374.514828_epoch_36_lr5e-06_wd1e-07===ht1_2850.0000_ht10_6612
ht1_7378.0000_ht10_16346.0000_nd1_0.000000_nd10_8390.389333_epoch_37_lr5e-06_wd1e-07===ht1_2844.0000_ht10_6614
null
ht1_7378.0000_ht10_16368.0000_nd1_0.000000_nd10_8396.962418_epoch_39_lr5e-06_wd1e-07===ht1_2851.0000_ht10_6608
* ht1_7437.0000_ht10_16365.0000_nd1_0.000000_nd10_8411.565664_epoch_40_lr5e-06_wd1e-07===ht1_2859.0000_ht10_6613
ht1_7414.0000_ht10_16392.0000_nd1_0.000000_nd10_8414.721065_epoch_41_lr5e-06_wd1e-07===ht1_2846.0000_ht10_6608
ht1_7438.0000_ht10_16413.0000_nd1_0.000000_nd10_8428.455498_epoch_42_lr5e-06_wd1e-07===ht1_2845.0000_ht10_6617
ht1_7450.0000_ht10_16431.0000_nd1_0.000000_nd10_8446.966082_epoch_43_lr5e-06_wd1e-07===ht1_2826.0000_ht10_6600



ht1_7538.0000_ht10_16440.0000_nd1_0.000000_nd10_8473.364841_epoch_47_lr5e-06_wd1e-07===ht1_2837.0000_ht10_6598
'''
sys.exit()


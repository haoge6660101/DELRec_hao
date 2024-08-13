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
from transformers import get_linear_schedule_with_warmup
from ..utils import creat_Verbalizer
from ..MTL.MTL import dynamic_loss_weighting
from ..utils import check_suffix, rearrange_string,create_prompt,check_seq_spell
from ..dataload import create_sas_dataset, read_file_portions_based_rows, create_TA_dataset, read_file_portions_based_ratio
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
from ..dataload import create_RPS_dataset
from temporal_analysis import creat_TA_datalines
from transformers import AutoTokenizer,AutoModelForMultipleChoice,AutoConfig,AutoModelWithLMHead
from openprompt.plms.seq2seq import T5LMTokenizerWrapper, T5TokenizerWrapper


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print( torch.cuda.device_count(), "GPUs")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def creat_RPS_datalines(load_datalines):

    if load_datalines == True:
        with open('TA_dataset1.pkl', 'rb') as f:
            dataset1 = pickle.load(f)

        with open('TA_dataset2.pkl', 'rb') as f:
            dataset2 = pickle.load(f)

        with open('TA_dataset3.pkl', 'rb') as f:
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

        f1, f2, f3 = read_file_portions_based_rows('../../user_interactions_with_text_title_and_predicted_10_items_by_SR',train_list_lines, test_list_lines, val_list_lines)
    else:
        f1, f2, f3 = read_file_portions_based_ratio('../../user_interactions_with_text_title_and_predicted_10_items_by_SR')

    return f1, f2, f3

load_sas = False
if load_sas == False:

    df = pd.read_csv(r'../title_set.csv')
    id_set = df.set_index('Title')['ID'].to_dict()

    def RPS_data_partition(ff):
        usernum = 0
        itemnum = 0
        User = defaultdict(list)
        user_seq = {}
        user_can = {}
        user_label = {}
        SR_pre = {}
        model_name = {}

        for line in ff:
            u = line.rstrip().split(' ')[0]
            i = ' '.join(line.rstrip().split(' ')[1:])
            u = int(u)
            i = str(i)
            usernum = max(u, usernum)
            User[u].append(i)

        for user in tqdm(User):
            if len(User[user]) < 14: #or19
                user_seq[user] = User[user]
                user_can[user] = []
                user_label[user] = []
                SR_pre[user] = []
                model_name[user] = 'SASRec'

            else:
                model_name[user] = 'SASRec'
                SR_pre[user] = []
                SR_pre[user] = User[user][-9:] #or-14
                user_seq[user] = User[user][:-10] #or-15
                user_can[user] = []
                it = User[user][-10]#or-15
                user_can[user].append(it)
                user_label[user] = []
                user_label[user].append(id_set[it])

                for pre_item in User[user][-9:]:#or-14
                    user_can[user].append(pre_item)

                for _ in range(10):
                    can_item = np.random.choice(list(id_set.keys()))
                    while can_item in user_can[user] and can_item in User[user]:
                        can_item = np.random.choice(list(id_set.keys()))
                    user_can[user].append(can_item)
                np.random.shuffle(user_can[user])

        return [user_seq, user_can, user_label, model_name,SR_pre]


    rpsf1, rpsf2, rpsf3 = creat_RPS_datalines(False)
    RPS_dataset1 = RPS_data_partition(rpsf1)
    RPS_dataset2 = RPS_data_partition(rpsf2)
    RPS_dataset3 = RPS_data_partition(rpsf3)

    with open('./RPS_dataset1.pkl', 'wb') as f:
        pickle.dump(RPS_dataset1, f)

    with open('./RPS_dataset2.pkl', 'wb') as f:
        pickle.dump(RPS_dataset2, f)

    with open('./RPS_dataset3.pkl', 'wb') as f:
        pickle.dump(RPS_dataset3, f)

else:

    with open('RPS_dataset1.pkl', 'rb') as f:
        RPS_dataset1 = pickle.load(f)

    with open('RPS_dataset2.pkl', 'rb') as f:
        RPS_dataset2 = pickle.load(f)

    with open('RPS_dataset3.pkl', 'rb') as f:
        RPS_dataset3 = pickle.load(f)




RPS_ori_dataset = {
    'train': create_RPS_dataset(RPS_dataset1, 'train'),
    'validation': create_RPS_dataset(RPS_dataset2, 'test'),
    'test': create_RPS_dataset(RPS_dataset3, 'validation')
}

RPS_dataset = {}
for split in ['train', 'test', 'validation']:
    RPS_dataset[split] = []
    for data in RPS_ori_dataset[split]:

        check_seq_spell(data['user_seq'])
        check_seq_spell(data['user_can'])
        check_seq_spell(data['SR_pre'])

        user_seq = ', '.join(data['user_seq'])
        user_can = ', '.join(data['user_can'])
        SR_pre = ','.join(data['SR_pre'])

        if len(data['user_label']) > 0:
            input_example = InputExample(label=(data['user_label'][0] - 1), guid=int(data['user_id']),
                                         meta={'SR_pre': SR_pre, 'user_seq': user_seq, 'user_can': user_can, 'model_name': data['model_name']})
            RPS_dataset[split].append(input_example)



plm, tokenizer, model_config, WrapperClass = load_plm("t5","../../flan-t5-xl")

mytemplate = create_prompt('RPS',plm,tokenizer,prompt_id=0)

def load_RPS_prompt():
    mytemplate_RPS = create_prompt('RPS', plm, tokenizer, prompt_id=0)
    return mytemplate_RPS


def load_RPS_dataset():
    train_dataloader = PromptDataLoader(dataset=RPS_dataset["train"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=1065, decoder_max_length=10,
        batch_size=20,shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")

    test_dataloader = PromptDataLoader(dataset=RPS_dataset["test"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=1065, decoder_max_length=10,
        batch_size=20,shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")

    validation_dataloader = PromptDataLoader(dataset=RPS_dataset["validation"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=1065, decoder_max_length=10,
        batch_size=20,shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")
    return train_dataloader,test_dataloader,validation_dataloader
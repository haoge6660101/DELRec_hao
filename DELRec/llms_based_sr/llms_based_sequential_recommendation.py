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
from ..dataload import create_sas_dataset, read_file_portions_based_rows, create_LSR_dataset, read_file_portions_based_ratio
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
from ..distill_pattern_from_conventional_SR_models.temporal_analysis import creat_TA_datalines
from transformers import AutoTokenizer,AutoModelForMultipleChoice,AutoConfig,AutoModelWithLMHead
from openprompt.plms.seq2seq import T5LMTokenizerWrapper, T5TokenizerWrapper


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print( torch.cuda.device_count(), "GPUs")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def creat_LSR_datalines(load_datalines):

    if load_datalines == True:
        with open('LSR_dataset1.pkl', 'rb') as f:
            dataset1 = pickle.load(f)

        with open('LSR_dataset2.pkl', 'rb') as f:
            dataset2 = pickle.load(f)

        with open('LSR_dataset3.pkl', 'rb') as f:
            dataset3 = pickle.load(f)

        dataset = {
            'train': create_LSR_dataset(dataset1, 'train'),
            'validation': create_LSR_dataset(dataset2, 'validation'),
            'test': create_LSR_dataset(dataset3, 'test')
        }

        pre_dataset = {}
        for split in ['train', 'test', 'validation']:
            pre_dataset[split] = []
            for data in LSR_ori_dataset[split]:

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

        f1, f2, f3 = read_file_portions_based_rows('../../user_interactions_with_text_title_and_ground_truth',train_list_lines, test_list_lines, val_list_lines)
    else:
        f1, f2, f3 = read_file_portions_based_ratio('../../user_interactions_with_text_title_and_ground_truth')

    return f1, f2, f3

load_sas = False
if load_sas == False:

    df = pd.read_csv(r'../title_set.csv')
    id_set = df.set_index('Title')['ID'].to_dict()

    def LSR_data_partition(ff):
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
            if len(User[user]) < 5:
                user_seq[user] = User[user]
                user_can[user] = []
                user_label[user] = []
                model_name[user] = 'SASRec'
            else:
                model_name[user] = 'SASRec'
                user_seq[user] = User[user][:-1]
                user_can[user] = []
                it = User[user][-1]
                user_can[user].append(it)
                user_label[user] = []
                user_label[user].append(id_set[it])

                for _ in range(19):
                    can_item = np.random.choice(list(id_set.keys()))
                    while can_item in user_can[user] and can_item in User[user]:
                        can_item = np.random.choice(list(id_set.keys()))
                    user_can[user].append(can_item)
                np.random.shuffle(user_can[user])

        return [user_seq, user_can, user_label, model_name]


    lsrf1, lsrf2, lsrf3 = creat_LSR_datalines(False)
    LSR_dataset1 = LSR_data_partition(lsrf1)
    LSR_dataset2 = LSR_data_partition(lsrf2)
    LSR_dataset3 = LSR_data_partition(lsrf3)

    with open('./LSR_dataset1.pkl', 'wb') as f:
        pickle.dump(LSR_dataset1, f)

    with open('./LSR_dataset2.pkl', 'wb') as f:
        pickle.dump(LSR_dataset2, f)

    with open('./LSR_dataset3.pkl', 'wb') as f:
        pickle.dump(LSR_dataset3, f)

else:

    with open('LSR_dataset1.pkl', 'rb') as f:
        LSR_dataset1 = pickle.load(f)

    with open('LSR_dataset2.pkl', 'rb') as f:
        LSR_dataset2 = pickle.load(f)

    with open('LSR_dataset3.pkl', 'rb') as f:
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
                                         meta={'user_seq': user_seq, 'user_can': user_can, 'model_name': data['model_name']})
            LSR_dataset[split].append(input_example)



plm, tokenizer, model_config, WrapperClass = load_plm("t5","../../flan-t5-xl")

mytemplate = create_prompt('LSR',plm,tokenizer,prompt_id=0)

def load_LSR_prompt():
    mytemplate_LSR = create_prompt('LSR', plm, tokenizer, prompt_id=0)
    return mytemplate_LSR


def load_LSR_dataset():
    train_dataloader = PromptDataLoader(dataset=LSR_dataset["train"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=1065, decoder_max_length=10,
        batch_size=20,shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")

    test_dataloader = PromptDataLoader(dataset=LSR_dataset["test"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=1065, decoder_max_length=10,
        batch_size=20,shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")

    validation_dataloader = PromptDataLoader(dataset=LSR_dataset["validation"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=1065, decoder_max_length=10,
        batch_size=20,shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")
    return train_dataloader,test_dataloader,validation_dataloader
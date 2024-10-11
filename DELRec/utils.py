import re
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from openprompt.prompts import ManualVerbalizer
from openprompt.prompts import MixedTemplate


def check_suffix(s):
    pattern = r",\s(The|A|An)\s\(\d{4}\)$"

    if re.search(pattern, s):
        return True
    else:
        return False


def rearrange_string(s):
    pattern = r"(.*),\s(The|A|An)\s\((\d{4})\)"

    return re.sub(pattern, r"\2 \1 (\3)", s)


def check_seq_spell(dataset):
    for index, i in enumerate(dataset):
        if check_suffix(i):
            dataset[index] = rearrange_string(i)


def find_param(prompt_model, n):
    params_before = {name: p.clone().cpu() for name, p in prompt_model.named_parameters()}
    for name in params_before.keys():
        if n in name:
            param = params_before[name]
            return param, name


def creat_Verbalizer(tokenizer):
    with open('../title_set.txt', 'r') as f:
        lines = f.readlines()
    cla = [line.strip() for line in lines]
    saspre_label = {item: [item, item[:-7]] for item in cla}
    myverbalizer = ManualVerbalizer(tokenizer=tokenizer, classes=cla, label_words=saspre_label)
    return myverbalizer


def create_prompt(scriptsbase, plm, tokenizer, prompt_id):
    mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer).from_file(f'data/prompts/{scriptsbase}.txt',
                                                                         choice=prompt_id)
    return mytemplate


def calculate_metrics(logits, labels, nd5, ht5, ht1):
    probabilities = F.softmax(logits, dim=-1)
    sorted_indices = torch.argsort(probabilities, descending=True)
    rankss = (sorted_indices == labels.unsqueeze(-1)).nonzero()[:, -1] + 1
    for j in range(0, len(rankss)):
        i = rankss[j]

        if i.cpu() <= 5:
            nd5 += 1 / np.log2(i.cpu() + 1)
            ht5 += 1
        if i.cpu() <= 1:
            ht1 += 1
    return nd5, ht5, ht1


def evaluate(prompt_model, dataloader, nd5=0.0000, ht5=0.0000, ht1=0.0000, use_cuda=True):
    prompt_model.eval()

    for step, inputs in enumerate(dataloader):
        if use_cuda:
            inputs = inputs.cuda()

        logits = prompt_model(inputs)
        labels = inputs['label']
        probabilities = F.softmax(logits, dim=-1)
        sorted_indices = torch.argsort(probabilities, descending=True)
        rankss = (sorted_indices == labels.unsqueeze(-1)).nonzero()[:, -1] + 1
        for j in range(0, len(rankss)):
            i = rankss[j]

            if i.cpu() <= 5:
                nd5 += 1 / np.log2(i.cpu() + 1)
                ht5 += 1
            if i.cpu() <= 1:
                ht1 += 1

    return nd5, ht5, ht1

from transformers import get_linear_schedule_with_warmup
from bitsandbytes.optim import PagedLion8bit
from ..MTL.MTL import dynamic_loss_weighting
from ..utils import calculate_metrics
from openprompt.plms import load_plm
from tqdm import tqdm
import torch
from openprompt import PromptForClassification
from ..utils import evaluate,creat_Verbalizer
from data.distill_pattern_from_conventional_SR_models.temporal_analysis import load_TA_dataset,load_TA_prompt
from data.distill_pattern_from_conventional_SR_models.recommendation_pattern_simulating import load_RPS_dataset,load_RPS_prompt
from llms_based_sequential_recommendation import load_LSR_prompt,load_LSR_dataset
from peft import AdaLoraConfig,get_peft_model

def training_of_second_stage(args, learned_soft_prompt):
    plm, tokenizer, model_config, WrapperClass = load_plm("t5","../../flan-t5-xl")

    LSR_train_dataloader,LSR_test_dataloader,LSR_val_dataloader = load_LSR_dataset()
    LSR_template = load_LSR_prompt()

    prompt_model = PromptForClassification(plm=plm,template=LSR_template, verbalizer=creat_Verbalizer(tokenizer), freeze_plm=True)
    load_dir = learned_soft_prompt
    prompt_model.load_state_dict(torch.load(load_dir,map_location="cuda:0"))
    for name, param in prompt_model.named_parameters():
         if ('soft' in name):
             param.requires_grad = False
    adalora_config = AdaLoraConfig(peft_type="ADALORA", init_r=320, lora_alpha=32, lora_dropout=0.05,target_modules=["q","v"])  # ["o"]+[f"block.{i}.layer.1.EncDecAttention.q" for i in range(15, 24)] + [f"block.{i}.layer.0.SelfAttention.q" for i in range(15, 24)] +[f"block.{j}.layer.0.SelfAttention.v" for j in range(15, 24)] + [f"block.{i}.layer.1.EncDecAttention.v" for i in range(15, 24)])#[f"block.{i}.layer.0.SelfAttention.o" for i in range(15, 24)] + [f"block.{i}.layer.1.EncDecAttention.o" for i in range(15, 24)]+
    prompt_model = get_peft_model(prompt_model, peft_config=adalora_config)

    prompt_model.parallelize()
    device = torch.device("cuda:0")
    prompt_model = prompt_model.to(device)


    loss_func = torch.nn.CrossEntropyLoss()


    no_decay = ["bias", "LayerNorm.weight", "raw_embedding"]
    optimizer_grouped_parameters = [
    {
        "params": [p for n, p in prompt_model.template.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
    }
    ]

    tot_step = 400
    lr = 7e-4
    weight_decay = 5e-8

    optimizer = PagedLion8bit(optimizer_grouped_parameters, lr=lr,weight_decay = weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=200, num_training_steps=400)
    parameters_loss = [p for group in optimizer_grouped_parameters for p in group['params']]

    use_cuda = True
    best_ht1 = 0.0000
    best_ht5 = 0.0000
    best_nd5 = 0.0000
    ht1 = 0.0000
    ht5 = 0.0000
    nd5 = 0.0000
    gradient_accumulation_steps = 4
    eval_every_steps = 2
    tot_loss = 0
    log_loss = 0
    best_val_acc = 0
    glb_step = 0
    actual_step = 0
    leave_training = False
    total_epoch = 400
    tot_train_time = 0
    pbar_update_freq = 50
    prompt_model.train()

    #torch.cuda.empty_cache()
    for epoch in tqdm(range(total_epoch)):
        tot_loss = 0
        ht1 = 0.0000
        ht5 = 0.0000
        nd5 = 0.0000
        nd5_val = 0.0000
        ht5_val = 0.0000
        ht1_val = 0.0000

        #torch.cuda.empty_cache()
        for (step1, inputs1) in enumerate(LSR_train_dataloader):
            if use_cuda:
                inputs1 = inputs1.cuda()
            logits1 = prompt_model(inputs1)
            labels1 = inputs1['label']
            nd5, ht5, ht1 = calculate_metrics(logits1, labels1, nd5, ht5, ht1)
            loss1 = loss_func(logits1, labels1)
            loss1.backward()
            tot_loss += loss1.item()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if step1 %100 ==1:
                print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step1+1)), flush=True)
            if step1 % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 2.0)
            if actual_step % gradient_accumulation_steps == 0 and glb_step > 0 and glb_step % eval_every_steps == 0:
                nd5_val, ht5_val, ht1_val = evaluate(prompt_model, LSR_val_dataloader)
                nd5_val, ht5_val, ht1_val = evaluate(prompt_model, LSR_val_dataloader, nd5_val, ht5_val, ht1_val,True)
                if ht5 > best_ht5 and ht1 > best_ht1 and nd5 > best_nd5 and ht1_val > best_ht1 and ht5_val >best_ht5 and nd5_val > best_nd5:
                    best_ht5 = ht5
                    best_ht1 = ht1
                    best_nd5 = nd5
                    torch.save(prompt_model.state_dict(),"./model.ckpt")
                    print("Save Success!")
                prompt_model.train()
        print(f"ht1_{ht1:.4f}_ht5_{ht5:.4f}_nd5{nd5:.4f}_epoch_{epoch}_lr{lr}_wd{weight_decay}_t5xl.ckpt")
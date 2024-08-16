from transformers import get_linear_schedule_with_warmup
from bitsandbytes.optim import PagedLion8bit
from ..MTL.MTL import dynamic_loss_weighting
from ..utils import calculate_metrics
from openprompt.plms import load_plm
from tqdm import tqdm
import torch
from openprompt import PromptForClassification
from ..utils import evaluate, creat_Verbalizer
from DELRec.distill_pattern_from_conventional_SR_models.temporal_analysis import load_TA_dataset, load_TA_prompt
from DELRec.distill_pattern_from_conventional_SR_models.recommendation_pattern_simulating import load_RPS_dataset, \
    load_RPS_prompt
from llms_based_sequential_recommendation import load_LSR_prompt, load_LSR_dataset
from peft import AdaLoraConfig, get_peft_model


def training_of_second_stage(args, learned_soft_prompt_path):
    plm, tokenizer, model_config, WrapperClass = load_plm(args.llm, args.llm_path)

    LSR_train_dataloader, LSR_test_dataloader, LSR_val_dataloader = load_LSR_dataset(args)
    LSR_template = load_LSR_prompt(args)

    if args.second_if_peft:
        prompt_model = PromptForClassification(plm=plm, template=LSR_template, verbalizer=creat_Verbalizer(tokenizer),
                                               freeze_plm=True)
        load_dir = learned_soft_prompt_path
        prompt_model.load_state_dict(torch.load(load_dir, map_location=args.device))
        for name, param in prompt_model.named_parameters():
            if ('soft' in name):
                param.requires_grad = False
        adalora_config = AdaLoraConfig(peft_type=args.second_peft_type, init_r=args.second_init_r,
                                       lora_alpha=args.second_lora_alpha, lora_dropout=args.second_lora_dropout,
                                       target_modules=args.second_target_modules)
        prompt_model = get_peft_model(prompt_model, peft_config=adalora_config)
    else:
        prompt_model = PromptForClassification(plm=plm, template=LSR_template, verbalizer=creat_Verbalizer(tokenizer),
                                               freeze_plm=False)
        load_dir = learned_soft_prompt_path
        prompt_model.load_state_dict(torch.load(load_dir, map_location=args.device))
        for name, param in prompt_model.named_parameters():
            if ('soft' in name):
                param.requires_grad = False

    if args.parallelize:
        prompt_model.parallelize()
    if args.device == "cuda":
        device = torch.device(args.device)
        prompt_model = prompt_model.to(device)
        use_cuda = True
    else:
        use_cuda = False

    loss_func = torch.nn.CrossEntropyLoss()

    no_decay = ["bias", "LayerNorm.weight", "raw_embedding"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in prompt_model.template.named_parameters() if
                       (not any(nd in n for nd in no_decay)) and p.requires_grad],
        }
    ]

    optimizer = PagedLion8bit(optimizer_grouped_parameters, lr=args.second_lr, weight_decay=args.second_weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.second_num_warmup_steps,
                                                num_training_steps=args.second_num_training_steps)

    best_ht1 = 0.0000
    best_ht5 = 0.0000
    best_nd5 = 0.0000
    gradient_accumulation_steps = args.second_gradient_accumulation_steps
    eval_every_steps = args.second_eval_every_steps
    glb_step = 0
    actual_step = 0
    prompt_model.train()

    for epoch in tqdm(range(args.second_total_epoch)):
        tot_loss = 0
        ht1 = 0.0000
        ht5 = 0.0000
        nd5 = 0.0000

        for (step1, inputs1) in enumerate(LSR_train_dataloader):
            if use_cuda:
                inputs1 = inputs1.cuda()
            logits1 = prompt_model(inputs1)
            labels1 = inputs1['label']
            nd5, ht5, ht1 = calculate_metrics(logits1, labels1, nd5, ht5, ht1)
            loss = loss_func(logits1, labels1)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            actual_step += 1
            if step1 % 100 == 1:
                print("Epoch {}, average loss: {}".format(epoch, tot_loss / (step1 + 1)), flush=True)
            if actual_step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 2.0)
                glb_step += 1
            if actual_step % gradient_accumulation_steps == 0 and glb_step > 0 and glb_step % eval_every_steps == 0:
                nd5_val, ht5_val, ht1_val = evaluate(prompt_model, LSR_val_dataloader)
                if ht5 > best_ht5 and ht1 > best_ht1 and nd5 > best_nd5 and ht1_val > best_ht1 and ht5_val > best_ht5 and nd5_val > best_nd5:
                    best_ht5 = ht5
                    best_ht1 = ht1
                    best_nd5 = nd5
                    torch.save(prompt_model.state_dict(), args.second_model_path)
                    print("Save Success!")
                prompt_model.train()
        print(
            f"ht1: {ht1:.4f} ht5: {ht5:.4f} nd5: {nd5:.4f} epoch: {epoch} lr: {args.second_lr} wd: {args.second_weight_decay}")

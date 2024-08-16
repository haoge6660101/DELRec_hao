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


def training_of_first_stage(args):
    plm, tokenizer, model_config, WrapperClass = load_plm(args.llm, args.llm_path)
    TA_train_dataloader, TA_test_dataloader, TA_val_dataloader = load_TA_dataset(args)
    TA_template = load_TA_prompt(args)

    RPS_train_dataloader, RPS_test_dataloader, RPS_val_dataloader = load_RPS_dataset(args)
    RPS_template = load_RPS_prompt(args)

    prompt_model = PromptForClassification(plm=plm, template=[TA_template, RPS_template],
                                           verbalizer=creat_Verbalizer(tokenizer), freeze_plm=True)

    load_model = args.load_soft_prompt_log

    if load_model == True:
        load_dir = args.first_learned_soft_prompt_path
        prompt_model.load_state_dict(torch.load(load_dir, map_location=args.device))

    if args.parallelize:
        prompt_model.parallelize()
    if args.device == "cuda":
        device = torch.device(args.device)
        prompt_model = prompt_model.to(device)
        use_cuda = True
    else:
        use_cuda = False

    loss_func = torch.nn.CrossEntropyLoss()

    optimizer_grouped_parameters = [{'params': [p for name, p in prompt_model.template.named_parameters() if
                                                'raw_embedding' not in name and p.requires_grad]}]

    optimizer = PagedLion8bit(optimizer_grouped_parameters, lr=args.first_lr, weight_decay=args.first_weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.first_num_warmup_steps,
                                                num_training_steps=args.first_num_training_steps)
    parameters_loss = [p for group in optimizer_grouped_parameters for p in group['params']]

    best_ht1 = 0.0000
    best_ht5 = 0.0000
    best_nd5 = 0.0000
    gradient_accumulation_steps = args.first_gradient_accumulation_steps
    eval_every_steps = args.first_eval_every_steps
    glb_step = 0
    actual_step = 0
    soft_prompt_path = ""
    prompt_model.train()

    for epoch in tqdm(range(args.first_total_epoch)):
        tot_loss = 0
        ht1 = 0.0000
        ht5 = 0.0000
        nd5 = 0.0000

        # torch.cuda.empty_cache()
        for (step1, inputs1), (step2, inputs2) in zip(enumerate(TA_train_dataloader), enumerate(RPS_train_dataloader)):
            if use_cuda:
                inputs1 = inputs1.cuda()
                inputs2 = inputs2.cuda()
            logits1 = prompt_model(inputs1)
            logits2 = prompt_model(inputs2)
            labels1 = inputs1['label']
            labels2 = inputs2['label']
            nd5, ht5, ht1 = calculate_metrics(logits1, labels1, nd5, ht5, ht1)
            nd5, ht5, ht1 = calculate_metrics(logits2, labels2, nd5, ht5, ht1)
            loss1 = loss_func(logits1, labels1)
            loss2 = loss_func(logits2, labels2)
            loss = dynamic_loss_weighting(loss1, loss2, parameters_loss)
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
                nd5_val, ht5_val, ht1_val = evaluate(prompt_model, TA_val_dataloader)
                nd5_val, ht5_val, ht1_val = evaluate(prompt_model, RPS_val_dataloader, nd5_val, ht5_val, ht1_val, True)
                if ht5 > best_ht5 and ht1 > best_ht1 and nd5 > best_nd5 and ht1_val > best_ht1 and ht5_val > best_ht5 and nd5_val > best_nd5:
                    best_ht5 = ht5
                    best_ht1 = ht1
                    best_nd5 = nd5
                    torch.save(prompt_model.state_dict(), args.first_learned_soft_prompt_path)
                    soft_prompt_path = args.first_learned_soft_prompt_path
                    print("Save Success!")
                prompt_model.train()
        print(
            f"ht1: {ht1:.4f} ht5: {ht5:.4f} nd5: {nd5:.4f} epoch: {epoch} lr: {args.first_lr} wd: {args.first_weight_decay}")
    return soft_prompt_path

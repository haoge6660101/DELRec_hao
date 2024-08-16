from DELRec.utils import calculate_metrics
from openprompt.plms import load_plm
from tqdm import tqdm
import torch
from openprompt import PromptForClassification
from DELRec.utils import evaluate, creat_Verbalizer
from DELRec.llms_based_sr.llms_based_sequential_recommendation import load_LSR_prompt, load_LSR_dataset
from peft import AdaLoraConfig, get_peft_model


def test(args):
    plm, tokenizer, model_config, WrapperClass = load_plm(args.llm, args.llm_path)

    TES_train_dataloader, TES_test_dataloader, TES_val_dataloader = load_LSR_dataset(args)
    TES_template = load_LSR_prompt(args)

    if args.second_if_peft:
        prompt_model = PromptForClassification(plm=plm, template=TES_template, verbalizer=creat_Verbalizer(tokenizer),
                                               freeze_plm=True)
        adalora_config = AdaLoraConfig(peft_type=args.second_peft_type, init_r=args.second_init_r,
                                       lora_alpha=args.second_lora_alpha, lora_dropout=args.second_lora_dropout,
                                       target_modules=args.second_target_modules)
        prompt_model = get_peft_model(prompt_model, peft_config=adalora_config)
        load_dir = args.second_model_path
        prompt_model.load_state_dict(torch.load(load_dir, map_location=args.device))
        for name, param in prompt_model.named_parameters():
            if ('soft' in name):
                param.requires_grad = False

    else:
        prompt_model = PromptForClassification(plm=plm, template=TES_template, verbalizer=creat_Verbalizer(tokenizer),
                                               freeze_plm=True)
        load_dir = args.second_model_path
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
    prompt_model.eval()

    for (step1, inputs1) in tqdm(enumerate(TES_test_dataloader)):
        ht1 = 0.0000
        ht5 = 0.0000
        nd5 = 0.0000
        if use_cuda:
            inputs1 = inputs1.cuda()
        logits1 = prompt_model(inputs1)
        labels1 = inputs1['label']
        nd5, ht5, ht1 = calculate_metrics(logits1, labels1, nd5, ht5, ht1)
        print(f"ht1: {ht1:.4f} ht5: {ht5:.4f} nd5: {nd5:.4f}")

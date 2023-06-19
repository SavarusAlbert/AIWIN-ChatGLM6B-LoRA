# 使用AutoDL云GPU平台
# 加速端口
# !export http_proxy=http://100.72.64.19:12798 && export https_proxy=http://100.72.64.19:12798
# 关闭加速
# !unset http_proxy && unset https_proxy

# 安装工具包
# !pip install peft -i https://pypi.tuna.tsinghua.edu.cn/simple
# !pip install cpm_kernels -i https://pypi.tuna.tsinghua.edu.cn/simple
# !pip install icetk -i https://pypi.tuna.tsinghua.edu.cn/simple
# !pip install -U synonyms

# !git clone https://github.com/THUDM/ChatGLM-6B.git
# !cd ChatGLM-6B
# !pip install -r requirements.txt
# !cd ..

# !nvidia-smi


import os
import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel, AutoConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset


device = torch.device('cuda:0')

def load_premodel():
    """
    读取chatglm-6b模型，首次需要下载模型
    checkpoint = "THUDM/chatglm-6b"
    version = "a10da4c68b5d616030d3531fc37a13bb44ea814d"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, version=version, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, version=version, trust_remote_code=True)
    """
    checkpoint = "/root/.cache/huggingface/hub/models--THUDM--chatglm-6b/snapshots/a10da4c68b5d616030d3531fc37a13bb44ea814d"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)

    return tokenizer, model


def load_train_data(train_file):
    # 读取训练集数据
    train_data = []
    for line in open(train_file):
        train_data.append(json.loads(line))

    return train_data


def load_lora_config(model):
    # 设定 lora 微调参数，r=16层可训练参数
	config = LoraConfig(
	    task_type=TaskType.CAUSAL_LM, 
	    inference_mode=False,
	    r=8, 
	    lora_alpha=32, 
	    lora_dropout=0,
	    target_modules=["query_key_value"]
	)
        
	return get_peft_model(model, config)


def get_spicial_tokens(tokenizer):
    # 输出特殊 token 的编码
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    mask = tokenizer.mask_token_id
    gmask = tokenizer.gmask_token_id

    return bos, eos, pad, mask, gmask


def src_maxlength(train_data):
    # 问题的最长 token 长度
    input_len_list = [len(data['input']) for data in train_data]
    max_data_idx = input_len_list.index(max(input_len_list))
    prompt = train_data[max_data_idx]['input']
    prompt_ids = tokenizer.encode(prompt, max_length=1024, truncation=True)
    max_src_length = len(prompt_ids)
    
    # 答案的最长 token 长度
    ans_len_list = [len(data['answer']) for data in train_data]
    max_ans_idx = ans_len_list.index(max(ans_len_list))
    completion = train_data[max_ans_idx]['answer']
    completion_ids = tokenizer.encode(completion, max_length=1024, truncation=True)
    max_com_length = len(completion_ids)

    return max_src_length, max_com_length


def create_prompt(question):
    # 构造问答数据格式
    PROMPT_PATTERN = "问：{}"
    SEP_PATTERN = "\n答： "

    return PROMPT_PATTERN.format(question), SEP_PATTERN


def create_prompt_ids(tokenizer, question, max_src_length):
    # 构造问答的 token 编码序列，
    prompt, sep = create_prompt(question)
    sep_ids = tokenizer.encode(
        sep, 
        add_special_tokens = True
    )
    sep_len = len(sep_ids)
    special_tokens_num = 2
    prompt_ids = tokenizer.encode(
        prompt, 
        max_length = max_src_length - (sep_len - special_tokens_num),
        truncation = True,
        add_special_tokens = False
    )

    return prompt_ids + sep_ids


def create_inputs_and_labels(tokenizer, question, answer, device):
    # 构造 inputs 和 labels 编码数据
    prompt = create_prompt_ids(tokenizer, question, max_src_length)
    completion = tokenizer.encode(
        answer, 
        max_length = max_com_length,
        truncation = True,
        add_special_tokens = False
    )

    inputs = prompt + completion + [eos]
    labels = [-100] * len(prompt) + completion + [eos]
    
    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    return inputs, labels


def get_attention_mask(tokenizer, input_ids, device):
    # 构建 Attention Mask
    seq = input_ids.tolist()
    context_len = seq.index(bos)
    seq_len = len(seq)    
    attention_mask = torch.ones((seq_len, seq_len), device=device)
    attention_mask.tril_()
    attention_mask[..., :context_len] = 1
    attention_mask.unsqueeze_(0)
    attention_mask = (attention_mask < 0.5).bool()
    return attention_mask


def get_position_ids(tokenizer, input_ids, device, position_encoding_2d=True):
    # 构建 Position IDs
    seq = input_ids.tolist()
    context_len = seq.index(bos)
    seq_len = len(seq)

    mask_token = mask if mask in seq else gmask
    use_gmask = False if mask in seq else gmask

    mask_position = seq.index(mask_token)

    if position_encoding_2d:
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        if not use_gmask:
            position_ids[context_len:] = mask_position
        block_position_ids = torch.cat((
            torch.zeros(context_len, dtype=torch.long, device=device),
            torch.arange(seq_len - context_len, dtype=torch.long, device=device) + 1
        ))
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        if not use_gmask:
            position_ids[context_len:] = mask_position
    
    return position_ids


class QADataset(Dataset):
    """
    创建QA基类，构建带有掩码token的token数据
    """
    def __init__(self, data, tokenizer) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
 

    def __getitem__(self, index):
        item_data = self.data[index]
        tokenizer = self.tokenizer
        input_ids, labels = create_inputs_and_labels(
            tokenizer=tokenizer, 
            device=device,
            question=item_data["input"],
            answer=item_data["answer"],
        )

        attention_mask = get_attention_mask(tokenizer, input_ids, device)
        position_ids = get_position_ids(tokenizer, input_ids, device)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }
        

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    """
    创建Data Collator
    """
    input_ids = []
    attention_mask = []
    labels = []
    position_ids = []
    
    for obj in batch:
        input_ids.append(obj['input_ids'])
        labels.append(obj['labels'])
        attention_mask.append(obj['attention_mask'])
        position_ids.append(obj['position_ids'])
        
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask), 
        'labels': torch.stack(labels),
        'position_ids':torch.stack(position_ids)
    }



class ModifiedTrainer(Trainer):
    """
    构建Trainer子类，定义loss function
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            labels=inputs["labels"],
        ).loss


def save_tuned_parameters(model, path):
    # 保存模型
    saved_params = {
        k: v.to(device)
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)




if __name__ == "__main__":
    # 读取训练集数据
    train_file = 'train.json'
    train_data = load_train_data(train_file)

    # 读取模型
    tokenizer, model = load_premodel()
    model = load_lora_config(model)
    model.to(device)

    # 特殊token的编码
    bos, eos, pad, mask, gmask = get_spicial_tokens(tokenizer)
    # 问答文本最长长度
    max_src_length, max_com_length = src_maxlength(train_data)

    
    # 检查点输出路径
    output_dir = "/root/autodl-tmp"

    # 配置训练参数
    training_args = TrainingArguments(
        output_dir,
        fp16 =True,
        save_steps = 1000,
        save_total_limit = 10,
        gradient_accumulation_steps = 16,
        per_device_train_batch_size = 1,
        learning_rate = 1e-3,
        max_steps=10000,
        logging_steps=100,
        remove_unused_columns=False,
        seed=500,
        data_seed=500,
        group_by_length=False,
        dataloader_pin_memory=False
    )

    # from transformers import DataCollatorWithPadding
    # A = DataCollatorWithPadding(tokenizer, padding="max_length")
    
    # 配置QADataset和Trainer实例
    train_dataset = QADataset(train_data, tokenizer=tokenizer)
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collate_fn,
        tokenizer=tokenizer
    )

    # 模型训练
    trainer.train()
    # 由最后一个检查点加载模型
    # trainer.train(resume_from_checkpoint=True)

    # 模型参数保存
    # save_tuned_parameters(model, os.path.join("/root/autodl-tmp", "chatglm-6b-lora.pt"))


    # 重载保存的模型参数
    # tokenizer, model = load_premodel()
    # model = load_lora_config(model)
    # model.load_state_dict(torch.load(f"/root/autodl-tmp/chatglm-6b-lora.pt"), strict=False)

    # 结果测试
    # model.half().cuda().eval()
    # response, history = model.chat(tokenizer, "在卫健委信用信息网_护的网页中操作，请查询北京中医医院的护士信息", history=[])
    # print(response)
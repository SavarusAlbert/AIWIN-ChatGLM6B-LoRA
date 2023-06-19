import os
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModel, AutoConfig


device = torch.device('cuda:0')


def load_premodel():
    """
    读取本地chatglm-6b模型
    """
    checkpoint = "/root/.cache/huggingface/hub/models--THUDM--chatglm-6b/snapshots/a10da4c68b5d616030d3531fc37a13bb44ea814d"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)

    return tokenizer, model

def load_lora_config(model):
	config = LoraConfig(
	    task_type=TaskType.CAUSAL_LM, 
	    inference_mode=False,
	    r=8, 
	    lora_alpha=32, 
	    lora_dropout=0,
	    target_modules=["query_key_value"]
	)
	return get_peft_model(model, config)


def save_tuned_parameters(model, path):
    saved_params = {
        k: v.to(device)
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)



if __name__ == "__main__":
    # 读取模型，配置 lora 层参数，保持与训练时的参数一致
    tokenizer, model = load_premodel()
    model = load_lora_config(model)

    # 读取检查点模型参数
    checkpoint = f"./checkpoint-10000/pytorch_model.bin"
    model.load_state_dict(torch.load(checkpoint))

    # 保存模型参数
    save_tuned_parameters(model, os.path.join("/root/autodl-tmp", "chatglm-6b-lora-cp10000.pt"))

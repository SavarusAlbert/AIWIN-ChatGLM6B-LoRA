import torch
import numpy as np
import json
import re
import jieba
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModel, AutoConfig
from nltk.translate.bleu_score import sentence_bleu
            



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


def validation(mini_label_devset_len):
     # 读取验证集
    label_devset = []
    for line in open('dev.json'):
        label_devset.append(json.loads(line))
    average_len = int(np.round(mini_label_devset_len / 20))
    mini_label_devset = []
    for i in range(20):   
        mini_label_devset += label_devset[24*i:24*i+average_len]

    # 模型预测
    predict_score = []
    for input, answer in zip([mini_label_devset[i]["input"] for i in range(mini_label_devset_len)], [mini_label_devset[i]["answer"] for i in range(mini_label_devset_len)]):
        response, history = model.chat(tokenizer, input, history=[])

        # response文本列表
        res_text_iter = re.split('第\d步：', response)
        res_text_list = [re.sub(':|;|"|，|,', '', text) for text in res_text_iter[1:]]
        res_len = len(res_text_list)

        # ans文本列表
        ans_text_iter = re.split('第\d步:', answer)
        ans_text_list = [re.sub(':|;|"|，|,', '', text) for text in ans_text_iter[1:]]
        ans_len = len(ans_text_list)

        score_list = [0] * max(res_len, ans_len)
        for i in range(min(res_len, ans_len)):
            reference = [jieba.lcut(ans_text_list[i])]
            candidate = jieba.lcut(res_text_list[i])
            score = sentence_bleu(reference, candidate)
            score = np.around(score, 2)
            score_list[i] = score
        
        predict_score.append(np.mean(score_list))

    
    print("分数列表为:", predict_score)

    accuracy_rate1 = len([i for i in predict_score if i >= 0.8]) / len(predict_score)
    print("80分正确率为:", accuracy_rate1)

    accuracy_rate2 = len([i for i in predict_score if i >= 0.7]) / len(predict_score)
    print("70分正确率为:", accuracy_rate2)

    error_rate = len([i for i in predict_score if i <= 0.3]) / len(predict_score)
    print("30分以下为:", error_rate)




if __name__ == "__main__":
    # 读取模型，配置 lora 层参数，保持与训练时的参数一致
    tokenizer, model = load_premodel()
    model = load_lora_config(model)

    # 方式1：读取检查点模型参数
    checkpoint = f"./checkpoint-10000/pytorch_model.bin"
    model.load_state_dict(torch.load(checkpoint))

    # 方式2：读取Save.py文件保存的模型参数
    # model.load_state_dict(torch.load(f"chatglm-6b-lora-cp10000.pt"), strict=False)

    # 模型推理模式
    model = model.half().cuda().eval()

    # 在验证集的部分数据上进行评估，取20 * (mini_label_devset_len // 20) 个验证集数据进行评估
    validation(mini_label_devset_len=40)
import torch
import json
import re
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModel, AutoConfig



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



if __name__ == "__main__":
    # 读取模型，配置 lora 层参数，保持与训练时的参数一致
    tokenizer, model = load_premodel()
    model = load_lora_config(model)

    # 读取检查点模型参数
    checkpoint = f"./checkpoint-10000/pytorch_model.bin"
    model.load_state_dict(torch.load(checkpoint))

    # 模型推理模式
    model = model.half().cuda().eval()


    # 读取测试集
    instruction_testB_with_node =  json.load(open('instruction_testB_with_node.json', encoding="utf-8"))


    # 模型预测
    submit_prediction = []
    for idx in tqdm(range(20)):
    # for idx in tqdm(range(len(label_trainset))):
        page_source = instruction_testB_with_node[idx]['page_source']
        
        page_predictions = []
        for instruction in instruction_testB_with_node[idx]["instruction"]:
            response, history = model.chat(tokenizer, f'在{instruction_testB_with_node[idx]["page_source"]}的网页中操作，{instruction}', history=[])
            
            # 将结果输出为对应的格式
            instruction_result = {'instruction': instruction}
            instruction_result['key-value'] = response

            page_predictions.append(instruction_result)

        submit_prediction.append({
            'page_source': page_source,
            'instruction_detail': page_predictions
        })

    # 保存中间结果
    with open('submit_prediction.json', 'w') as up:
        json.dump(submit_prediction, up)



    # 修改格式
    submission = []
    for idx in range(20):
        page_source = submit_prediction[idx]['page_source']
        
        page_predictions = []
        for instruction in submit_prediction[idx]['instruction_detail']:
            # 将结果输出为对应的格式
            instruction_result = {'instruction': instruction['instruction']}
            key_value_result = dict()
            kv = re.split('"', instruction['key-value'])
            kv_len = len(kv)
            for kv_idx in range(kv_len-1):
                kvpair = kv[kv_idx+1]
                if kvpair == "点击" and kv_idx+5<kv_len-1:
                    key_value_result[kv[kv_idx+3]] = {'dom_type': kv[kv_idx+5], 'value': "", 'action': "点击"}
                if kvpair == "输入" and kv_idx+3<kv_len-1:
                    key_value_result[kv[kv_idx-3]] = {'dom_type': kv[kv_idx-1], 'value': kv[kv_idx+3], 'action': "输入"}
            instruction_result['key-value'] = key_value_result

            page_predictions.append(instruction_result)

        submission.append({
            'page_source': page_source,
            'instruction_detail': page_predictions
        })

    # 保存结果
    with open('submission.json', 'w') as up:
        json.dump(submission, up)
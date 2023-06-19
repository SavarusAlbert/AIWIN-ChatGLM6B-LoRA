import torch
import re
import copy
import random
import json
import synonyms
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from nlpcda import Similarword, RandomDeleteChar, CharPositionExchange
# from nlpcda import Simbert


device = torch.device('cuda:0')


def dataset_generator(label_trainset):
    """
    生成训练数据和验证数据
    """
    train_set = []
    dev_set = []

    for idx in range(len(label_trainset)):
        train_dict = {'page_source': f'{label_trainset[idx]["page_source"]}',}
        dev_dict = {'page_source': f'{label_trainset[idx]["page_source"]}',}
        
        random.shuffle(label_trainset[idx]['instruction_detail'])

        train_dict['instruction_detail'] = label_trainset[idx]['instruction_detail'][:-24]
        dev_dict['instruction_detail'] = label_trainset[idx]['instruction_detail'][-24:]

        train_set.append(train_dict)
        dev_set.append(dev_dict)

    return train_set, dev_set
    

def loadmodel_eval():
    """
    读取chatglm-6b模型，切换评估模式
    """
    # checkpoint = "THUDM/chatglm-6b"
    checkpoint = "/root/.cache/huggingface/hub/models--THUDM--chatglm-6b/snapshots/a10da4c68b5d616030d3531fc37a13bb44ea814d"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)

    model = model.half().cuda().eval()
    
    return tokenizer, model


def ask_synonyms(keyword, syn_num):
    """
    对instruction问题中的输入词语keyword进行同义词替换
    """
    synonym_word = synonyms.nearby(keyword, syn_num+2)
    res = []
    for i in range(len(synonym_word[1])):
        if synonym_word[1] != 1:
            res.append(synonym_word[0][i])
    
    return res[:syn_num]


def data_generator(instructions, label, keyword, res):
    """
    生成同义词替换后的数据集
    """
    train_data = []
    for word in res:
        newdata = dict()

        instructions_data = instructions['instruction']
        instructions_data = re.sub(keyword, word, instructions_data)

        label_dict = copy.deepcopy(label['key-value'])
        for value in label_dict.values():
            if value['value'] == keyword:
                value['value'] = word

        newdata["instruction"] = instructions_data
        newdata["key-value"] = label_dict
        
        train_data.append(newdata)
    return train_data


def train_dataaug(label_trainset, model, tokenizer, syn_num):
    """
    运行模型生成新数据，保存结果
    """
    new_train_data = list()
    
    for idx in tqdm(range(20)):
        new_train_data_dict = dict()
        new_train_data_list = list()

        new_train_data_dict['page_source'] = label_trainset[idx]['page_source']
        for instructions, label in zip(label_trainset[idx]['instruction_detail'], label_trainset[idx]['instruction_detail']):
            for i, (key, value) in enumerate(label['key-value'].items()):
                if value['action'] == '输入':
                    res = ask_synonyms(value['value'], syn_num)
                    train_data = data_generator(instructions, label, value['value'], res)
                    new_train_data_list += train_data
        new_train_data_dict['instruction_detail'] = new_train_data_list
        new_train_data.append(new_train_data_dict)
        
        # with open(f'new_train_data{idx}.json', 'w') as up:
        #     for line in new_train_data:
        #         up.write(json.dumps(line)+'\n')
    
    return new_train_data


def cda_instruction(label_trainset, sim_num, del_num, exc_num):
    """
    instruction字段数据增强
    """
    new_train_data = list()

    smw_sim = Similarword(create_num=sim_num+2)
    smw_del = RandomDeleteChar(create_num=del_num+2)
    smw_exc = CharPositionExchange(create_num=exc_num+2, char_gram=2)

    
    for idx in tqdm(range(20)):
        new_train_data_dict = dict()
        cda_data_list = list()

        new_train_data_dict['page_source'] = label_trainset[idx]['page_source']
        for instructions, label in zip(label_trainset[idx]['instruction_detail'], label_trainset[idx]['instruction_detail']):
            sen = instructions['instruction']
            
            # 近义词替换
            rs1 = smw_sim.replace(sen)
            if len(rs1) == sim_num+2:
                for _ in range(sim_num):
                    rs1_data = {'instruction': rs1[_+1], 'key-value': instructions['key-value']}
                    cda_data_list.append(rs1_data)


            # 随机字删除
            rs2 = smw_del.replace(sen)
            if len(rs1) == del_num+2:
                for _ in range(del_num):
                    rs2_data = {'instruction': rs2[_+1], 'key-value': instructions['key-value']}
                    cda_data_list.append(rs2_data)


            # 随机置换临近字
            rs3 = smw_exc.replace(sen)
            if len(rs1) == exc_num+2:
                for _ in range(exc_num):
                    rs3_data = {'instruction': rs3[_+1], 'key-value': instructions['key-value']}
                    cda_data_list.append(rs3_data)


            # simbert
            # config = {
            #     'model_path': 'chinese_simbert_L-12_H-768_A-12',
            #     'device': 'cuda:0',
            #     'max_len': 32,
            #     'seed': 1
            #     }
            # simbert = Simbert(config=config)
            # synonyms = simbert.replace(sent=sen, create_num=5)


        new_train_data_dict['instruction_detail'] = cda_data_list
        new_train_data.append(new_train_data_dict)
        
        # with open(f'new_train_data{idx}.json', 'w') as up:
        #     for line in new_train_data:
        #         up.write(json.dumps(line)+'\n')
    
    return new_train_data


def merge_data(data_A, data_B):
    # 按照data_A的格式合并数据集data_B
    dataset = []
    for data_a in data_A:
        data_temp = copy.deepcopy(data_a)
        for data_b in data_B:
            if data_b["page_source"] == data_a["page_source"]:
                data_temp["instruction_detail"] += data_b["instruction_detail"]
        dataset.append(data_temp)
    return dataset


def translation_zh_en_zh(text, tokenizer_zh_en, model_zh_en, tokenizer_en_zh, model_en_zh):
    """
    对文本text进行中英回译，如需其他语言可以下载相应模型
    """
    # 中译英词向量编码及翻译
    tokenized_text_zh_en = tokenizer_zh_en([text], return_tensors='pt')
    translation_zh_en = model_zh_en.generate(**tokenized_text_zh_en)
    translated_text_zh_en = tokenizer_zh_en.batch_decode(translation_zh_en, skip_special_tokens=True)
    
    # 英译中词向量编码及翻译
    tokenized_text_en_zh = tokenizer_en_zh(translated_text_zh_en, return_tensors='pt')
    translation_en_zh = model_en_zh.generate(**tokenized_text_en_zh)
    translated_text_en_zh = tokenizer_en_zh.batch_decode(translation_en_zh, skip_special_tokens=True)
    return translated_text_en_zh[0]


def back_translation(data):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    # 中译英模型
    tran_zh_en_checkpoint = "/root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-zh-en/snapshots/820b01075a2e1c1f575e83c1ecc9d43d589195e8"
    tokenizer_zh_en = AutoTokenizer.from_pretrained(tran_zh_en_checkpoint)
    model_zh_en = AutoModelForSeq2SeqLM.from_pretrained(tran_zh_en_checkpoint)
    model_zh_en.eval()

    # 英译中模型
    tran_en_zh_checkpoint = "/root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-en-zh/snapshots/a4193836671069f1f80ce341f9227a850ffb52d4"
    tokenizer_en_zh = AutoTokenizer.from_pretrained(tran_en_zh_checkpoint)
    model_en_zh = AutoModelForSeq2SeqLM.from_pretrained(tran_en_zh_checkpoint)
    model_en_zh.eval()

    # 对data的'instruction'进行回译
    bktrans_data = copy.deepcopy(data)
    for idx in tqdm(range(len(data))):
        for datapair in data[idx]["instruction_detail"]:
            text = datapair["instruction"]
            bktrans_text = translation_zh_en_zh(text, tokenizer_zh_en, model_zh_en, tokenizer_en_zh, model_en_zh)

            bktrans_instruction = dict()
            bktrans_instruction["instruction"] = bktrans_text
            bktrans_instruction["key-value"] = datapair["key-value"]

            bktrans_data[idx]["instruction_detail"].append(bktrans_instruction)

    return bktrans_data



def trainset_generator(label_trainset, file_name):
    """
    生成训练集和验证集
    """
    chat_data = []
    for idx in range(20):
        for label in label_trainset[idx]['instruction_detail']:
            temp = f'在{label_trainset[idx]["page_source"]}的网页中操作，'
            for i, (key, value) in enumerate(label['key-value'].items()):
                temp += f'第{i+1}步:'
                if value["value"]:
                    temp += ''.join(['在', f'"{key}"', '的', f'"{value["dom_type"]}"', '中', f'"{value["action"]}"', f'"{value["value"]}"'])
                else:
                    temp += ''.join([f'"{value["action"]}"', f'"{key}"', '的', f'"{value["dom_type"]}"'])
                temp += ';'
            temp = temp[:-1]
            chat_data.append({
                'input': f'在{label_trainset[idx]["page_source"]}的网页中操作，{label["instruction"]}',
                'answer': temp,
                "history": []
            })
    
    # 随机打乱
    random.shuffle(chat_data)
    
    with open(file_name, 'w') as fn:
        for line in chat_data:
            fn.write(json.dumps(line)+'\n')


def data_aug(label_trainset, syn_num=0, sim_num=0, del_num=0, exc_num=0, bkt_num=False):
    """
    label_trainset为待增广数据集，syn_num是"输入"数据增广同义词数，sim_num是近义词替换词数，del_num是随机删除词数，exc_num是随机置换词数，bkt_num表示是否进行回译
    """
    # 生成训练数据和验证数据
    train_set, dev_set = dataset_generator(label_trainset)

    
    if syn_num:
        # 同义替换后的新数据集
        syn_data = train_dataaug(train_set, model=None, tokenizer=None, syn_num=syn_num)

        # 汇总所有训练数据集
        train_set = merge_data(train_set, syn_data)

    if sim_num or del_num or exc_num:
        # nlpcda数据增广
        cda_data = cda_instruction(train_set, sim_num=sim_num, del_num=del_num, exc_num=exc_num)

        # 汇总所有训练数据集
        train_set = merge_data(train_set, cda_data)

    if bkt_num:
        # 通过回译进行数据增广
        train_set = back_translation(train_set)


    # 生成训练集
    trainset_generator(train_set, "train.json")
    
    # 生成验证集
    trainset_generator(dev_set, "dev.json")


if __name__ == "__main__":
    # tokenizer, model = loadmodel_eval()
    # 读取数据集
    LABEL_PATH = "" # './2023S-T1-A-Data[供选手] 0513/指令数据&标签/'
    label_trainset = json.load(open(LABEL_PATH + 'label_trainset.json', encoding="utf-8"))
    instruction_trainset = json.load(open(LABEL_PATH + 'instruction_trainset.json', encoding="utf-8"))
    instruction_testA = json.load(open(LABEL_PATH + 'instruction_testA.json', encoding="utf-8"))

    # 进行syn_num个同义词替换"输入"数据，sin_num个近义词替换，del_num个随机删除，exc_num个随机置换，并进行回译
    # 生成训练集，验证集，并保存为 "train.json" 和 "dev.json" 文件
    data_aug(label_trainset, syn_num=1, sim_num=1, del_num=1, exc_num=1, bkt_num=True)
    

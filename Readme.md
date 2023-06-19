# ChatGLM-6B-Lora微调

- 安装相应的机器学习库
```
pip install -r requirements.txt
```

- 数据增广：
  - 运行DataAugmentation.py文件，生成训练集和验证集
  - 需要配置的参数为 `data_aug(label_trainset, syn_num=1, sim_num=1, del_num=1, exc_num=1, bkt_num=True)`
    - label_trainset: 带标签的训练集
    - syn_num: 同义词替换，生成新数据个数
    - sim_num: nlpcda近义词替换，生成新数据个数
    - del_num: nlpcda随机删除，生成新数据个数
    - exc_num: nlpcda随机置换，生成新数据个数
    - bkt_num: 是否进行回译，生成新数据


- 模型训练：
  - 运行Train.py文件，单卡Lora微调ChatGLM-6B


- 模型保存：
  - 运行Save.py文件，保存训练过程中的checkpoint


- 验证性能：
  - 运行Validation.py文件，在部分验证集数据上进行评估


- 输出结果：
  - 运行Predict.py文件，输出预测的标签，保存为 "submission.json"
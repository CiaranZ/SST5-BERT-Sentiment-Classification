import transformers
from transformers import BertTokenizer, BertModel, BertForSequenceClassification 
import torch
from torch import nn
from torch.utils import data
import torch.optim as optim  
from torch.utils.data import Dataset, DataLoader  
import os  
import requests  
import re  
import collections
import numpy as np
import random  
import itertools  
from collections import defaultdict  
from collections import deque
import csv  
import datasets
from datasets import Dataset
import pandas as pd  
from sklearn.model_selection import train_test_split  
import logging


text_file_path = 'stanfordSentimentTreebank/dictionary.txt'  
label_file_path = 'stanfordSentimentTreebank/sentiment_labels.txt'  

# 读取标签数据集，并创建一个字典来存储ID到标签的映射  
label_dict = {}  
with open(label_file_path, newline='') as csvfile:  
    reader = csv.reader(csvfile, delimiter='|')  
    for row in reader:  
        label_id, label_value = row  
        label_dict[label_id] = float(label_value)  
     
features = []
count = 0  # 计数器  
with open(text_file_path, newline='') as csvfile:  
    reader = csv.reader(csvfile, delimiter='|')  
    for row in reader:  
        text, text_id = row  
        if text_id in label_dict and count < 100000:  # 确保ID存在于标签字典中，并且计数器小于50000     
            label_value = label_dict[text_id]  
            # 根据label的值划分分类  
            if 0 <= label_value < 0.2:  
                category_label = 0  
            elif 0.2 <= label_value < 0.4:  
                category_label = 1  
            elif 0.4 <= label_value < 0.6:  
                category_label = 2 
            elif 0.6 <= label_value < 0.8:  
                category_label = 3  
            elif 0.8 <= label_value <= 1.0:  
                category_label = 4  
  
            # 创建一个字典，包含文本、原始标签和分类标签，并将其添加到features列表中  
            feature_dict = {'text': text,'label': category_label}  
            features.append(feature_dict)  
            count += 1  # 递增计数器  

            
first_ten_rows = features[:5]  
for row in first_ten_rows:  
   print(row)
              
# 打印特征和标签的数量
print("Number of features:", len(features))  

df = pd.DataFrame(features)  
train_df, valid_df = train_test_split(df, test_size=0.2) 
train_features = Dataset.from_pandas(train_df)  
valid_features = Dataset.from_pandas(valid_df)


#对数据进行标记化（tokenize），
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)
train_dataset = train_features.map(preprocess_function, batched=True)
valid_dataset = valid_features.map(preprocess_function, batched=True)


print("Number of train_dataset:", len(train_dataset))  
print("Number of valid_dataset:", len(valid_dataset))  


from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
valid_dataloader = DataLoader(valid_dataset, batch_size=8)


from transformers import BertForSequenceClassification, AdamW
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)


from transformers import Trainer, TrainingArguments


training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="no",
)

# 计算准确率
from sklearn.metrics import accuracy_score 
def compute_metrics(p):  
    preds = np.argmax(p.predictions, axis=1)  
    labels = p.label_ids  
    accuracy = accuracy_score(labels, preds)  
    return {"accuracy": accuracy}  

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics
)

trainer.train()


metrics = trainer.evaluate()  
print(f"Test set accuracy: {metrics['eval_accuracy']:.4f}")

#保存模型
model_path = 'model.bin'  
torch.save(model.state_dict(), model_path) 
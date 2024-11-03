SST5 数据集短文本分类任务

项目简介

本项目旨在利用 SST5（Stanford Sentiment Treebank 5-class）数据集进行短文本分类任务。通过微调 BERT（Bidirectional Encoder Representations from Transformers）模型，我们实现了对文本情感倾向的分类。项目涵盖了数据预处理、模型训练、评估和模型保存等步骤。

项目结构

.  

├── README.md                  # 项目说明文件  

├── main.py                    # 主代码文件  

├── stanfordSentimentTreebank  # 数据集文件夹  

│   ├── dictionary.txt         # 文本数据  

│   └── sentiment_labels.txt   # 标签数据  

├── requirements.txt           # 依赖库文件  

└── results                    # 训练结果输出目录  

    └── ...                    # 训练过程中生成的模型文件、日志等

依赖库

项目依赖以下 Python 库，请确保在运行代码前安装这些库。可以通过 pip install -r requirements.txt 来安装所有依赖。

transformers  
torch  
datasets  
pandas  
scikit-learn

数据预处理

读取数据集：从 stanfordSentimentTreebank 文件夹中读取 dictionary.txt 和 sentiment_labels.txt 文件。

标签映射：创建一个字典来存储标签 ID 到标签值的映射。

文本筛选：根据标签值将文本划分为 5 个类别，并筛选出前 100000 条数据。

数据划分：使用 train_test_split 将数据集划分为训练集和验证集。

模型训练

标记化：使用 BertTokenizer 对文本进行标记化。

数据加载：使用 DataLoader 加载训练集和验证集。

模型初始化：使用 BertForSequenceClassification 初始化模型，并设置分类标签数量为 5。

训练配置：设置训练参数，包括学习率、批量大小、训练轮数等。

训练过程：使用 Trainer 进行模型训练，并在每个 epoch 后进行评估。

评估与模型保存

计算准确率：使用 accuracy_score 计算验证集的准确率。

模型保存：将训练好的模型保存到 model.bin 文件中。


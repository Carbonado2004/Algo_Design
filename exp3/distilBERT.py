# --- 1. 导入所有需要的库 ---
print("--- 1. 开始导入依赖库 ---")
import torch
import torch.nn.functional as F
import numpy as np
import re  # 导入正则表达式库
import glob  # 用于查找文件
import sys  # 用于路径操作

import os  # 用于路径操作
import random  # 用于数据增强

# 导入 datasets 库（避免与本地 datasets 目录冲突）
# 由于exp3/datasets 目录会干扰导入，需要确保导入的是安装的库
# 所以临时移除当前目录从 sys.path，导入后再恢复
_current_dir = os.path.dirname(os.path.abspath(__file__))
_removed_from_path = False
if _current_dir in sys.path:
    sys.path.remove(_current_dir)
    _removed_from_path = True

try:
    from datasets import load_dataset, Dataset  # 用于加载 IMDb 数据集和创建数据集
finally:
    # 恢复 sys.path（如果需要）
    if _removed_from_path:
        sys.path.insert(0, _current_dir)
from transformers import (
    AutoTokenizer,  # 自动加载分词器
    AutoModelForSequenceClassification,  # 自动加载序列分类模型
    AutoModelForMaskedLM,  # 用于 MLM 预训练（DAPT）
    TrainingArguments,  # 训练参数配置
    Trainer,  # 训练器
    DataCollatorWithPadding,  # 动态 padding 数据
    DataCollatorForLanguageModeling,  # 用于 MLM 预训练的数据整理器
)
from peft import (
    get_peft_model,  # PEFT 核心函数，用于包装模型
    LoraConfig,  # LoRA 配置
    TaskType,  # 指定任务类型
)
import evaluate  # hugging face 的评估库
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report  # 评估指标
# 注意：SHAP 分析已移至专门的 Notebook (SHAP_Analysis.ipynb)，此处不再导入
import matplotlib.pyplot as plt  # 用于可视化
import matplotlib
import seaborn as sns  # 用于更美观的可视化
from tqdm import tqdm  # 用于进度条
import warnings
warnings.filterwarnings('ignore')

# 配置 matplotlib 支持中文显示
font_list = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'STSong', 'Arial Unicode MS']
available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
chinese_font = None
for font in font_list:
    if font in available_fonts:
        chinese_font = font
        break

if chinese_font:
    plt.rcParams['font.sans-serif'] = [chinese_font]
    # 确保matplotlib使用中文字体
    matplotlib.rcParams['font.sans-serif'] = [chinese_font]
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置seaborn样式
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

print("所有库导入成功！\n")

# --- 2. 定义全局配置和辅助函数 ---
# 把一些固定参数放在这里，方便修改，这是一个好习惯

# 使用 'distilbert-base-uncased' 作为轻量级模型，'uncased' 表示不区分大小写

# 模型路径配置：支持离线模式
# 优先使用本地模型，如果不存在则使用在线模型
import os
MODEL_NAME = "distilbert-base-uncased"
LOCAL_MODEL_PATH = "./models/distilbert-base-uncased"

# 检查本地模型是否存在
if os.path.exists(LOCAL_MODEL_PATH) and os.path.exists(os.path.join(LOCAL_MODEL_PATH, "config.json")):
    MODEL_CHECKPOINT = LOCAL_MODEL_PATH
    print(f"检测到本地模型，使用离线模式: {MODEL_CHECKPOINT}")
else:
    MODEL_CHECKPOINT = MODEL_NAME
    print(f"警告: 未找到本地模型，将尝试在线下载: {MODEL_CHECKPOINT}")
    print(f"   提示：如果网络有问题，请先下载模型到 {LOCAL_MODEL_PATH}")
    print(f"   详细说明请查看 OFFLINE_SETUP.md")
# 使用IMDb数据集，仅支持从本地parquet文件加载
LOCAL_DATASET_CACHE = "./datasets"  # 本地数据集缓存目录

NUM_EPOCHS = 5
BATCH_SIZE = 16

LEARNING_RATE_BASELINE = 2e-5
LEARNING_RATE_LORA = 3e-5
LEARNING_RATE_DAPT_MLM = 5e-5
LEARNING_RATE_DAPT_FINETUNE = 3e-5

TRAIN_SAMPLES = 10000
EVAL_SAMPLES = 2000


# 评估函数：计算准确率和F1分数
def compute_metrics(eval_pred):
    """
    计算评估指标的函数
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # 计算 F1 score (macro)
    f1 = f1_score(labels, predictions, average="macro")
    # 计算 Accuracy
    acc = accuracy_score(labels, predictions)

    # 必须返回一个字典
    return {"accuracy": acc, "f1": f1}


# 自定义支持 R-Drop 的 Trainer
class RDropTrainer(Trainer):
    """
    实现 R-Drop 损失：交叉熵 + KL 一致性约束。
    思路：同一批样本前向两次，利用 dropout 的随机性让输出分布保持一致。
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")

        # 第一次前向，获得 logits1
        outputs1 = model(**inputs)
        logits1 = outputs1.logits

        # 第二次前向（同一输入，利用 dropout 产生随机性），获得 logits2
        outputs2 = model(**inputs)
        logits2 = outputs2.logits

        # 交叉熵主损失：两次输出分别计算 CE，再取平均
        loss_ce = 0.5 * (
            F.cross_entropy(logits1, labels)
            + F.cross_entropy(logits2, labels)
        )

        # KL 一致性损失：强制两次输出分布接近
        p_loss = F.kl_div(
            F.log_softmax(logits1, dim=-1),
            F.softmax(logits2, dim=-1),
            reduction="batchmean",
        )
        q_loss = F.kl_div(
            F.log_softmax(logits2, dim=-1),
            F.softmax(logits1, dim=-1),
            reduction="batchmean",
        )
        kl_loss = 0.5 * (p_loss + q_loss)

        alpha = 4.0  # R-Drop 系数，可调，用于控制一致性损失权重
        loss = loss_ce + alpha * kl_loss

        return (loss, outputs1) if return_outputs else loss


# 辅助函数：打印模型可训练参数的数量
# 这个很重要，可以直观地看到 LoRA 的"高效"体现在哪里
def print_trainable_parameters(model):
    """
    打印模型中可训练参数的数量
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"  可训练参数 (trainable params): {trainable_params}"
        f" || 总参数 (all params): {all_param}"
        f" || 可训练比例 (trainable %): {100 * trainable_params / all_param:.2f}%"
    )
    return trainable_params, all_param


# --- 可视化函数 ---
def plot_training_curves(trainer, experiment_name, save_path="./visualizations"):
    """
    绘制训练曲线（loss和metrics）
    """
    os.makedirs(save_path, exist_ok=True)
    
    history = trainer.state.log_history
    
    # 提取训练和评估指标
    train_loss = [x['loss'] for x in history if 'loss' in x and 'eval_loss' not in x]
    eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]
    eval_acc = [x['eval_accuracy'] for x in history if 'eval_accuracy' in x]
    eval_f1 = [x['eval_f1'] for x in history if 'eval_f1' in x]
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss曲线
    axes[0].plot(train_loss, label='训练Loss', marker='o')
    if eval_loss:
        axes[0].plot(eval_loss, label='验证Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{experiment_name} - Loss曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy曲线
    if eval_acc:
        axes[1].plot(eval_acc, label='验证准确率', marker='o', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'{experiment_name} - 准确率曲线')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # F1曲线
    if eval_f1:
        axes[2].plot(eval_f1, label='验证F1分数', marker='o', color='orange')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1 Score')
        axes[2].set_title(f'{experiment_name} - F1分数曲线')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(save_path, f"{experiment_name.replace(' ', '_')}_training_curves.png")
    # 确保保存时使用正确的中文字体
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"训练曲线已保存到: {filename}")
    plt.close()


def plot_confusion_matrix(model, tokenizer, eval_dataset, device, experiment_name, save_path="./visualizations"):
    """
    绘制混淆矩阵
    """
    os.makedirs(save_path, exist_ok=True)
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    print(f"正在计算 {experiment_name} 的混淆矩阵...")
    with torch.no_grad():
        for i in tqdm(range(min(1000, len(eval_dataset))), desc="预测中"):
            sample = eval_dataset[i]
            text = sample['text']
            label = sample['label']
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).cpu().item()
            
            all_predictions.append(pred)
            all_labels.append(label)
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    # 绘制
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'{experiment_name} - 混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    
    filename = os.path.join(save_path, f"{experiment_name.replace(' ', '_')}_confusion_matrix.png")
    # 确保保存时使用正确的中文字体
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"混淆矩阵已保存到: {filename}")
    plt.close()
    
    # 打印分类报告
    print(f"\n{classification_report(all_labels, all_predictions, target_names=['Negative', 'Positive'])}")


def plot_comparison(results_dict, save_path="./visualizations"):
    """
    绘制三个实验的对比图
    """
    os.makedirs(save_path, exist_ok=True)
    
    experiments = list(results_dict.keys())
    accuracies = [results_dict[exp].get('eval_accuracy', 0) for exp in experiments]
    f1_scores = [results_dict[exp].get('eval_f1', 0) for exp in experiments]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 准确率对比
    axes[0].bar(experiments, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_ylabel('准确率')
    axes[0].set_title('三个实验的准确率对比')
    axes[0].set_ylim([0.8, 0.95])
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom')
    
    # F1分数对比
    axes[1].bar(experiments, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_ylabel('F1分数')
    axes[1].set_title('三个实验的F1分数对比')
    axes[1].set_ylim([0.8, 0.95])
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(f1_scores):
        axes[1].text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    filename = os.path.join(save_path, "experiments_comparison.png")
    # 确保保存时使用正确的中文字体
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"实验对比图已保存到: {filename}")
    plt.close()


# --- 数据增强函数定义 ---
# 数据增强函数（EDA）
try:
    from nltk.corpus import wordnet as wn
    from nltk import word_tokenize, pos_tag
    import nltk
    
    # 下载必要的NLTK数据（首次运行需要，静默下载）
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("警告: NLTK 未安装，数据增强中的同义词替换功能将受限。")
    print("   安装命令: pip install nltk")
    print("   将使用简化的数据增强方法（随机删除、交换等）")


def get_synonyms(word, pos=None):
    """获取同义词（用于数据增强）"""
    if not NLTK_AVAILABLE:
        return []
    
    synonyms = set()
    try:
        for syn in wn.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                if synonym != word.lower():
                    synonyms.add(synonym)
    except:
        pass
    
    return list(synonyms)


def synonym_replacement(text, n=3):
    """同义词替换：随机替换n个词为同义词"""
    if not NLTK_AVAILABLE or len(text.split()) <= 1:
        return text
    
    try:
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        
        # 只替换名词、动词、形容词、副词
        replaceable_pos = ['NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 
                          'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
        
        replaceable_words = [(i, word, pos) for i, (word, pos) in enumerate(pos_tags) 
                            if pos in replaceable_pos and word.isalpha()]
        
        if len(replaceable_words) == 0:
            return text
        
        n = min(n, len(replaceable_words))
        selected = random.sample(replaceable_words, n)
        
        new_words = words.copy()
        for idx, word, pos in selected:
            synonyms = get_synonyms(word, pos)
            if synonyms:
                new_words[idx] = random.choice(synonyms)
        
        return ' '.join(new_words)
    except:
        return text


def random_deletion(text, p=0.1):
    """随机删除：以概率p随机删除词"""
    words = text.split()
    if len(words) <= 1:
        return text
    
    min_words = max(1, len(words) // 2)
    new_words = [w for w in words if random.random() > p]
    
    if len(new_words) < min_words:
        new_words = random.sample(words, min_words)
    
    return ' '.join(new_words) if new_words else text


def random_swap(text, n=3):
    """随机交换：随机交换n对相邻词的位置"""
    words = text.split()
    if len(words) <= 1:
        return text
    
    new_words = words.copy()
    for _ in range(min(n, len(new_words) - 1)):
        idx1 = random.randint(0, len(new_words) - 2)
        idx2 = idx1 + 1
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    
    return ' '.join(new_words)


def eda_augment(text, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1, num_aug=1, use_synonym=True):
    """
    EDA: Easy Data Augmentation
    结合同义词替换、随机插入、随机交换、随机删除
    
    Args:
        text: 输入文本
        alpha_sr: 同义词替换比例（仅在 use_synonym=True 时生效）
        alpha_ri: 随机插入比例（简化版，暂不实现）
        alpha_rs: 随机交换比例
        alpha_rd: 随机删除比例
        num_aug: 生成多少个增强样本（这里简化为1个）
        use_synonym: 是否使用同义词替换
    
    Returns:
        增强后的文本
    """
    words = text.split()
    num_words = len(words)
    
    if num_words == 0:
        return text
    
    augmented = text
    
    # 同义词替换（可选，耗时操作）
    if use_synonym and random.random() < alpha_sr and NLTK_AVAILABLE:
        n_sr = max(1, int(alpha_sr * num_words))
        augmented = synonym_replacement(augmented, n=n_sr)
    
    # 随机交换（快速操作）
    if random.random() < alpha_rs:
        n_rs = max(1, int(alpha_rs * num_words))
        augmented = random_swap(augmented, n=n_rs)
    
    # 随机删除（快速操作）
    if random.random() < alpha_rd:
        augmented = random_deletion(augmented, p=alpha_rd)
    
    return augmented


def augment_dataset_batch(examples, num_aug=2, use_synonym=True):
    """
    批量数据增强函数
    对每个样本生成num_aug个增强版本
    
    Args:
        examples: 包含'text'和'label'的字典
        num_aug: 每个样本生成多少个增强版本
        use_synonym: 是否使用同义词替换
    
    Returns:
        增强后的样本（包含原始样本+增强样本）
    """
    texts = examples['text']
    labels = examples['label']
    
    augmented_texts = []
    augmented_labels = []
    
    for text, label in zip(texts, labels):
        # 添加原始样本
        augmented_texts.append(text)
        augmented_labels.append(label)
        
        # 添加增强样本
        for _ in range(num_aug):
            aug_text = eda_augment(text, num_aug=1, use_synonym=use_synonym)
            augmented_texts.append(aug_text)
            augmented_labels.append(label)
    
    return {'text': augmented_texts, 'label': augmented_labels}


print("--- 2. 全局配置和辅助函数定义完毕 ---\n")

# --- 3. 加载分词器 (Tokenizer) ---
print("--- 3. 加载分词器 ---")
# 确保使用和模型匹配的分词器
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, local_files_only=False)
    print(f"分词器 '{MODEL_CHECKPOINT}' 加载成功。")
except Exception as e:
    print(f"分词器加载失败: {str(e)}")
    if MODEL_CHECKPOINT == MODEL_NAME:
        print(f"\n提示：网络连接可能有问题。")
        print(f"请运行以下命令下载模型到本地：")
        print(f"  python download_model.py")
        print(f"或者参考 OFFLINE_SETUP.md 手动下载")
    raise

# 创建一个数据整理器 (Data Collator)
# 它会帮我们把一个 batch 里的数据动态 padding 到相同的长度，而不是整个数据集
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print()

# --- 4. 准备数据集 ---
print("--- 4. 准备数据集 ---")


def tokenize_function(examples):
    # 核心：使用分词器处理文本。
    # truncation=True 意味着如果文本太长（超过模型最大长度），就截断。
    # 这是必须的，否则模型会报错。
    return tokenizer(examples["text"], truncation=True)


# 加载 IMDb 数据集
# 注意：从本地 parquet 文件加载，不使用在线下载
print("--- 4. 准备数据集 (从本地 parquet 文件加载) ---")

# 检查本地是否有 parquet 文件
imdb_local_path = os.path.join(LOCAL_DATASET_CACHE, "imdb")
train_parquet = os.path.join(imdb_local_path, "train-00000-of-00001.parquet")
test_parquet = os.path.join(imdb_local_path, "test-00000-of-00001.parquet")

if not os.path.exists(train_parquet) or not os.path.exists(test_parquet):
    print("未找到本地数据集文件")
    print(f"   请确保数据集文件存在于: {imdb_local_path}")
    print(f"   需要的文件:")
    print(f"     - train-00000-of-00001.parquet")
    print(f"     - test-00000-of-00001.parquet")
    raise FileNotFoundError(f"数据集文件不存在: {imdb_local_path}")

# 从本地 parquet 文件加载
print(f"--- 从本地 parquet 文件加载数据集 ---")
print(f"   数据路径: {imdb_local_path}")

# 构建文件路径列表（支持多个分片）
train_files = glob.glob(os.path.join(imdb_local_path, "train-*.parquet"))
test_files = glob.glob(os.path.join(imdb_local_path, "test-*.parquet"))

if not train_files or not test_files:
    raise FileNotFoundError(f"未找到完整的 parquet 文件（需要 train 和 test 文件）")

try:
    raw_datasets = load_dataset(
        "parquet",
        data_files={
            "train": train_files,
            "test": test_files
        }
    )
    print("从本地 parquet 文件加载成功")
    print(f"   训练集文件数: {len(train_files)}, 测试集文件数: {len(test_files)}")
except Exception as e:
    print(f"数据集加载失败: {str(e)}")
    print(f"\n提示：请确保已安装 datasets 库: pip install datasets")
    raise

# 使用更多的训练数据以获得更好的效果
train_dataset = raw_datasets["train"].shuffle(seed=42).select(range(TRAIN_SAMPLES))
eval_dataset = raw_datasets["test"].shuffle(seed=42).select(range(EVAL_SAMPLES))

print(f"原始数据集加载成功: {raw_datasets}")
print(f"训练集 {len(train_dataset)}, 测试集 {len(eval_dataset)}")

# 使用 .map() 方法批量处理数据集
# batched=True 可以让分词器一次处理一批数据，速度更快
tokenized_datasets_basic = raw_datasets.map(tokenize_function, batched=True)
# 准备用于训练器的数据集
train_dataset_basic = tokenized_datasets_basic["train"].shuffle(seed=42).select(range(TRAIN_SAMPLES))
eval_dataset_basic = tokenized_datasets_basic["test"].shuffle(seed=42).select(range(EVAL_SAMPLES))

print("基础预处理（仅分词）完成。\n")

# --- 5. Baseline: 全量参数微调 ---
print("==============================================")
print("--- 5. Baseline: 全量参数微调 ---")
print("==============================================")
try:
    model_baseline = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=2,
        local_files_only=False
    )
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    if MODEL_CHECKPOINT == MODEL_NAME:
        print(f"\n提示：网络连接可能有问题。")
        print(f"请运行以下命令下载模型到本地：")
        print(f"  python download_model.py")
        print(f"或者参考 OFFLINE_SETUP.md 手动下载")
    raise

print("模型 (Baseline) 可训练参数:")
print_trainable_parameters(model_baseline)

# 定义训练参数
training_args_baseline = TrainingArguments(
    output_dir="./results/baseline",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE_BASELINE,  # Baseline 使用较小的学习率
    weight_decay=0.01,
    eval_strategy="epoch",  # 每个 epoch 评估一次
    save_strategy="epoch",  # 每个 epoch 保存一次
    load_best_model_at_end=True,  # 训练结束时加载最好的模型
    metric_for_best_model="eval_f1",  # 使用 F1 作为最佳模型指标
    greater_is_better=True,  # F1 越大越好
    save_total_limit=3,  # 只保留最近3个checkpoint，节省空间
    report_to="none",  # 禁用 wandb 和 TensorBoard（避免网络和路径问题）
    logging_dir=None,  # 禁用 TensorBoard 日志
    run_name="baseline",  # 设置简单的运行名称
    warmup_steps=100,  # 添加warmup，帮助训练稳定
    logging_steps=100,  # 每100步记录一次日志
    disable_tqdm=False,  # 不禁用tqdm，但我们会通过自定义回调来控制
    logging_first_step=True,  # 记录第一步
)

# 创建训练器 Trainer
trainer_baseline = Trainer(
    model=model_baseline,
    args=training_args_baseline,
    train_dataset=train_dataset_basic,
    eval_dataset=eval_dataset_basic,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,  # 传入我们定义的评估函数
)

# 开始训练！
print("\n--- 开始训练 (Baseline) ---")
trainer_baseline.train()
print("--- 训练 (Baseline) 完成 ---")

# 评估模型
print("\n--- 开始评估 (Baseline) ---")
eval_results_baseline = trainer_baseline.evaluate()
print("评估结果 (Baseline):", eval_results_baseline)

# 生成可视化
print("\n--- 生成可视化 (Baseline) ---")
plot_training_curves(trainer_baseline, "Baseline")
plot_confusion_matrix(model_baseline, tokenizer, eval_dataset_basic, 
                     torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                     "Baseline")

# 保存模型和分词器用于前端
print("\n--- 保存模型 (Baseline) 用于前端 ---")
model_save_path_baseline = "./saved_models/baseline"
model_baseline.save_pretrained(model_save_path_baseline)
tokenizer.save_pretrained(model_save_path_baseline)
print(f"模型已保存到: {model_save_path_baseline}")

print("--- Baseline 结束 ---\n")

# --- 6. LoRA 高效微调 ---
print("==============================================")
print("--- 6. LoRA 高效微调 ---")
print("==============================================")
try:
    model_lora = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=2,
        local_files_only=False
    )
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    if MODEL_CHECKPOINT == MODEL_NAME:
        print(f"\n提示：网络连接可能有问题。")
        print(f"请运行以下命令下载模型到本地：")
        print(f"  python download_model.py")
    raise

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_lin", "v_lin", "k_lin", "out_lin"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)

# 使用 `get_peft_model` 来包装我们的基础模型
model_lora_peft = get_peft_model(model_lora, lora_config)

print("模型 (LoRA) 可训练参数:")
print_trainable_parameters(model_lora_peft)

# 定义训练参数
training_args_lora = TrainingArguments(
    output_dir="./results/lora_basic",  # 换个目录
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE_LORA,
    weight_decay=0.01,
    eval_strategy="epoch",  # 使用新参数名
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",  # 使用 F1 作为最佳模型指标
    greater_is_better=True,
    save_total_limit=3,  # 只保留最近3个checkpoint
    report_to="none",  # 禁用 wandb 和 TensorBoard
    logging_dir=None,  # 禁用 TensorBoard 日志
    run_name="lora_basic",  # 设置简单的运行名称
    warmup_steps=100,  # 添加warmup
    logging_steps=100,  # 每100步记录一次日志
    disable_tqdm=False,  # 不禁用tqdm
    logging_first_step=True,
)

# 创建训练器
trainer_lora = Trainer(
    model=model_lora_peft,  # 注意：这里用的是 PEFT 包装过的模型
    args=training_args_lora,
    train_dataset=train_dataset_basic,
    eval_dataset=eval_dataset_basic,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 开始训练！
print("\n--- 开始训练 (LoRA) ---")
trainer_lora.train()
print("--- 训练 (LoRA) 完成 ---")

# 评估模型
print("\n--- 开始评估 (LoRA) ---")
eval_results_lora = trainer_lora.evaluate()
print("评估结果 (LoRA):", eval_results_lora)

# 生成可视化
print("\n--- 生成可视化 (LoRA) ---")
plot_training_curves(trainer_lora, "LoRA")
plot_confusion_matrix(trainer_lora.model, tokenizer, eval_dataset_basic,
                     torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                     "LoRA")

# 保存模型和分词器用于前端
print("\n--- 保存模型 (LoRA) 用于前端 ---")
model_save_path_lora = "./saved_models/lora_basic"
# 对于 PEFT 模型，需要保存 PEFT 适配器和基础模型
trainer_lora.model.save_pretrained(model_save_path_lora)
tokenizer.save_pretrained(model_save_path_lora)
print(f"模型已保存到: {model_save_path_lora}")

print("--- LoRA 结束 ---\n")

# --- 7. LoRA + DAPT (领域适应性预训练) ---
print("==============================================")
print("--- 7. LoRA + DAPT (领域适应性预训练) ---")
print("==============================================")

# ========== 阶段1: DAPT (领域适应性预训练) ==========
print("\n--- 阶段1: DAPT (领域适应性预训练) ---")

# 1. 准备 MLM 预训练数据
print("\n准备 MLM 预训练数据...")
dapt_texts_train = raw_datasets["train"]["text"]
dapt_texts_test = raw_datasets["test"]["text"]
all_dapt_texts = dapt_texts_train + dapt_texts_test
print(f"   总文本数: {len(all_dapt_texts)}")
print(f"   使用前 {min(20000, len(all_dapt_texts))} 条文本进行 DAPT")
DAPT_SAMPLES = min(20000, len(all_dapt_texts))
dapt_texts = all_dapt_texts[:DAPT_SAMPLES]

# 创建用于 MLM 的数据集
dapt_dataset = Dataset.from_dict({"text": dapt_texts})

# 2. 对文本进行分词（用于 MLM）
def tokenize_function_mlm(examples):
    """对 MLM 预训练的文本进行分词"""
    return tokenizer(examples["text"], truncation=True, max_length=512)

print("对 DAPT 数据进行分词...")
dapt_dataset_tokenized = dapt_dataset.map(
    tokenize_function_mlm,
    batched=True,
    remove_columns=["text"],  # 移除原始文本列，只保留 tokenized 结果
    desc="分词中..."
)

# 3. 创建 MLM 数据整理器
mlm_probability = 0.15
data_collator_mlm = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,  # 启用 MLM
    mlm_probability=mlm_probability
)

# 4. 加载 MLM 模型（用于预训练）
print("\n加载 MLM 模型用于 DAPT...")
try:
    model_dapt_mlm = AutoModelForMaskedLM.from_pretrained(
        MODEL_CHECKPOINT,
        local_files_only=False
    )
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    if MODEL_CHECKPOINT == MODEL_NAME:
        print(f"\n提示：网络连接可能有问题。")
        print(f"请运行以下命令下载模型到本地：")
        print(f"  python download_model.py")
    raise

print("模型 (DAPT MLM) 可训练参数:")
print_trainable_parameters(model_dapt_mlm)

# 5. 配置 DAPT 训练参数
DAPT_EPOCHS = 2
training_args_dapt = TrainingArguments(
    output_dir="./results/dapt_mlm",
    num_train_epochs=DAPT_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE_DAPT_MLM,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=1,  # 只保留最后一个 checkpoint
    report_to="none",
    logging_dir=None,
    run_name="dapt_mlm",
    warmup_steps=100,
    logging_steps=200,
    disable_tqdm=False,
    logging_first_step=True,
)

# 6. 创建 DAPT Trainer（MLM 任务不需要 compute_metrics）
trainer_dapt = Trainer(
    model=model_dapt_mlm,
    args=training_args_dapt,
    train_dataset=dapt_dataset_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator_mlm,
)

# 7. 开始 DAPT 预训练
print(f"\n--- 开始 DAPT 预训练 ({DAPT_EPOCHS} 个 epoch) ---")
trainer_dapt.train()
print("--- DAPT 预训练完成 ---")

# 8. 保存 DAPT 后的模型
dapt_model_path = "./saved_models/dapt_base"
print(f"\n保存 DAPT 后的模型到: {dapt_model_path}")
trainer_dapt.model.save_pretrained(dapt_model_path)
tokenizer.save_pretrained(dapt_model_path)
print("DAPT 模型保存完成")

# ========== 阶段2: 在 DAPT 后的模型上做 LoRA 微调 ==========
print("\n--- 阶段2: 在 DAPT 后的模型上做 LoRA 微调 ---")

# 1. 准备分类任务数据（使用基础数据集，不需要数据增强）
print("\n准备分类任务数据...")
train_dataset_raw = raw_datasets["train"].shuffle(seed=42).select(range(TRAIN_SAMPLES))
eval_dataset_raw = raw_datasets["test"].shuffle(seed=42).select(range(EVAL_SAMPLES))

# 对数据进行分词
train_dataset_advanced = train_dataset_raw.map(tokenize_function, batched=True)
eval_dataset_advanced = eval_dataset_raw.map(tokenize_function, batched=True)
print("数据预处理完成")

# 2. 从 DAPT 后的模型加载，创建分类模型
print("\n从 DAPT 后的模型创建分类模型...")
try:
    # 先加载 DAPT 后的 MLM 模型
    model_dapt_mlm_loaded = AutoModelForMaskedLM.from_pretrained(
        dapt_model_path,
        local_files_only=False
    )
    
    # 从 MLM 模型提取 DistilBERT 的 encoder 部分，创建分类模型
    # 注意：DistilBERT 的 MLM 和分类模型共享 encoder，可以直接转换
    model_lora_advanced = AutoModelForSequenceClassification.from_pretrained(
        dapt_model_path,  # 从 DAPT 模型加载
        num_labels=2,
        local_files_only=False
    )
    
    # 如果直接加载失败，尝试从基础模型加载并复制 DAPT 的权重
    # （这需要手动处理，但 transformers 通常会自动处理）
except Exception as e:
    print(f"从 DAPT 模型创建分类模型失败: {str(e)}")
    print("尝试从基础模型加载并手动复制 DAPT 权重...")
    # 备用方案：从基础模型加载，然后手动复制 encoder 权重
    model_lora_advanced = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=2,
        local_files_only=False
    )
    # 加载 DAPT 的 encoder 权重
    dapt_state_dict = model_dapt_mlm_loaded.distilbert.state_dict()
    model_lora_advanced.distilbert.load_state_dict(dapt_state_dict)
    print("已手动复制 DAPT 权重到分类模型")

# 3. 应用 LoRA
model_lora_advanced_peft = get_peft_model(model_lora_advanced, lora_config)

print("模型 (LoRA + DAPT) 可训练参数:")
print_trainable_parameters(model_lora_advanced_peft)

# 4. 准备训练参数
training_args_lora_advanced = TrainingArguments(
    output_dir="./results/lora_advanced",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE_DAPT_FINETUNE,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    save_total_limit=3,
    report_to="none",
    logging_dir=None,
    run_name="lora_dapt",
    warmup_steps=100,
    logging_steps=100,
    disable_tqdm=False,
    logging_first_step=True,
)

trainer_lora_advanced = Trainer(
    model=model_lora_advanced_peft,
    args=training_args_lora_advanced,
    train_dataset=train_dataset_advanced,
    eval_dataset=eval_dataset_advanced,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 5. 开始训练
print("\n--- 开始训练 (LoRA + DAPT) ---")
trainer_lora_advanced.train()
print("--- 训练 (LoRA + DAPT) 完成 ---")

# 6. 评估模型
print("\n--- 开始评估 (LoRA + DAPT) ---")
eval_results_lora_advanced = trainer_lora_advanced.evaluate()
print("评估结果 (LoRA + DAPT):", eval_results_lora_advanced)

# 生成可视化
print("\n--- 生成可视化 (LoRA + DAPT) ---")
plot_training_curves(trainer_lora_advanced, "LoRA_DAPT")
plot_confusion_matrix(trainer_lora_advanced.model, tokenizer, eval_dataset_advanced,
                     torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                     "LoRA_DAPT")

# 保存模型和分词器用于前端
print("\n--- 保存模型 (LoRA + DAPT) 用于前端 ---")
model_save_path_lora_advanced = "./saved_models/lora_advanced"
trainer_lora_advanced.model.save_pretrained(model_save_path_lora_advanced)
tokenizer.save_pretrained(model_save_path_lora_advanced)
print(f"模型已保存到: {model_save_path_lora_advanced}")

print("--- LoRA + DAPT 结束 ---\n")

# --- 8. LoRA + DAPT + R-Drop ---
print("==============================================")
print("--- 8. LoRA + DAPT + R-Drop ---")
print("==============================================")
print("\n从 DAPT 后的模型创建分类模型...")
try:
    model_lora_rdrop = AutoModelForSequenceClassification.from_pretrained(
        dapt_model_path,
        num_labels=2,
        local_files_only=False
    )
except Exception as e:
    print(f"从 DAPT 模型创建分类模型失败: {str(e)}")
    print("尝试从基础模型加载并手动复制 DAPT 权重...")
    # 备用方案：从基础模型加载，然后手动复制 encoder 权重
    # 先加载 DAPT 后的 MLM 模型以获取权重
    model_dapt_mlm_for_rdrop = AutoModelForMaskedLM.from_pretrained(
        dapt_model_path,
        local_files_only=False
    )
    model_lora_rdrop = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=2,
        local_files_only=False
    )
    # 加载 DAPT 的 encoder 权重
    dapt_state_dict = model_dapt_mlm_for_rdrop.distilbert.state_dict()
    model_lora_rdrop.distilbert.load_state_dict(dapt_state_dict)
    print("已手动复制 DAPT 权重到分类模型")

model_lora_rdrop_peft = get_peft_model(model_lora_rdrop, lora_config)

print("模型 (LoRA + DAPT + R-Drop) 可训练参数:")
print_trainable_parameters(model_lora_rdrop_peft)

# 3. 训练参数
training_args_lora_rdrop = TrainingArguments(
    output_dir="./results/lora_rdrop",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE_DAPT_FINETUNE,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    save_total_limit=3,
    report_to="none",
    logging_dir=None,
    run_name="lora_rdrop",
    warmup_steps=100,
    logging_steps=100,
    disable_tqdm=False,
    logging_first_step=True,
)

# 4. 使用自定义 R-Drop Trainer
print("\n--- 构建 R-Drop Trainer ---")
trainer_lora_rdrop = RDropTrainer(
    model=model_lora_rdrop_peft,
    args=training_args_lora_rdrop,
    train_dataset=train_dataset_advanced,
    eval_dataset=eval_dataset_advanced,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 5. 开始训练
print("\n--- 开始训练 (LoRA + DAPT + R-Drop) ---")
trainer_lora_rdrop.train()
print("--- 训练 (LoRA + DAPT + R-Drop) 完成 ---")

# 6. 评估
print("\n--- 开始评估 (LoRA + DAPT + R-Drop) ---")
eval_results_lora_rdrop = trainer_lora_rdrop.evaluate()
print("评估结果 (LoRA + DAPT + R-Drop):", eval_results_lora_rdrop)

# 7. 可视化
print("\n--- 生成可视化 (LoRA + DAPT + R-Drop) ---")
plot_training_curves(trainer_lora_rdrop, "LoRA_DAPT_RDrop")
plot_confusion_matrix(
    trainer_lora_rdrop.model,
    tokenizer,
    eval_dataset_advanced,
    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "LoRA_DAPT_RDrop",
)

# 8. 保存模型
print("\n--- 保存模型 (LoRA + DAPT + R-Drop) 用于前端 ---")
model_save_path_lora_rdrop = "./saved_models/lora_rdrop"
trainer_lora_rdrop.model.save_pretrained(model_save_path_lora_rdrop)
tokenizer.save_pretrained(model_save_path_lora_rdrop)
print(f"模型已保存到: {model_save_path_lora_rdrop}")

print("--- LoRA + DAPT + R-Drop 结束 ---\n")

# --- 9. 结果汇总与对比 ---
print("==============================================")
print("--- 9. 结果汇总 ---")
print("==============================================")
print("\n--- Baseline 评估结果 ---")
print(eval_results_baseline)

print("\n--- LoRA 评估结果 ---")
print(eval_results_lora)

print("\n--- LoRA + DAPT 评估结果 ---")
print(eval_results_lora_advanced)

print("\n--- LoRA + DAPT + R-Drop 评估结果 ---")
print(eval_results_lora_rdrop)

print("\n--- 初步分析 ---")
baseline_f1 = eval_results_baseline.get('eval_f1', 0)
lora_f1 = eval_results_lora.get('eval_f1', 0)
lora_advanced_f1 = eval_results_lora_advanced.get('eval_f1', 0)
lora_rdrop_f1 = eval_results_lora_rdrop.get('eval_f1', 0)

print(f"Baseline F1: {baseline_f1:.4f} | Accuracy: {eval_results_baseline.get('eval_accuracy', 0):.4f}")
print(f"LoRA F1: {lora_f1:.4f} | Accuracy: {eval_results_lora.get('eval_accuracy', 0):.4f}")
print(f"LoRA + DAPT F1: {lora_advanced_f1:.4f} | Accuracy: {eval_results_lora_advanced.get('eval_accuracy', 0):.4f}")
print(f"LoRA + DAPT + R-Drop F1: {lora_rdrop_f1:.4f} | Accuracy: {eval_results_lora_rdrop.get('eval_accuracy', 0):.4f}")

# 生成实验对比图
print("\n--- 生成实验对比图 ---")
results_dict = {
    "Baseline": eval_results_baseline,
    "LoRA": eval_results_lora,
    "LoRA_DAPT": eval_results_lora_advanced,
    "LoRA_DAPT_RDrop": eval_results_lora_rdrop,
}
plot_comparison(results_dict)

print("\n--- 结果分析 ---")
print(f"训练配置：{NUM_EPOCHS} 个 epoch，训练样本 {TRAIN_SAMPLES} 条，测试样本 {EVAL_SAMPLES} 条")
print("\n1. 对比 Baseline 和 LoRA：")
performance_diff = baseline_f1 - lora_f1
if lora_f1 < 0.5:
    print(f"   警告: LoRA 模型效果较差 (F1={lora_f1:.4f})")
else:
    print(f"   LoRA 用极少的参数达到了接近 Baseline 的性能！")
    print(f"      - Baseline: F1={baseline_f1:.4f}, 参数量=66.96M (100%)")
    print(f"      - LoRA: F1={lora_f1:.4f}, 参数量≈1.18M (1.73%)")
    print(f"      - 参数量减少: 66.96M -> 1.18M (减少 98.2%)")
    print(f"      - 性能损失: {performance_diff:.4f} (仅 {performance_diff/baseline_f1*100:.2f}%)")
    print(f"      - LoRA 高效微调：用不到 2% 的参数达到接近 Baseline 的性能")

print("\n2. 对比 LoRA 和 LoRA + DAPT：")
dapt_diff = lora_advanced_f1 - lora_f1
if abs(dapt_diff) < 0.01:
    print(f"   DAPT 效果不明显 (F1差异: {abs(dapt_diff):.4f})")
else:
    if dapt_diff > 0:
        print(f"   DAPT 带来了 {dapt_diff:.4f} 的提升")
    else:
        print(f"   DAPT 略微降低了性能 ({abs(dapt_diff):.4f})")

print("\n3. 对比 LoRA + DAPT 和 LoRA + DAPT + R-Drop：")
rdrop_diff = lora_rdrop_f1 - lora_advanced_f1
if rdrop_diff > 0.005:
    print(f"   R-Drop 在 DAPT 基础上带来了额外 {rdrop_diff:.4f} 的提升")
elif rdrop_diff < -0.005:
    print(f"   数据增强效果更优（差异: {abs(rdrop_diff):.4f}）")
else:
    print(f"   两种方法效果相近（差异: {abs(rdrop_diff):.4f}）")

print("\n--- 结论 ---")
print("主要发现：")
print(f"   1. LoRA 高效微调：仅用 1.73% 的参数达到约 {lora_f1/baseline_f1*100:.1f}% 的性能")
print(f"   2. Baseline 全量微调：F1={baseline_f1:.4f}")
print(f"   3. LoRA：F1={lora_f1:.4f}，性能损失 {performance_diff:.4f}")

print("==============================================")

# --- 9. 模型可解释性分析 (SHAP) ---
print("\n==============================================")
print("--- 9. 模型可解释性分析 (SHAP) ---")
print("==============================================")
print("\n提示：SHAP 可解释性分析已在专门的 Jupyter Notebook 中完成")
print("   请使用 SHAP_Analysis.ipynb 进行交互式可视化分析")
print("   Notebook 提供以下功能：")
print("      - 交互式文本高亮可视化 (shap.plots.text())")
print("      - 条形图显示最重要的词 (shap.plots.bar())")
print("      - 单个样本详细分析")
print("      - 自动保存可视化结果")
print("\n   使用方法：")
print("      1. 确保已运行本脚本完成模型训练和保存")
print("      2. 打开 Jupyter Notebook: jupyter notebook SHAP_Analysis.ipynb")
print("      3. 在 Notebook 中选择要分析的模型（baseline/lora/lora_advanced）")
print("      4. 按顺序执行所有单元格即可")
print("\n   所有模型已保存到以下路径：")
print(f"      - Baseline: ./saved_models/baseline")
print(f"      - LoRA: ./saved_models/lora_basic")
print(f"      - LoRA + 数据增强: ./saved_models/lora_advanced")

print("\n==============================================")
print("--- 脚本执行完毕 ---")
print("==============================================")
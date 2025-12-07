# Gradio 前端应用 - 用于展示训练好的文本分类模型
# 这个文件提供了一个友好的 Web 界面，用于测试和展示训练好的模型

import torch
import gradio as gr
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib
import imageio
from PIL import Image
from io import BytesIO
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import re
import os
import time
from matplotlib import colormaps
import warnings
import shutil
import pandas as pd

plt.switch_backend("Agg")

# 抑制各种警告信息
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*TypedStorage.*')
warnings.filterwarnings('ignore', message='.*UntypedStorage.*')

# 抑制 transformers 的警告
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# 抑制 Gradio 的网络连接警告（离线环境）
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

# 配置 matplotlib 支持中文显示（避免字体警告）
font_list = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'STSong', 'Arial Unicode MS']
available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
chinese_font = None
for font in font_list:
    if font in available_fonts:
        chinese_font = font
        break

if chinese_font:
    plt.rcParams['font.sans-serif'] = [chinese_font]
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 抑制字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# --- 全局配置 ---
# 优先使用本地模型，如果不存在则使用在线模型
# 与 distilBERT.py 保持一致的检查逻辑
MODEL_NAME = "distilbert-base-uncased"
LOCAL_MODEL_PATH = "./models/distilbert-base-uncased"

# 检查本地模型是否存在（不仅检查路径，还检查 config.json）
if os.path.exists(LOCAL_MODEL_PATH) and os.path.exists(os.path.join(LOCAL_MODEL_PATH, "config.json")):
    MODEL_CHECKPOINT = LOCAL_MODEL_PATH
    print(f"[INFO] 检测到本地模型，使用离线模式: {MODEL_CHECKPOINT}")
else:
    MODEL_CHECKPOINT = MODEL_NAME
    print(f"[INFO] 警告: 未找到本地模型，将尝试在线下载: {MODEL_CHECKPOINT}")
    print(f"[INFO] 提示：如果网络有问题，请先下载模型到 {LOCAL_MODEL_PATH}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型路径配置（包含指标数据）
MODEL_PATHS = {
    "实验一: Baseline (全量微调)": {
        "path": "./saved_models/baseline",
        "metrics": {
            "F1": 0.9300,
            "Accuracy": 0.9300,
            "参数量": "66.96M (100%)"
        }
    },
    "实验二: LoRA (高效微调)": {
        "path": "./saved_models/lora_basic",
        "metrics": {
            "F1": 0.9100,
            "Accuracy": 0.9100,
            "参数量": "1.18M (1.73%)"
        }
    },
    "实验三: LoRA + 数据增强 (当前最佳)": {
        "path": "./saved_models/lora_advanced",
        "metrics": {
            "F1": 0.9190,
            "Accuracy": 0.9190,
            "参数量": "1.18M (1.73%)"
        }
    }
}

# 展示用模型描述
MODEL_SUMMARY = {
    "实验一: Baseline (全量微调)": {
        "title": "Baseline",
        "bullets": [
            "完整微调 DistilBERT，参数量最大",
            "作为性能基准，推理较慢",
            "适合对照改进效果"
        ]
    },
    "实验二: LoRA (高效微调)": {
        "title": "LoRA",
        "bullets": [
            "冻结大部分参数，仅训练 LoRA Adapter",
            "显著降低训练/推理成本",
            "通过增大 r / α 并优化学习率获得更好表现"
        ]
    },
    "实验三: LoRA + 数据增强 (当前最佳)": {
        "title": "LoRA + EDA 数据增强",
        "bullets": [
            "在 LoRA 基础上叠加 EDA 同义词替换、随机删除/交换",
            "显著提升鲁棒性与泛化能力",
            "当前综合指标最优"
        ]
    }
}

# 例句库（用于轮换功能）
EXAMPLE_SENTENCES = [
    "This is one of the best films I've ever seen. Truly brilliant and captivating.",
    "I hated every second of this movie. It was a complete waste of time and money.",
    "The acting was decent and the cinematography was beautiful, but the plot was predictable and boring.",
    "The movie had its moments, but overall it was just okay. Nothing special.",
    "While the special effects were impressive, the story lacked depth and the characters were poorly developed.",
    "This movie is absolutely amazing! I loved every minute of it. The acting was brilliant and the plot was engaging.",
    "I was disappointed by this film. The trailer looked promising, but the actual movie was boring.",
    "A masterpiece of cinema! The director's vision is truly remarkable and the performances are outstanding.",
    "Terrible movie. Poor acting, weak script, and completely uninteresting. Save your money.",
    "An average film with some good moments but nothing that stands out. Watchable but forgettable."
]

ANIMATION_DIR = "./visualizations/gradio"
os.makedirs(ANIMATION_DIR, exist_ok=True)

# shap 色带
HEATMAP = colormaps.get_cmap("coolwarm")

# 类别标签
LABELS = ["Negative (消极)", "Positive (积极)"]

CUSTOM_CSS = """
#animation-view {
  min-height: 420px;
}
#animation-view img {
  object-fit: contain !important;
}
.replay-btn {
  display: flex;
  align-items: center;
  justify-content: center;
}
.replay-btn button {
  min-height: 44px;
  min-width: 140px;
}
"""

# --- 辅助函数 ---

def clean_html(text):
    """
    清洗 HTML 标签（用于实验三的模型）
    """
    return re.sub(r'<.*?>', ' ', text)


def get_model_path(model_name):
    """获取模型路径（兼容新旧格式）"""
    if model_name in MODEL_PATHS:
        model_info = MODEL_PATHS[model_name]
        if isinstance(model_info, dict):
            return model_info["path"]
        else:
            return model_info
    return None

def get_model_metrics(model_name):
    """获取模型指标"""
    if model_name in MODEL_PATHS:
        model_info = MODEL_PATHS[model_name]
        if isinstance(model_info, dict):
            return model_info.get("metrics", {})
    return {}

def load_model(model_path):
    """
    加载模型和分词器（兼容普通模型与 LoRA 适配器）。
    - 优先从保存目录加载分词器；否则回退到基础 checkpoint。
    - 基础模型用来承载分类头，LoRA 适配器存在时再套上 PeftModel。
    """
    print(f"[DEBUG] load_model 开始，路径: {model_path}")
    try:
        # 加载分词器（从保存的模型路径加载，支持离线）
        print(f"[DEBUG] 加载分词器...")
        # 保存的模型路径中应该包含分词器，优先从保存路径加载
        # 如果保存路径中没有，则从基础模型路径加载
        if os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
            # 保存的模型中有分词器，直接使用
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            print(f"[DEBUG] 分词器从保存路径加载成功")
        else:
            # 保存的模型中没有分词器，从基础模型路径加载
            # 与 distilBERT.py 保持一致：如果 MODEL_CHECKPOINT 是本地路径，使用 local_files_only=True
            # 如果是模型名，使用 local_files_only=False（允许在线下载）
            use_local_only = (MODEL_CHECKPOINT != MODEL_NAME)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, local_files_only=use_local_only)
            print(f"[DEBUG] 分词器从基础模型路径加载成功 (local_files_only={use_local_only})")
        
        # 加载基础模型（与 distilBERT.py 保持一致）
        print(f"[DEBUG] 加载基础模型: {MODEL_CHECKPOINT}")
        # 如果 MODEL_CHECKPOINT 是本地路径，使用 local_files_only=True
        # 如果是模型名，使用 local_files_only=False（允许在线下载）
        use_local_only = (MODEL_CHECKPOINT != MODEL_NAME)
        
        # 抑制模型权重未初始化的警告（这是正常的，因为基础模型没有分类头）
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*Some weights.*were not initialized.*')
            warnings.filterwarnings('ignore', message='.*You should probably TRAIN.*')
            base_model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_CHECKPOINT,
                num_labels=2,
                local_files_only=use_local_only
            )
        print(f"[DEBUG] 基础模型加载成功 (local_files_only={use_local_only})")
        print(f"[INFO] 注意：分类头权重未初始化是正常的，加载保存的模型时会使用训练好的权重")
        
        # 检查是否存在 PEFT 适配器
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        print(f"[DEBUG] 检查 PEFT 适配器: {adapter_config_path}")
        print(f"[DEBUG] 适配器配置文件存在: {os.path.exists(adapter_config_path)}")
        
        if os.path.exists(adapter_config_path):
            # 这是 PEFT 模型，需要加载适配器
            print(f"[DEBUG] 检测到 PEFT 模型，加载适配器...")
            model = PeftModel.from_pretrained(base_model, model_path)
            print(f"[DEBUG] 已加载 PEFT 模型: {model_path}")
        else:
            # 这是普通模型（Baseline），直接加载
            print(f"[DEBUG] 检测到普通模型，直接加载...")
            # 保存的模型应该包含完整配置，使用 local_files_only=True
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                local_files_only=True
            )
            print(f"[DEBUG] 已加载普通模型: {model_path}")
        
        print(f"[DEBUG] 将模型移动到设备: {DEVICE}")
        model.to(DEVICE)
        model.eval()
        print(f"[DEBUG] 模型加载完成，已设置为评估模式")
        return model, tokenizer
    except Exception as e:
        error_msg = f"加载模型失败: {e}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return None, None


# 全局变量存储当前加载的模型/解释器
current_model = None
current_tokenizer = None
current_model_name = None
current_shap_explainer = None
# 存储最后一次生成的动画路径，用于重播
last_animation_path = None


def load_model_for_inference(model_name):
    """
    根据模型名称加载对应的模型
    """
    global current_model, current_tokenizer, current_model_name
    
    print(f"[DEBUG] load_model_for_inference 被调用，参数: model_name={model_name}")
    print(f"[DEBUG] 当前工作目录: {os.getcwd()}")
    print(f"[DEBUG] 可用模型路径: {MODEL_PATHS}")
    
    if model_name not in MODEL_PATHS:
        error_msg = f"错误: 未知的模型名称: {model_name}"
        print(f"[ERROR] {error_msg}")
        return error_msg
    
    model_path = get_model_path(model_name)
    print(f"[DEBUG] 选择的模型路径: {model_path}")
    print(f"[DEBUG] 路径是否存在: {os.path.exists(model_path)}")
    
    # 转换为绝对路径以便调试
    abs_path = os.path.abspath(model_path)
    print(f"[DEBUG] 绝对路径: {abs_path}")
    print(f"[DEBUG] 绝对路径是否存在: {os.path.exists(abs_path)}")
    
    if not os.path.exists(model_path):
        error_msg = f"错误: 模型路径不存在: {model_path}\n绝对路径: {abs_path}\n请先运行 distilBERT.py 训练模型。"
        print(f"[ERROR] {error_msg}")
        return error_msg
    
    print(f"[DEBUG] 开始加载模型...")
    model, tokenizer = load_model(model_path)
    
    if model is None or tokenizer is None:
        error_msg = f"错误: 无法加载模型: {model_path}"
        print(f"[ERROR] {error_msg}")
        return error_msg
    
    current_model = model
    current_tokenizer = tokenizer
    current_model_name = model_name
    
    print(f"[DEBUG] 模型加载成功，开始构建 SHAP 解释器...")
    build_shap_explainer()
    print(f"[DEBUG] SHAP 解释器构建完成")
    
    success_msg = f"成功加载模型: {model_name}\n模型路径: {model_path}\n设备: {DEVICE}"
    print(f"[SUCCESS] {success_msg}")
    return success_msg


def build_shap_explainer():
    """
    构建 SHAP 解释器，支撑逐词贡献与动画。
    - masker 用 tokenizer 的 mask_token
    - predict_proba 返回 softmax 概率供 SHAP 调用
    """
    global current_shap_explainer, current_model, current_tokenizer
    
    print(f"[DEBUG] build_shap_explainer 开始")
    print(f"[DEBUG] current_model: {current_model is not None}")
    print(f"[DEBUG] current_tokenizer: {current_tokenizer is not None}")
    
    if current_model is None or current_tokenizer is None:
        print(f"[WARN] 模型或分词器未加载，跳过 SHAP 解释器构建")
        current_shap_explainer = None
        return
    
    mask_token = current_tokenizer.mask_token or "[MASK]"
    print(f"[DEBUG] mask_token: {mask_token}")
    
    def predict_proba(text_list):
        if isinstance(text_list, str):
            processed_texts = [text_list]
        else:
            processed_texts = list(text_list)
        
        inputs = current_tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = current_model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy()
    
    print(f"[DEBUG] 创建 SHAP masker...")
    masker = shap.maskers.Text(tokenizer=current_tokenizer, mask_token=mask_token)
    print(f"[DEBUG] 创建 SHAP Explainer...")
    current_shap_explainer = shap.Explainer(predict_proba, masker)
    print(f"[DEBUG] SHAP Explainer 创建完成")


def predict_text(text, use_advanced_preprocessing=True):
    """
    完整推理流程：
    1) 可选清洗 HTML
    2) 分词并送入当前模型
    3) softmax 得到概率与预测类别
    4) 生成解释文本 + SHAP 贡献 + 动画
    """
    global current_model, current_tokenizer, current_model_name
    
    if current_model is None or current_tokenizer is None:
        return {
            "error": "请先加载模型！",
            "prediction": None,
            "probabilities": None,
            "explanation": None
        }
    
    if not text or len(text.strip()) == 0:
        return {
            "error": "请输入文本！",
            "prediction": None,
            "probabilities": None,
            "explanation": None
        }
    
    try:
        # 数据预处理
        if use_advanced_preprocessing and (
            "数据增强" in current_model_name or "数据清洗" in current_model_name
        ):
            # 如果使用的是实验三的模型，使用数据清洗
            processed_text = clean_html(text)
        else:
            processed_text = text
        
        # 分词
        inputs = current_tokenizer(
            processed_text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # 移动到设备
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # 模型推理
        with torch.no_grad():
            outputs = current_model(**inputs)
            logits = outputs.logits
        
        # 计算概率
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs_np = probs.cpu().numpy()[0]
        
        # 获取预测类别
        predicted_class = int(np.argmax(probs_np))
        confidence = float(probs_np[predicted_class])
        
        # 生成解释性文本
        explanation = generate_explanation(text, predicted_class, confidence, probs_np)
        
        token_contributions = compute_token_contributions(text)
        animation_path = None
        if token_contributions:
            animation_path = generate_token_animation(token_contributions, LABELS[predicted_class])
        
        return {
            "error": None,
            "prediction": predicted_class,
            "label": LABELS[predicted_class],
            "confidence": confidence,
            "probabilities": {
                LABELS[0]: float(probs_np[0]),
                LABELS[1]: float(probs_np[1])
            },
            "explanation": explanation,
            "token_contributions": token_contributions,
            "animation": animation_path
        }
    
    except Exception as e:
        return {
            "error": f"预测过程中出错: {str(e)}",
            "prediction": None,
            "probabilities": None,
            "explanation": None
        }


def generate_explanation(text, predicted_class, confidence, probs):
    """
    生成可解释性分析文本
    """
    label = LABELS[predicted_class]
    
    explanation = "### 模型解读\n"
    
    if confidence > 0.8:
        explanation += f"- 模型对 **{label}** 的判断非常确信（置信度 > 80%）。\n"
    elif confidence > 0.6:
        explanation += f"- 模型倾向于预测 **{label}**（置信度在 60%-80%）。\n"
    else:
        explanation += "- 置信度较低，建议结合上下文进一步验证。\n"
    
    # 简单的关键词分析
    positive_words = ["good", "great", "excellent", "amazing", "wonderful", "best", "love", "brilliant", "fantastic"]
    negative_words = ["bad", "terrible", "awful", "hate", "worst", "boring", "waste", "horrible", "disappointing"]
    
    text_lower = text.lower()
    found_positive = [w for w in positive_words if w in text_lower]
    found_negative = [w for w in negative_words if w in text_lower]
    
    if found_positive or found_negative:
        explanation += "\n### 关键词分析\n"
        if found_positive:
            explanation += f"- 检测到积极词汇: {', '.join(found_positive)}\n"
        if found_negative:
            explanation += f"- 检测到消极词汇: {', '.join(found_negative)}\n"
    else:
        explanation += "\n- 未检测到明显的情绪关键词。\n"
    
    return explanation


def compute_token_contributions(text):
    """
    调用 SHAP 解释器输出逐词贡献
    """
    if current_shap_explainer is None or current_tokenizer is None:
        return []
    
    try:
        values = current_shap_explainer([text])
        exp = values[0]
        
        token_values = exp.values
        if isinstance(token_values, np.ndarray) and token_values.ndim == 2:
            positives = token_values[:, 1]
        else:
            positives = np.array(token_values).reshape(-1)
        
        tokens = exp.data
        if tokens is None:
            tokens = current_tokenizer.tokenize(text)
        else:
            tokens = list(tokens)
        
        max_len = min(len(tokens), len(positives))
        tokens = tokens[:max_len]
        positives = positives[:max_len]
        
        # 过滤空 token
        cleaned = []
        for tok, val in zip(tokens, positives):
            tok_clean = tok if isinstance(tok, str) and tok.strip() else "[BLANK]"
            cleaned.append({
                "token": tok_clean,
                "value": float(val),
                "direction": "Positive" if val >= 0 else "Negative"
            })
        return cleaned
    except Exception as ex:
        print(f"[WARN] 计算 SHAP 贡献失败: {ex}")
        return []


def rgba_to_hex(rgba):
    """
    matplotlib 颜色转 16 进制
    """
    r, g, b, a = rgba
    return "#{:02x}{:02x}{:02x}".format(
        int(r * 255),
        int(g * 255),
        int(b * 255)
    )


def compute_frame_duration(num_tokens, target_total_duration=60.0, min_duration=1.2, max_duration=4.5):
    """
    根据文本长度（词数）自适应计算每帧动画持续时间
    减慢动画速度：增加目标总时长和最小/最大持续时间
    """
    if num_tokens <= 0:
        return 2.5
    per_frame = target_total_duration / float(num_tokens)
    return float(np.clip(per_frame, min_duration, max_duration))


def generate_token_animation(contributions, predicted_label, frame_duration=None, loop=False):
    """
    根据逐词贡献生成动图，按顺序依次高亮每个词，同时展示 SHAP 值的动态变化
    - 左侧：按网格高亮当前词，颜色强度随贡献大小
    - 右侧：动态条形图 + 累积值折线（twin x 轴）
    - 帧尺寸统一后写出 GIF，默认播放一次
    """
    if not contributions:
        return None
    
    try:
        # 确保 contributions 是列表且包含字典
        if not isinstance(contributions, list):
            print(f"[WARN] contributions 不是列表: {type(contributions)}")
            return None
        
        if len(contributions) == 0:
            print("[WARN] contributions 为空")
            return None
        
        # 提取 token 和 value，确保类型正确
        token_texts = []
        values_list = []
        for c in contributions:
            if not isinstance(c, dict):
                print(f"[WARN] 贡献项不是字典: {type(c)}")
                continue
            token_texts.append(str(c.get("token", "[UNK]")))
            val = c.get("value", 0.0)
            try:
                values_list.append(float(val))
            except (ValueError, TypeError):
                print(f"[WARN] 无法转换值为浮点数: {val}")
                values_list.append(0.0)
        
        if len(token_texts) == 0:
            print("[WARN] 没有有效的 token")
            return None
        
        values = np.array(values_list, dtype=np.float64)
        max_abs = max(np.max(np.abs(values)), 1e-6)

        if frame_duration is None:
            frame_duration = compute_frame_duration(len(token_texts))
        
        # 计算累积 SHAP 值（用于动态展示）
        cumulative_values = np.cumsum(values)
        
        # 确保所有数组长度一致
        assert len(token_texts) == len(values) == len(cumulative_values), \
            f"初始数组长度不一致: tokens={len(token_texts)}, values={len(values)}, cumulative={len(cumulative_values)}"
        
        frames = []
        for focus_idx in range(len(token_texts)):
            # 创建包含两个子图的图表：左侧显示词，右侧显示 SHAP 条形图
            fig = plt.figure(figsize=(16, 8))
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], hspace=0.3)
            
            # 左侧：词的高亮显示
            ax_left = fig.add_subplot(gs[0])
            ax_left.axis("off")
            ax_left.set_title(f"Token Highlighting · Step {focus_idx + 1}/{len(token_texts)}", 
                             fontsize=14, pad=12, loc="left")
            
            cols = 8
            rows = int(np.ceil(len(token_texts) / cols))
            
            for idx, (tok, val) in enumerate(zip(token_texts, values)):
                row = idx // cols
                col = idx % cols
                x = 0.03 + col * (0.115)
                y = 0.85 - row * (0.12)
                norm = 0.5 + 0.5 * (val / max_abs)
                color = rgba_to_hex(HEATMAP(norm))
                alpha = 1.0 if idx == focus_idx else 0.35
                edge = "#333333" if idx == focus_idx else color
                bbox_props = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=alpha, 
                                edgecolor=edge, linewidth=1.2 if idx == focus_idx else 0.1)
                ax_left.text(x, y, tok, fontsize=12, bbox=bbox_props, transform=ax_left.transAxes)
            
            ax_left.text(0.02, 0.05, f"Highlighting: {token_texts[focus_idx]}", 
                        fontsize=11, transform=ax_left.transAxes)
            
            # 右侧：SHAP 值的动态条形图（显示到当前词为止的累积贡献）
            ax_right = fig.add_subplot(gs[1])
            
            # 只显示到当前词为止的词
            display_tokens = token_texts[:focus_idx + 1]
            display_values = values[:focus_idx + 1]
            display_cumulative = cumulative_values[:focus_idx + 1]
            
            # 确保所有数组长度一致（转换为列表以确保一致性）
            display_tokens = list(display_tokens)
            display_values = np.array(display_values)
            display_cumulative = np.array(display_cumulative)
            
            min_len = min(len(display_tokens), len(display_values), len(display_cumulative))
            if min_len == 0:
                min_len = 1  # 至少显示一个词
                display_tokens = display_tokens[:1] if display_tokens else ["[BLANK]"]
                display_values = display_values[:1] if len(display_values) > 0 else np.array([0.0])
                display_cumulative = display_cumulative[:1] if len(display_cumulative) > 0 else np.array([0.0])
            else:
                display_tokens = display_tokens[:min_len]
                display_values = display_values[:min_len]
                display_cumulative = display_cumulative[:min_len]
            
            # 确保所有数组长度完全一致
            assert len(display_tokens) == len(display_values) == len(display_cumulative), \
                f"数组长度不一致: tokens={len(display_tokens)}, values={len(display_values)}, cumulative={len(display_cumulative)}"
            
            # 创建条形图：显示每个词的贡献
            y_pos = np.arange(len(display_tokens))
            colors = ['#ff4444' if v > 0 else '#4444ff' for v in display_values]
            ax_right.barh(y_pos, display_values, color=colors, alpha=0.7, label='Word Contribution')
            
            # 添加累积值折线图（叠加显示）
            # 使用 twiny() 创建第二个 x 轴来显示累积值
            ax_right_twin = ax_right.twiny()
            # 确保数组长度一致后再绘制
            if len(display_cumulative) == len(y_pos) and len(display_cumulative) > 0:
                # plot(x, y): x 是累积值（显示在第二个 x 轴），y 是 token 位置（y 轴）
                # 确保都是 numpy 数组且形状一致
                plot_x = np.asarray(display_cumulative).flatten()
                plot_y = np.asarray(y_pos).flatten()
                if len(plot_x) == len(plot_y):
                    ax_right_twin.plot(plot_x, plot_y, 'o-', color='green', 
                                      linewidth=2, markersize=6, label='Cumulative SHAP', alpha=0.8)
            ax_right_twin.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
            ax_right_twin.set_xlabel('Cumulative SHAP Value', fontsize=11, color='green')
            ax_right_twin.tick_params(axis='x', labelcolor='green')
            
            # 设置标签
            ax_right.set_yticks(y_pos)
            ax_right.set_yticklabels(display_tokens, fontsize=10)
            ax_right.set_xlabel('SHAP Value (Contribution)', fontsize=12)
            ax_right.set_title(f'Dynamic SHAP Visualization · {predicted_label}', 
                              fontsize=14, pad=12)
            ax_right.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
            ax_right.grid(True, alpha=0.3, axis='x')
            
            # 添加图例
            ax_right.legend(loc='upper left')
            ax_right_twin.legend(loc='upper right')
            
            # 使用 subplots_adjust 代替 tight_layout，避免与 twinx 冲突
            # 固定边距以确保所有帧尺寸一致
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3)
            
            buf = BytesIO()
            # 移除 bbox_inches="tight"，使用固定尺寸确保所有帧一致
            # 使用固定的 figsize 和 pad_inches 确保所有帧尺寸相同
            fig.savefig(buf, format="png", dpi=120, bbox_inches=None, pad_inches=0.1, facecolor='white')
            buf.seek(0)
            frame_img = imageio.v2.imread(buf)
            frames.append(frame_img)
            plt.close(fig)
        
        # 统一所有帧的尺寸（确保所有帧尺寸完全一致）
        if len(frames) > 0:
            # 获取第一帧的尺寸作为标准
            target_shape = frames[0].shape
            print(f"[DEBUG] 目标帧尺寸: {target_shape}")
            
            # 统一所有帧的尺寸
            normalized_frames = []
            for idx, frame in enumerate(frames):
                if frame.shape != target_shape:
                    print(f"[WARN] 帧 {idx} 尺寸不匹配: {frame.shape} != {target_shape}, 正在调整...")
                    # 使用 PIL 调整尺寸
                    try:
                        # 将 numpy 数组转换为 PIL Image
                        if len(frame.shape) == 3:
                            pil_img = Image.fromarray(frame)
                            # 调整到目标尺寸
                            pil_img = pil_img.resize((target_shape[1], target_shape[0]), Image.Resampling.LANCZOS)
                            frame = np.array(pil_img)
                        else:
                            print(f"[WARN] 帧 {idx} 形状异常: {frame.shape}, 跳过")
                            continue
                    except Exception as e:
                        print(f"[ERROR] 调整帧 {idx} 尺寸时出错: {e}")
                        continue
                normalized_frames.append(frame)
            
            if len(normalized_frames) == 0:
                print("[ERROR] 没有有效的帧可以保存")
                return None
            
            filename = os.path.join(
                ANIMATION_DIR,
                f"token_animation_{int(time.time() * 1000)}.gif"
            )
            save_kwargs = {
                "duration": frame_duration
            }
            if loop:
                save_kwargs["loop"] = 0  # 0 表示无限循环
            imageio.mimsave(
                filename, 
                normalized_frames, 
                **save_kwargs
            )
            print(f"[DEBUG] 动画已保存: {filename}, 共 {len(normalized_frames)} 帧")
            return filename
        else:
            print("[WARN] 没有生成任何帧")
            return None
    except Exception as e:
        print(f"[ERROR] 生成动画时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def format_prediction_result(result):
    """
    格式化预测结果用于显示
    """
    if result["error"]:
        return f"错误: {result['error']}"
    
    output = f"""
## 预测结果

**类别**: {result['label']}
**置信度**: {result['confidence']:.2%}

### 概率分布
- **{LABELS[0]}**: {result['probabilities'][LABELS[0]]:.2%}
- **{LABELS[1]}**: {result['probabilities'][LABELS[1]]:.2%}

---
{result['explanation']}
"""
    return output


def prepare_contribution_table(contributions, top_n=30):
    if not contributions:
        return None
    
    ordered = sorted(contributions, key=lambda x: abs(x["value"]), reverse=True)
    limited = ordered[:top_n]
    rows = []
    for idx, item in enumerate(limited, 1):
        rows.append([idx, item["token"], round(item["value"], 4), item["direction"]])
    return rows


def create_shap_bar_plot(contributions, top_n=20, base_value=0.5):
    """
    创建 SHAP 条形图（改回之前的样式，从0开始绘制）
    
    Args:
        contributions: 词汇贡献列表
        top_n: 显示前 N 个最重要的词
        base_value: 基础值（通常为0.5，表示中性概率）
    
    Returns:
        matplotlib figure 对象
    """
    if not contributions:
        return None
    
    try:
        # 按绝对值排序，取前 top_n 个
        ordered = sorted(contributions, key=lambda x: abs(x["value"]), reverse=True)
        limited = ordered[:top_n]
        
        if not limited:
            return None
        
        tokens = [item["token"] for item in limited]
        values = [item["value"] for item in limited]
        
        # 确保 tokens 和 values 长度一致
        if len(tokens) != len(values):
            min_len = min(len(tokens), len(values))
            tokens = tokens[:min_len]
            values = values[:min_len]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, max(6, len(tokens) * 0.4)))
        
        # 计算累积值（用于箭头式位移展示）
        cumulative_values = np.cumsum(values)
        final_value = base_value + cumulative_values[-1] if len(cumulative_values) > 0 else base_value
        
        # 绘制基础值线（灰色虚线）
        ax.axvline(x=base_value, color='gray', linestyle='--', linewidth=1.5, label='基础值 (0.5)')
        
        # 绘制箭头式位移（从base_value到最终值）
        y_pos = np.arange(len(tokens))
        
        # 绘制每个词的贡献条形图（从0开始，而不是从base_value开始）
        colors = ['#ff4444' if v > 0 else '#4444ff' for v in values]
        ax.barh(y_pos, values, left=0, color=colors, alpha=0.7, label='词汇贡献')
        
        # 添加箭头显示累积效果
        if len(cumulative_values) > 0:
            # 在右侧添加箭头，显示从base_value到最终值的累积效果
            arrow_y = len(tokens) - 0.5
            arrow_start = base_value
            arrow_end = final_value
            arrow_length = arrow_end - arrow_start
            
            # 绘制箭头
            if abs(arrow_length) > 0.01:  # 只在有显著变化时绘制
                arrow_color = 'red' if arrow_length > 0 else 'blue'
                ax.annotate('', xy=(arrow_end, arrow_y), xytext=(arrow_start, arrow_y),
                           arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2.5, alpha=0.8))
                
                # 添加最终值标签
                ax.text(arrow_end, arrow_y + 0.3, f'Output值: {final_value:.3f}', 
                       fontsize=10, ha='center', color=arrow_color, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=arrow_color))
        
        # 设置标签
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens, fontsize=10)
        ax.set_xlabel('SHAP值 (对Positive类别的贡献)', fontsize=12)
        ax.set_title(f'Top {top_n} 词汇贡献（条形图）', fontsize=14, pad=12)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加图例
        ax.legend(loc='lower right', fontsize=9)
        
        # 使用 subplots_adjust 代替 tight_layout，更稳定
        plt.subplots_adjust(left=0.2, right=0.95, top=0.92, bottom=0.1)
        return fig
    except Exception as e:
        print(f"[WARN] 创建 SHAP 条形图失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_file_input(file_path):
    """
    处理文件输入，支持CSV和TXT格式
    
    Args:
        file_path: 上传的文件路径
    
    Returns:
        tuple: (success, results, error_message, output_file_path)
    """
    try:
        import csv
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return False, None, "文件不存在", None
        
        # 获取文件扩展名
        file_ext = os.path.splitext(file_path)[1].lower()
        
        results = []
        texts = []
        
        # 根据文件类型读取
        if file_ext == '.csv':
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                # 尝试找到文本列（通常名为 'text', 'review', 'comment' 等）
                text_column = None
                for col in df.columns:
                    if col.lower() in ['text', 'review', 'comment', 'sentence', 'content']:
                        text_column = col
                        break
                
                if text_column is None:
                    # 如果没有找到，使用第一列
                    text_column = df.columns[0]
                
                texts = df[text_column].astype(str).tolist()
            except Exception as e:
                return False, None, f"CSV文件读取失败: {str(e)}", None
        
        elif file_ext == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts = [line.strip() for line in f if line.strip()]
            except Exception as e:
                return False, None, f"TXT文件读取失败: {str(e)}", None
        
        else:
            return False, None, f"不支持的文件格式: {file_ext}。支持格式: .csv, .txt", None
        
        if not texts:
            return False, None, "文件中没有找到有效的文本数据", None
        
        # 批量预测
        if current_model is None or current_tokenizer is None:
            return False, None, "请先加载模型！", None
        
        print(f"[INFO] 开始处理 {len(texts)} 条文本...")
        for i, text in enumerate(texts):
            if not text or len(text.strip()) == 0:
                continue
            
            result = predict_text(text, use_advanced_preprocessing=True)
            if result.get("error"):
                results.append({
                    "index": i + 1,
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "label": "Error",
                    "confidence": 0.0,
                    "error": result["error"]
                })
            else:
                results.append({
                    "index": i + 1,
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "label": result["label"],
                    "confidence": result["confidence"],
                    "prob_negative": result["probabilities"][LABELS[0]],
                    "prob_positive": result["probabilities"][LABELS[1]]
                })
        
        # 保存结果到CSV文件
        output_dir = "./visualizations/gradio"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"batch_results_{int(time.time())}.csv")
        
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"[INFO] 处理完成，结果已保存到: {output_file}")
        return True, results, None, output_file
    
    except Exception as e:
        error_msg = f"文件处理失败: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return False, None, error_msg, None


def build_model_info_md(model_name):
    info = MODEL_SUMMARY.get(model_name)
    if not info:
        return "（暂无模型概览）"
    
    bullet_md = "\n".join([f"- {item}" for item in info["bullets"]])
    return f"""
### {info['title']} 模型特点
{bullet_md}
"""


# --- Gradio 界面定义 ---

def create_interface():
    """
    创建 Gradio 界面：
    - 左侧：模型选择/加载 + 状态 + 模型简介
    - 右侧：文本输入与预测/清空
    - 下方：预测结果、概率、SHAP 条形图、逐词动画 + 重播
    """
    
    # 尝试使用最简单的主题，避免主题导致的问题
    with gr.Blocks(title="文本情感分类 - DistilBERT 模型演示", css=CUSTOM_CSS) as demo:
        gr.Markdown("""
        # DistilBERT 文本情感分类系统
        
        这是一个基于轻量级预训练模型 DistilBERT 的文本情感分类演示系统。
        
        **功能特点**:
        - 支持三种不同的模型（Baseline、LoRA、LoRA+数据增强）
        - 实时文本情感分析
        - 概率分布可视化
        - 可解释性分析
        
        **使用说明**:
        1. 首先选择一个已训练的模型
        2. 点击“加载模型”按钮
        3. 在文本框中输入要分析的文本（如电影评论）
        4. 点击“预测”按钮查看结果
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 模型选择")
                
                # 显示所有模型的指标
                gr.Markdown("**模型性能指标对比：**")
                metrics_text = "\n\n".join([
                    f"**{name}**: F1={info.get('metrics', {}).get('F1', 0):.4f}, "
                    f"Accuracy={info.get('metrics', {}).get('Accuracy', 0):.4f}, "
                    f"参数量={info.get('metrics', {}).get('参数量', 'N/A')}"
                    for name, info in MODEL_PATHS.items()
                    if isinstance(info, dict)
                ])
                model_metrics_display = gr.Markdown(value=metrics_text, visible=True)
                
                model_radio = gr.Radio(
                    choices=list(MODEL_PATHS.keys()),
                    value=None,  # 初始不选择任何模型
                    label="选择模型",
                    info="点击选择模型，再次点击可取消选择"
                )
                load_btn = gr.Button("加载模型", variant="primary")
                model_status = gr.Textbox(
                    label="模型状态",
                    value="请先选择模型并点击加载按钮",
                    interactive=False,
                    lines=3
                )
                model_info_box = gr.Markdown(
                    value="请选择一个模型查看详细信息"
                )
            
            with gr.Column(scale=2):
                # 使用Tabs区分语句输入和文件输入
                with gr.Tabs():
                    with gr.Tab("语句输入"):
                        gr.Markdown("### 文本输入")
                        text_input = gr.Textbox(
                            label="输入文本",
                            placeholder="请输入要分析的文本，例如：\nThis movie is absolutely amazing! I loved every minute of it.",
                            lines=5,
                            info="支持输入英文电影评论或其他文本"
                        )
                        
                        # 例句区域
                        with gr.Row():
                            example_btn = gr.Button("使用例句", variant="secondary", size="sm")
                            rotate_example_btn = gr.Button("轮换例句", variant="secondary", size="sm")
                        current_example = gr.Textbox(
                            label="当前例句",
                            value="",
                            interactive=False,
                            lines=2,
                            visible=False
                        )
                        
                        with gr.Row():
                            predict_btn = gr.Button("预测", variant="primary", scale=2)
                            clear_btn = gr.Button("清空", scale=1)
                    
                    with gr.Tab("文件输入"):
                        gr.Markdown("### 批量文件处理")
                        gr.Markdown("""
                        **支持的文件格式：**
                        - CSV文件：必须包含文本列（列名可以是 'text', 'review', 'comment', 'sentence', 'content' 或第一列）
                        - TXT文件：每行一个文本样本
                        
                        **处理说明：**
                        - 文件输入将批量处理所有文本
                        - 处理完成后可下载结果CSV文件
                        - 文件输入不提供可视化展示
                        """)
                        file_input = gr.File(
                            label="上传文件",
                            file_types=[".csv", ".txt"]
                        )
                        file_process_btn = gr.Button("处理文件", variant="primary")
                        file_status = gr.Textbox(
                            label="处理状态",
                            value="请上传文件并点击处理",
                            interactive=False,
                            lines=3
                        )
                        file_download = gr.File(
                            label="下载处理结果",
                            visible=False
                        )
        
        # 结果展示区域（仅用于语句输入）
        with gr.Row(visible=True) as results_row:
            with gr.Column():
                gr.Markdown("### 预测结果与可视化（仅语句输入）")
                result_output = gr.Markdown(label="结果")
                
                gr.Markdown("### SHAP 可解释性分析")
                gr.Markdown("""
                **图表说明：**
                - **红色条形**：推动预测为 Positive（积极）的词汇贡献
                - **蓝色条形**：推动预测为 Negative（消极）的词汇贡献
                - **Output值**：最终预测概率（>0.5为Positive，<0.5为Negative）
                """)
                
                # 使用 Plot 组件显示条形图
                shap_bar_plot = gr.Plot(
                    label="Top 20 词汇贡献（条形图 + 箭头式位移）",
                    show_label=True
                )
                
                gr.Markdown("### 逐词情绪动画")
                animation_view = gr.Image(
                    label="逐词情绪动画（播放一次后停止，包含动态 SHAP 可视化）",
                    type="filepath",
                    elem_id="animation-view",
                    height=420
                )
                replay_btn = gr.Button(
                    "重播动画",
                    variant="secondary",
                    size="sm",
                    elem_classes=["replay-btn"]
                )
                gr.Markdown("（动画按词序稳定播放，左侧高亮当前词，右侧展示 SHAP 值的动态变化。播放完成后可点击「重播动画」按钮重新播放。）")
        
        # 示例文本（暂时注释掉，用于调试）
        # gr.Markdown("### 示例文本")
        # example_texts = [
        #     "This movie is absolutely amazing! I loved every minute of it. The acting was brilliant and the plot was engaging.",
        #     "I hated this movie. It was a complete waste of time. The acting was terrible and the story made no sense.",
        #     "The movie was okay. Some parts were good, but overall it was just average.",
        #     "This is one of the best films I've ever seen. Truly brilliant and moving.",
        #     "I was disappointed by this film. The trailer looked promising, but the actual movie was boring."
        # ]
        # 
        # gr.Examples(
        #     examples=example_texts,
        #     inputs=text_input,
        #     label="点击示例快速填充"
        # )
        
        # ========== 事件绑定（必须在所有组件定义之后）==========
        
        # 全局变量用于例句轮换
        current_example_index = [0]  # 使用列表以便在函数中修改
        
        def get_random_example():
            """获取随机例句"""
            import random
            return random.choice(EXAMPLE_SENTENCES)
        
        def rotate_example():
            """轮换例句"""
            current_example_index[0] = (current_example_index[0] + 1) % len(EXAMPLE_SENTENCES)
            example = EXAMPLE_SENTENCES[current_example_index[0]]
            return example, example
        
        def use_example():
            """使用当前例句"""
            example = EXAMPLE_SENTENCES[current_example_index[0]]
            return example
        
        def on_file_process(file):
            """处理文件输入"""
            if file is None:
                return "请先上传文件！", None, gr.update(visible=False)
            
            # Gradio文件对象处理
            if isinstance(file, str):
                file_path = file
            elif hasattr(file, 'name'):
                file_path = file.name
            else:
                file_path = str(file)
            
            success, results, error_msg, output_file = process_file_input(file_path)
            
            if success:
                status_msg = f"处理完成！共处理 {len(results)} 条文本。\n结果已保存，请下载结果文件。"
                return status_msg, gr.update(visible=True, value=output_file), gr.update(visible=True)
            else:
                return f"处理失败: {error_msg}", gr.update(visible=False), gr.update(visible=False)
        
        def on_load_model(model_name):
            print(f"[DEBUG] ========== on_load_model 被触发 ==========")
            import sys
            sys.stdout.flush()  # 强制刷新输出
            print(f"[DEBUG] 接收到的 model_name 参数: {repr(model_name)}")
            print(f"[DEBUG] 参数类型: {type(model_name)}")
            sys.stdout.flush()
            
            # 如果model_name为None，表示取消选择
            if model_name is None:
                return "未选择模型，请先选择一个模型", "请选择一个模型查看详细信息"
            
            try:
                status = load_model_for_inference(model_name)
                info_md = build_model_info_md(model_name)
                print(f"[DEBUG] on_load_model 返回成功")
                sys.stdout.flush()
                return status, info_md
            except Exception as e:
                error_msg = f"加载模型时发生异常: {str(e)}"
                print(f"[ERROR] {error_msg}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                return error_msg, "（模型信息加载失败）"
        
        def on_predict(text):
            print(f"[DEBUG] ========== on_predict 被触发 ==========")
            import sys
            sys.stdout.flush()  # 强制刷新输出
            print(f"[DEBUG] 接收到的 text 参数: {repr(text)}")
            print(f"[DEBUG] 文本长度: {len(text) if text else 0}")
            sys.stdout.flush()
            
            empty_bar_plot = None
            empty_image = None
            
            if not text or len(text.strip()) == 0:
                print(f"[DEBUG] 文本为空，返回提示")
                return "请输入文本！", empty_bar_plot, empty_image
            
            if current_model is None or current_tokenizer is None:
                error_msg = "请先加载模型！"
                print(f"[ERROR] {error_msg}")
                return error_msg, empty_bar_plot, empty_image
            
            try:
                print(f"[DEBUG] 开始预测...")
                result = predict_text(text, use_advanced_preprocessing=True)
                print(f"[DEBUG] 预测完成，结果: {result.get('label', 'N/A')}")
                
                if result["error"]:
                    error_msg = f"错误: {result['error']}"
                    print(f"[ERROR] {error_msg}")
                    return error_msg, empty_bar_plot, empty_image
                
                formatted_result = format_prediction_result(result)
                chart_data = {
                    "类别": [LABELS[0], LABELS[1]],
                    "概率": [
                        result["probabilities"][LABELS[0]],
                        result["probabilities"][LABELS[1]]
                    ]
                }
                contrib_table = prepare_contribution_table(result.get("token_contributions"))
                animation_path = result.get("animation")
                
                # 创建 SHAP 条形图（使用改进后的版本，包含箭头式位移）
                shap_bar_fig = None
                if result.get("token_contributions"):
                    # 计算base_value（从概率反推）
                    prob_positive = result["probabilities"][LABELS[1]]
                    # base_value通常是0.5（中性），但我们可以从最终概率和贡献值反推
                    base_value = 0.5  # 使用标准base value
                    shap_bar_fig = create_shap_bar_plot(result["token_contributions"], top_n=20, base_value=base_value)
                
                # 动画使用文件路径（gr.Image 需要文件路径）
                image_update = animation_path if animation_path else None
                
                # 保存最后一次生成的动画路径，用于重播
                global last_animation_path
                last_animation_path = animation_path
                
                print(f"[DEBUG] on_predict 返回成功")
                return formatted_result, shap_bar_fig, image_update
            except Exception as e:
                error_msg = f"预测过程中发生异常: {str(e)}"
                print(f"[ERROR] {error_msg}")
                import traceback
                traceback.print_exc()
                return error_msg, empty_bar_plot, empty_image
        
        print("[DEBUG] 开始绑定事件处理器...")
        
        # 初始化例句
        initial_example = EXAMPLE_SENTENCES[0]
        
        # 绑定例句相关按钮
        example_btn.click(
            fn=use_example,
            outputs=text_input
        )
        
        rotate_example_btn.click(
            fn=rotate_example,
            outputs=[text_input, current_example]
        )
        
        # 绑定文件处理按钮
        file_process_btn.click(
            fn=on_file_process,
            inputs=file_input,
            outputs=[file_status, file_download]
        )
        
        # 绑定加载模型按钮
        print("[DEBUG] 绑定加载模型按钮...")
        load_btn.click(
            fn=on_load_model,
            inputs=model_radio,
            outputs=[model_status, model_info_box]
        )
        print("[DEBUG] load_btn.click 绑定完成")
        
        # 绑定预测按钮
        print("[DEBUG] 绑定预测按钮...")
        predict_btn.click(
            fn=on_predict,
            inputs=text_input,
            outputs=[result_output, shap_bar_plot, animation_view]
        )
        print("[DEBUG] predict_btn.click 绑定完成")
        
        # 绑定清空按钮
        def on_clear():
            print("[DEBUG] ========== on_clear 被触发 ==========")
            global last_animation_path
            last_animation_path = None
            return (
                "",  # text_input
                "",  # result_output (Markdown)
                None,  # shap_bar_plot
                None  # animation_view
            )
        
        print("[DEBUG] 绑定清空按钮...")
        clear_btn.click(
            fn=on_clear,
            outputs=[text_input, result_output, shap_bar_plot, animation_view]
        )
        print("[DEBUG] clear_btn.click 绑定完成")
        
        # 绑定重播动画按钮（复用上一份 GIF，复制为新文件触发前端刷新）
        def on_replay_animation():
            """
            重播最后一次生成的动画
            """
            global last_animation_path
            print(f"[DEBUG] ========== on_replay_animation 被触发 ==========")
            if last_animation_path and os.path.exists(last_animation_path):
                try:
                    new_path = os.path.join(
                        ANIMATION_DIR,
                        f"token_animation_replay_{int(time.time() * 1000)}.gif"
                    )
                    with Image.open(last_animation_path) as gif:
                        frames = []
                        durations = []
                        try:
                            while True:
                                frames.append(gif.copy())
                                durations.append(gif.info.get("duration", 800))
                                gif.seek(gif.tell() + 1)
                        except EOFError:
                            pass
                    if frames:
                        save_kwargs = {
                            "save_all": True,
                            "append_images": frames[1:],
                            "duration": durations
                        }
                        # 不设置 loop，保持播放一次后停止
                        frames[0].save(
                            new_path,
                            **save_kwargs
                        )
                    else:
                        shutil.copy(last_animation_path, new_path)
                    last_animation_path = new_path
                    print(f"[DEBUG] 重播动画: {new_path}")
                    return new_path
                except Exception as copy_err:
                    print(f"[ERROR] 重播动画复制失败: {copy_err}")
                    return last_animation_path
            else:
                print("[WARN] 没有可重播的动画")
                return None
        
        print("[DEBUG] 绑定重播动画按钮...")
        replay_btn.click(
            fn=on_replay_animation,
    outputs=animation_view
        )
        print("[DEBUG] replay_btn.click 绑定完成")
        
        # 页面加载时自动加载默认模型（暂时禁用，先测试手动点击）
        def on_demo_load(model_name):
            print(f"[DEBUG] ========== demo.load 被触发 ==========")
            print(f"[DEBUG] 默认模型: {model_name}")
            import sys
            sys.stdout.flush()
            return on_load_model(model_name)
        
        # 暂时注释掉自动加载，先测试手动点击
        # demo.load(
        #     fn=on_demo_load,
        #     inputs=model_radio,
        #     outputs=[model_status, model_info_box]
        # )
        print("[DEBUG] demo.load 已禁用（用于调试）")
        print("[DEBUG] 所有事件绑定完成")
    
    return demo


# --- 主程序入口 ---
if __name__ == "__main__":
    print("=" * 60)
    print("启动 Gradio 文本情感分类演示系统")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"可用模型: {list(MODEL_PATHS.keys())}")
    
    # 检查模型路径
    print("\n[DEBUG] 检查模型路径:")
    for name, info in MODEL_PATHS.items():
        if isinstance(info, dict):
            path = info.get("path", "")
        else:
            path = info
        abs_path = os.path.abspath(path)
        exists = os.path.exists(path)
        print(f"  {name}:")
        print(f"    相对路径: {path}")
        print(f"    绝对路径: {abs_path}")
        print(f"    存在: {exists}")
        if exists:
            files = os.listdir(path) if os.path.isdir(path) else []
            print(f"    文件数: {len(files)}")
    
    print("\n[DEBUG] 检查动画目录:")
    abs_anim_dir = os.path.abspath(ANIMATION_DIR)
    print(f"  相对路径: {ANIMATION_DIR}")
    print(f"  绝对路径: {abs_anim_dir}")
    print(f"  存在: {os.path.exists(ANIMATION_DIR)}")
    
    print("=" * 60)
    print("[DEBUG] 开始创建 Gradio 界面...")
    
    demo = create_interface()
    
    print("[DEBUG] Gradio 界面创建完成")
    print("[DEBUG] 准备启动服务器...")
    print(f"[DEBUG] 服务器地址: http://127.0.0.1:7860")
    
    # 启用队列以提高稳定性
    try:
        demo.queue()
        print("[DEBUG] 队列启用成功")
    except Exception as e:
        print(f"[WARN] 队列启用失败: {e}")
    
    print("[DEBUG] 启动服务器...")
    print("[DEBUG] 如果按钮无反应，请检查：")
    print("  1. 浏览器控制台是否有 JavaScript 错误")
    print("  2. 网络请求是否被拦截")
    print("  3. 是否使用了无痕/隐身模式")
    
    demo.launch(
        server_name="127.0.0.1",  # Windows 本地环境更稳定
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=False  # 不自动打开浏览器，手动打开更稳定
    )


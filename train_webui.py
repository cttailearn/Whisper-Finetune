import whisper
import gradio as gr
import os
import tempfile
import warnings
from whisper.tokenizer import LANGUAGES
import subprocess
import threading
import time
import logging
from pathlib import Path
import json
import pandas as pd
import traceback
import random
import soundfile
from tqdm import tqdm
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import argparse
import functools
import gc
import platform
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizerFast, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from peft import PeftModel, PeftConfig
import re
import shutil
import psutil  # 新增用于磁盘空间检查
import signal

# Faster Whisper 相关导入
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("Warning: faster-whisper not available. Install with: pip install faster-whisper")

# VAD 相关导入
try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    print("Warning: webrtcvad not available. Install with: pip install webrtcvad")

# Silero VAD 检查（延迟加载，不在启动时验证）
SILERO_VAD_AVAILABLE = False
try:
    import torch
    # 只检查基础依赖，不在启动时加载模型
    SILERO_VAD_AVAILABLE = True
    print("PyTorch可用，Silero VAD功能已启用（将在首次使用时加载模型）")
except ImportError:
    print("Warning: PyTorch不可用，无法使用Silero VAD")
    SILERO_VAD_AVAILABLE = False

# CT2 转换器导入
try:
    import ct2_transformers_converter
    CT2_CONVERTER_AVAILABLE = True
except ImportError:
    CT2_CONVERTER_AVAILABLE = False
    print("Warning: ct2-transformers-converter not available. Install with: pip install ct2-transformers-converter")

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 忽略不必要的警告
warnings.filterwarnings("ignore")

# 设置临时目录和缓存目录
temp_dir = tempfile.mkdtemp()
os.environ['GRADIO_TEMP_DIR'] = temp_dir
os.environ['GRADIO_CACHE_DIR'] = os.path.join(temp_dir, 'gradio_cache')

# 模型缓存字典，避免重复加载模型
model_cache = {}

# Silero VAD 模型缓存
silero_vad_model = None
silero_vad_utils = None

# 训练状态
TRAINING_PROCESS = None
TRAINING_THREAD = None
OUTPUT_FILE = None
TRAINING_ACTIVE = False

# ============================== VAD 语音活动检测功能 ==============================
def apply_vad_filter(audio_path, vad_method="silero", aggressiveness=3):
    """应用VAD过滤去除静音片段"""
    try:
        import librosa
        
        # 读取音频文件
        audio, sr = librosa.load(audio_path, sr=16000)
        
        if vad_method == "webrtc" and VAD_AVAILABLE:
            print("使用WebRTC VAD进行语音活动检测...")
            filtered_audio_path = apply_webrtc_vad(audio, sr, aggressiveness)
            if filtered_audio_path:
                print("WebRTC VAD过滤完成")
                return filtered_audio_path
            else:
                print("WebRTC VAD过滤失败，使用原始音频")
                return audio_path
        elif vad_method == "silero" and SILERO_VAD_AVAILABLE:
            print("使用Silero VAD进行语音活动检测...")
            filtered_audio_path = apply_silero_vad(audio, sr)
            if filtered_audio_path:
                print("Silero VAD过滤完成")
                return filtered_audio_path
            else:
                print("Silero VAD过滤失败，使用原始音频")
                return audio_path
        else:
            # 如果VAD不可用，返回原始音频
            if vad_method == "silero":
                if not SILERO_VAD_AVAILABLE:
                    print("VAD方法 silero 不可用：PyTorch未安装或不可用")
                    print("建议：pip install torch 或使用 webrtc VAD 作为替代")
                else:
                    print("VAD方法 silero 暂时不可用，可能是网络问题")
            elif vad_method == "webrtc":
                print("VAD方法 webrtc 不可用：webrtcvad包未安装")
                print("建议：pip install webrtcvad")
            else:
                print(f"未知的VAD方法: {vad_method}")
            
            print("VAD过滤未生效，使用原始音频")
            return audio_path
            
    except Exception as e:
        print(f"VAD过滤失败: {e}")
        print("使用原始音频")
        return audio_path

def apply_webrtc_vad(audio, sr, aggressiveness=3):
    """使用WebRTC VAD进行语音活动检测"""
    try:
        import webrtcvad
        import struct
        
        vad = webrtcvad.Vad(aggressiveness)
        
        # 转换为16位PCM格式
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # WebRTC VAD需要特定的帧长度（10ms, 20ms, 30ms）
        frame_duration = 30  # ms
        frame_length = int(sr * frame_duration / 1000)
        
        voiced_frames = []
        for i in range(0, len(audio_int16) - frame_length, frame_length):
            frame = audio_int16[i:i + frame_length]
            frame_bytes = struct.pack(f'{len(frame)}h', *frame)
            
            if vad.is_speech(frame_bytes, sr):
                voiced_frames.extend(frame)
        
        if voiced_frames:
            # 保存过滤后的音频
            filtered_audio = np.array(voiced_frames, dtype=np.int16) / 32767.0
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=temp_dir) as tmpfile:
                output_path = tmpfile.name
            
            import soundfile as sf
            sf.write(output_path, filtered_audio, sr)
            return output_path
        else:
            print("WebRTC VAD: 未检测到语音活动")
            return None
            
    except Exception as e:
        print(f"WebRTC VAD处理失败: {e}")
        return None

def apply_silero_vad(audio, sr):
    """使用Silero VAD进行语音活动检测"""
    global silero_vad_model, silero_vad_utils
    
    try:
        import torch
        
        # 检查torch是否可用
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            # 如果没有GPU，使用CPU
            device = torch.device('cpu')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        
        # 如果模型还未加载，尝试加载
        if silero_vad_model is None or silero_vad_utils is None:
            print("首次使用Silero VAD，正在加载模型...")
            
            # 尝试加载Silero VAD模型
            try:
                model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                              model='silero_vad',
                                              force_reload=False,
                                              trust_repo=True)
                silero_vad_model = model.to(device)
                silero_vad_utils = utils
                print("Silero VAD模型加载成功")
            except Exception as hub_error:
                print(f"无法从torch.hub加载Silero VAD模型: {hub_error}")
                # 尝试本地加载或使用替代方法
                try:
                    # 尝试直接导入silero_vad
                    import silero_vad
                    model, utils = silero_vad.load_silero_vad()
                    silero_vad_model = model
                    silero_vad_utils = utils
                    print("本地Silero VAD包加载成功")
                except Exception as local_error:
                    print(f"本地Silero VAD加载也失败: {local_error}")
                    print("建议：")
                    print("1. 检查网络连接")
                    print("2. 安装本地silero-vad包: pip install silero-vad")
                    print("3. 或者使用WebRTC VAD作为替代")
                    return None
        
        # 使用缓存的模型和工具
        model = silero_vad_model
        utils = silero_vad_utils
        
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
        
        # 确保音频是torch tensor格式
        if not isinstance(audio, torch.Tensor):
            audio_tensor = torch.from_numpy(audio).float().to(device)
        else:
            audio_tensor = audio.to(device)
        
        # 获取语音时间戳
        speech_timestamps = get_speech_timestamps(audio_tensor, model, sampling_rate=sr)
        
        if speech_timestamps:
            # 收集语音片段
            voiced_audio = collect_chunks(speech_timestamps, audio_tensor)
            
            # 转换回numpy数组
            if isinstance(voiced_audio, torch.Tensor):
                voiced_audio = voiced_audio.cpu().numpy()
            
            # 保存过滤后的音频
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=temp_dir) as tmpfile:
                output_path = tmpfile.name
            
            import soundfile as sf
            sf.write(output_path, voiced_audio, sr)
            print(f"Silero VAD处理完成，检测到 {len(speech_timestamps)} 个语音片段")
            return output_path
        else:
            print("Silero VAD: 未检测到语音活动")
            return None
            
    except Exception as e:
        print(f"Silero VAD处理失败: {e}")
        print(f"错误详情: {traceback.format_exc()}")
        return None

# ============================== CT2 模型转换功能 ==============================
def convert_model_to_ct2(model_path, output_dir, quantization="float16"):
    """将模型转换为CT2格式"""
    if not CT2_CONVERTER_AVAILABLE:
        return False, "ct2-transformers-converter 未安装"
    
    try:
        import ct2_transformers_converter
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 执行转换
        ct2_transformers_converter.convert_model(
            model_path,
            output_dir,
            quantization=quantization,
            force=True
        )
        
        return True, f"模型已成功转换为CT2格式并保存到: {output_dir}"
        
    except Exception as e:
        return False, f"CT2转换失败: {str(e)}"

# ============================== 多模型管理功能 ==============================
class MultiModelManager:
    """多模型管理器，支持同时加载和对比多个模型"""
    
    def __init__(self):
        self.models = {}
        self.model_types = {}
    
    def add_model(self, name, model_path, model_type="whisper", **kwargs):
        """添加模型到管理器"""
        try:
            if model_type == "faster_whisper" and FASTER_WHISPER_AVAILABLE:
                model = FasterWhisperModel(model_path, **kwargs)
            elif model_type == "whisper":
                model = load_model(model_path, kwargs.get('download_root'), kwargs.get('is_local_model', False))
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            self.models[name] = model
            self.model_types[name] = model_type
            return True, f"模型 {name} 加载成功"
            
        except Exception as e:
            return False, f"模型 {name} 加载失败: {str(e)}"
    
    def remove_model(self, name):
        """从管理器中移除模型"""
        if name in self.models:
            del self.models[name]
            del self.model_types[name]
            return True, f"模型 {name} 已移除"
        return False, f"模型 {name} 不存在"
    
    def transcribe_with_all_models(self, audio_path, task="transcribe", language=None):
        """使用所有加载的模型进行转录对比"""
        results = {}
        
        for name, model in self.models.items():
            try:
                model_type = self.model_types[name]
                
                if model_type == "faster_whisper":
                    # Faster Whisper 模型
                    segments, info = model.transcribe(audio_path, task=task, language=language)
                    text = " ".join([segment.text for segment in segments])
                    results[name] = {
                        "text": text,
                        "language": info.language if hasattr(info, 'language') else "unknown",
                        "status": "success"
                    }
                else:
                    # 标准 Whisper 模型
                    if hasattr(model, 'pipe'):  # TransformersWhisperModel
                        generate_kwargs = {"task": task}
                        if language:
                            generate_kwargs["language"] = language
                        result = model.pipe(audio_path, return_timestamps=True, generate_kwargs=generate_kwargs)
                        text = result.get("text", "")
                        detected_lang = language if language else "auto"
                    else:
                        result = model.transcribe(audio_path, task=task, language=language)
                        text = result["text"]
                        detected_lang = result.get("language", "unknown")
                    
                    results[name] = {
                        "text": text,
                        "language": detected_lang,
                        "status": "success"
                    }
                    
            except Exception as e:
                results[name] = {
                    "text": "",
                    "language": "unknown",
                    "status": "error",
                    "error": str(e)
                }
        
        return results
    
    def get_model_list(self):
        """获取已加载的模型列表"""
        return {name: {"model_type": self.model_types.get(name, "unknown")} for name in self.models.keys()}

# 全局多模型管理器实例
multi_model_manager = MultiModelManager()

# ============================== 步骤1: 数据准备功能 ==============================
def data_prep_get_columns(input_file):
    """获取文件的列名"""
    if input_file is None:
        return []
    try:
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file, nrows=0)
        elif input_file.endswith('.xls'):
            df = pd.read_excel(input_file, engine='xlrd', nrows=0)
        else:
            df = pd.read_excel(input_file, engine='openpyxl', nrows=0)
        return df.columns.tolist()
    except Exception as e:
        print(f"Error reading file: {e}")
        traceback.print_exc()
        return []

def check_disk_space(path, required_mb=100):
    """检查磁盘空间是否足够"""
    try:
        if platform.system() == 'Windows':
            free_bytes = psutil.disk_usage(path).free
        else:
            stat = os.statvfs(path)
            free_bytes = stat.f_frsize * stat.f_bavail
        
        free_mb = free_bytes / (1024 * 1024)
        return free_mb > required_mb
    except Exception as e:
        print(f"无法检查磁盘空间: {str(e)}")
        return True  # 如果无法检查，假设空间足够

def save_dataset(results, output_dir, enable_split, train_ratio, enable_final_test, final_test_ratio):
    """保存数据集到文件（增强错误处理，支持三个数据集划分）"""
    try:
        # 检查输出目录是否存在且可写
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        if not os.access(output_dir, os.W_OK):
            return None, f"错误：目录 {output_dir} 不可写！"
        
        # 检查磁盘空间（至少需要100MB）
        if not check_disk_space(output_dir, 100):
            return None, f"错误：磁盘空间不足（需要至少100MB）！"
        
        # 空数据集检查
        if not results:
            return None, "警告：处理后的数据集为空，跳过文件生成！"
        
        # 随机打乱数据（无论是否划分数据集都进行打乱）
        random.shuffle(results)
        
        # 文件保存逻辑
        if enable_split:
            total_count = len(results)
            
            if enable_final_test:
                # 三个数据集划分：训练集、验证集、最终测试集
                final_test_count = int(total_count * final_test_ratio)
                remaining_count = total_count - final_test_count
                train_count = int(remaining_count * train_ratio)
                
                # 划分数据
                train_data = results[:train_count]
                val_data = results[train_count:remaining_count]
                final_test_data = results[remaining_count:]
                
                # 生成文件名（带时间戳避免覆盖）
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                train_file = os.path.join(output_dir, f"train_{timestamp}.jsonl")
                val_file = os.path.join(output_dir, f"val_{timestamp}.jsonl")
                final_test_file = os.path.join(output_dir, f"final_test_{timestamp}.jsonl")
                info_file = os.path.join(output_dir, f"dataset_info_{timestamp}.txt")
                
                output_files = [train_file, val_file, final_test_file, info_file]
                
                # 保存训练集
                try:
                    with open(train_file, 'w', encoding='utf-8') as f:
                        for i in range(0, len(train_data), 10000):
                            chunk = train_data[i:i+10000]
                            for item in chunk:
                                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    print(f"训练集保存成功: {train_file}")
                except IOError as e:
                    return None, f"写入训练集失败: {str(e)}"
                
                # 保存验证集
                try:
                    with open(val_file, 'w', encoding='utf-8') as f:
                        for i in range(0, len(val_data), 10000):
                            chunk = val_data[i:i+10000]
                            for item in chunk:
                                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    print(f"验证集保存成功: {val_file}")
                except IOError as e:
                    return None, f"写入验证集失败: {str(e)}"
                
                # 保存最终测试集
                try:
                    with open(final_test_file, 'w', encoding='utf-8') as f:
                        for i in range(0, len(final_test_data), 10000):
                            chunk = final_test_data[i:i+10000]
                            for item in chunk:
                                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    print(f"最终测试集保存成功: {final_test_file}")
                except IOError as e:
                    return None, f"写入最终测试集失败: {str(e)}"
                
                # 创建数据集信息
                try:
                    with open(info_file, 'w', encoding='utf-8') as f:
                        f.write(f"数据集划分信息\n")
                        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"总数据量: {total_count}\n")
                        f.write(f"训练集: {len(train_data)} ({len(train_data)/total_count*100:.1f}%)\n")
                        f.write(f"验证集: {len(val_data)} ({len(val_data)/total_count*100:.1f}%)\n")
                        f.write(f"最终测试集: {len(final_test_data)} ({final_test_ratio*100:.1f}%)\n")
                        f.write(f"训练集文件: {train_file}\n")
                        f.write(f"验证集文件: {val_file}\n")
                        f.write(f"最终测试集文件: {final_test_file}\n")
                    print(f"数据集信息保存成功: {info_file}")
                except IOError as e:
                    return None, f"写入数据集信息失败: {str(e)}"
            else:
                # 两个数据集划分：训练集、测试集
                train_count = int(total_count * train_ratio)
                
                # 划分数据
                train_data = results[:train_count]
                test_data = results[train_count:]
                
                # 生成文件名（带时间戳避免覆盖）
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                train_file = os.path.join(output_dir, f"train_{timestamp}.jsonl")
                test_file = os.path.join(output_dir, f"test_{timestamp}.jsonl")
                info_file = os.path.join(output_dir, f"dataset_info_{timestamp}.txt")
                
                output_files = [train_file, test_file, info_file]
                
                # 保存训练集（分批写入）
                try:
                    with open(train_file, 'w', encoding='utf-8') as f:
                        for i in range(0, len(train_data), 10000):
                            chunk = train_data[i:i+10000]
                            for item in chunk:
                                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    print(f"训练集保存成功: {train_file}")
                except IOError as e:
                    return None, f"写入训练集失败: {str(e)}"
                
                # 保存测试集（分批写入）
                try:
                    with open(test_file, 'w', encoding='utf-8') as f:
                        for i in range(0, len(test_data), 10000):
                            chunk = test_data[i:i+10000]
                            for item in chunk:
                                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    print(f"测试集保存成功: {test_file}")
                except IOError as e:
                    return None, f"写入测试集失败: {str(e)}"
                
                # 创建数据集信息
                try:
                    with open(info_file, 'w', encoding='utf-8') as f:
                        f.write(f"数据集划分信息\n")
                        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"总数据量: {total_count}\n")
                        f.write(f"训练集: {len(train_data)} ({train_ratio*100:.1f}%)\n")
                        f.write(f"测试集: {len(test_data)} ({(1-train_ratio)*100:.1f}%)\n")
                        f.write(f"训练集文件: {train_file}\n")
                        f.write(f"测试集文件: {test_file}\n")
                    print(f"数据集信息保存成功: {info_file}")
                except IOError as e:
                    return None, f"写入数据集信息失败: {str(e)}"
            
            return output_files, f"数据集已生成: {len(output_files)}个文件"
        else:
            # 生成文件名（带时间戳避免覆盖）
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"whisper_data_{timestamp}.json")
            
            # 分批写入大型数据集
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    for i in range(0, len(results), 10000):
                        chunk = results[i:i+10000]
                        for item in chunk:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                print(f"数据集保存成功: {output_file}")
                return [output_file], f"数据集已生成: {output_file}"
            except IOError as e:
                return None, f"写入数据集失败: {str(e)}"
    
    except Exception as e:
        traceback.print_exc()
        return None, f"保存过程中出错: {str(e)}"

def data_prep_generate_json(input_file, audio_col, text_col, language_col, 
                 start_col, end_col, segment_text_col, output_dir, include_sentences, 
                 include_duration, train_ratio, enable_split, add_punctuation, auto_calc_duration, 
                 enable_final_test, final_test_ratio):
    """生成JSON文件的主函数（修复文件生成问题）"""
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取输入文件
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith('.xls'):
            df = pd.read_excel(input_file, engine='xlrd')
        else:
            df = pd.read_excel(input_file, engine='openpyxl')
        
        # 初始化结果列表
        results = []
        total_rows = len(df)
        processed_count = 0
        error_count = 0
        
        # 初始化标点模型（如果需要）
        pun_model = None
        if add_punctuation:
            try:
                pun_model = pipeline(
                    task=Tasks.punctuation,
                    model='iic/punc_ct-transformer_cn-en-common-vocab471067-large',
                    model_revision="v2.0.4"
                )
                print("标点恢复模型加载成功")
            except Exception as e:
                print(f"加载标点恢复模型失败: {e}")
                add_punctuation = False
        
        # 处理每一行数据
        for index, row in tqdm(df.iterrows(), total=len(df), desc="处理数据"):
            try:
                # 基础字段处理
                audio_path = row[audio_col] if audio_col != "不选择" else f"audio_{index}.wav"
                sentence = row[text_col] if text_col != "不选择" else ""
                
                # 应用标点恢复 - 修复: 使用正确的输入格式
                if add_punctuation and sentence:
                    try:
                        # 修复: 传递字典格式而不是字符串
                        result = pun_model({'text': sentence})
                        if isinstance(result, dict) and 'text' in result:
                            sentence = result['text']
                        else:
                            print(f"标点恢复返回意外格式: {type(result)}")
                    except Exception as e:
                        print(f"标点恢复失败: {e}")
                
                # 语言处理
                language = row[language_col] if language_col != "不选择" else "Chinese"
                
                # 构建基础JSON结构
                result = {
                    "audio": {
                        "path": str(audio_path)
                    },
                    "sentence": str(sentence),
                    "language": str(language)
                }
                
                # 自动计算音频时长（使用高效方法）
                duration = 0.6
                if auto_calc_duration:
                    try:
                        # 使用soundfile的SoundFile上下文管理器高效获取时长
                        with soundfile.SoundFile(audio_path) as f:
                            duration = round(len(f) / f.samplerate, 2)
                    except Exception as e:
                        print(f"计算音频时长失败: {e}")
                
                # 分段信息处理
                has_segment_data = (start_col != "不选择" and end_col != "不选择" and segment_text_col != "不选择")
                
                if include_sentences:
                    sentences_list = []
                    
                    if has_segment_data:
                        # 处理分段数据
                        starts = data_prep_parse_segment_data(row[start_col])
                        ends = data_prep_parse_segment_data(row[end_col])
                        texts = data_prep_parse_segment_data(row[segment_text_col], is_text=True)
                        
                        # 应用标点恢复到分段文本 - 修复: 使用正确的输入格式
                        if add_punctuation:
                            processed_texts = []
                            for t in texts:
                                try:
                                    # 修复: 传递字典格式而不是字符串
                                    result = pun_model({'text': t})
                                    if isinstance(result, dict) and 'text' in result:
                                        processed_texts.append(result['text'])
                                    else:
                                        processed_texts.append(t)
                                        print(f"标点恢复返回意外格式: {type(result)}")
                                except Exception as e:
                                    print(f"分段标点恢复失败: {e}")
                                    processed_texts.append(t)
                            texts = processed_texts
                        
                        # 创建分段列表
                        for s, e, t in zip(starts, ends, texts):
                            sentences_list.append({
                                "start": float(s),
                                "end": float(e),
                                "text": t
                            })
                        
                        # 如果同时需要duration字段，使用最后一个结束时间
                        if include_duration and ends:
                            duration = ends[-1]
                    else:
                        # 没有分段数据但用户要求包含sentences字段
                        sentences_list.append({
                            "start": 0.0,
                            "end": duration if auto_calc_duration and duration > 0 else 0.0,
                            "text": sentence
                        })
                    
                    result["sentences"] = sentences_list
                
                # 处理duration字段
                if include_duration:
                    if has_segment_data and not include_sentences:
                        ends = data_prep_parse_segment_data(row[end_col])
                        duration = ends[-1] if ends else 0.0
                    result["duration"] = float(duration) if duration > 0 else 0.0
                
                results.append(result)
                processed_count += 1
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                traceback.print_exc()
                error_count += 1
        
        # 调用增强版保存函数
        saved_files, save_message = save_dataset(
            results, 
            output_dir, 
            enable_split, 
            train_ratio,
            enable_final_test,
            final_test_ratio
        )
        
        if saved_files:
            # 添加文件大小信息
            file_info = []
            for file in saved_files:
                try:
                    size_mb = os.path.getsize(file) / (1024 * 1024)
                    file_info.append(f"{os.path.basename(file)}: {size_mb:.2f}MB")
                except:
                    file_info.append(f"{os.path.basename(file)}: 大小未知")
            
            summary = [
                f"✅ 数据集生成成功！",
                f"总行数: {total_rows}",
                f"成功处理: {processed_count}",
                f"失败行数: {error_count}",
                f"保存文件:"
            ] + file_info
            
            return "\n".join(summary)
        else:
            return save_message
    except Exception as e:
        print(f"Error generating JSON: {e}")
        traceback.print_exc()
        return f"生成失败: {str(e)}"

def data_prep_parse_segment_data(data, is_text=False):
    """解析分段数据，支持多种格式"""
    if pd.isna(data):
        return []
    
    if isinstance(data, list):
        return data
    elif isinstance(data, str):
        # 尝试多种分隔符
        if ";" in data:
            return data.split(";")
        elif "|" in data:
            return data.split("|")
        elif "," in data:
            return data.split(",")
        elif "\n" in data:
            return data.split("\n")
        else:
            return [data]
    else:
        return [str(data)]

def data_prep_update_preview(input_file):
    """更新数据预览"""
    if not input_file:
        return pd.DataFrame(), ["不选择"]
    
    try:
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith('.xls'):
            df = pd.read_excel(input_file, engine='xlrd')
        else:
            df = pd.read_excel(input_file, engine='openpyxl')
        
        preview = df.head(5)
        columns = df.columns.tolist()
        options = ["不选择"] + columns
        return preview, options
    except Exception as e:
        print(f"Error updating preview: {e}")
        traceback.print_exc()
        return pd.DataFrame(), ["不选择"]

# ============================== 步骤2: 模型训练功能 ==============================
def load_model(model_name_or_path, download_root=None, is_local_model=False, model_type="whisper", device="auto", compute_type="float16"):
    """加载模型，支持预定义模型名称或本地路径，支持Faster Whisper"""
    # 如果未指定下载目录，使用当前工作目录
    if download_root is None or download_root.strip() == "":
        download_root = os.getcwd()  # 默认下载到当前目录
    
    # 检查模型是否已加载
    cache_key = f"{model_name_or_path}_{download_root}_{is_local_model}_{model_type}_{device}_{compute_type}"
    if cache_key in model_cache:
        print(f"使用缓存的模型: {cache_key}")
        return model_cache[cache_key]
    
    # 处理Faster Whisper模型
    if model_type == "faster_whisper" and FASTER_WHISPER_AVAILABLE:
        try:
            # 设置设备
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            print(f"使用Faster Whisper加载模型: {model_name_or_path}")
            model = FasterWhisperModel(
                model_name_or_path,
                device=device,
                compute_type=compute_type,
                download_root=download_root if not is_local_model else None
            )
            
            model_cache[cache_key] = model
            return model
            
        except Exception as e:
            print(f"Faster Whisper加载失败: {e}")
            raise gr.Error(f"Faster Whisper模型加载失败: {str(e)}")
    
    elif model_type == "faster_whisper" and not FASTER_WHISPER_AVAILABLE:
        raise gr.Error("Faster Whisper未安装，请运行: pip install faster-whisper")
    
    try:
        if is_local_model:
            # 本地模型路径处理
            model_path = model_name_or_path.strip()
            
            # 检查路径是否存在
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"本地模型路径不存在: {model_path}")
            
            print(f"加载本地模型: {model_path}")
            
            # 尝试不同的加载方式
            try:
                # 方法1: 尝试使用transformers库加载（支持Hugging Face格式）
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
                print("尝试使用transformers库加载模型...")
                
                # 设置设备和数据类型
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                
                # 加载处理器和模型
                processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_path, 
                    torch_dtype=torch_dtype, 
                    low_cpu_mem_usage=True, 
                    use_safetensors=True,
                    local_files_only=True
                )
                model.generation_config.forced_decoder_ids = None
                model.to(device)
                
                # 创建推理管道
                from transformers import pipeline
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    chunk_length_s=30,
                    batch_size=16,
                    torch_dtype=torch_dtype,
                    device=device
                )
                
                print("使用transformers库加载成功")
                # 返回包装后的模型对象，兼容原有接口
                class TransformersWhisperModel:
                    def __init__(self, pipe, processor):
                        self.pipe = pipe
                        self.processor = processor
                        self.device = device
                    
                    def transcribe(self, audio, language=None, task="transcribe", **kwargs):
                        """兼容whisper.load_model的transcribe接口"""
                        generate_kwargs = {"task": task}
                        if language:
                            generate_kwargs["language"] = language
                        
                        result = self.pipe(audio, return_timestamps=True, generate_kwargs=generate_kwargs)
                        return result
                
                wrapped_model = TransformersWhisperModel(pipe, processor)
                model_cache[cache_key] = wrapped_model
                return wrapped_model
                
            except Exception as e1:
                print(f"transformers库加载失败: {e1}")
                
                # 方法2: 尝试直接使用whisper.load_model加载
                try:
                    print("尝试使用whisper.load_model加载...")
                    
                    # 如果是目录，查找模型文件
                    if os.path.isdir(model_path):
                        # 查找常见的模型文件
                        possible_files = [
                            os.path.join(model_path, "pytorch_model.bin"),
                            os.path.join(model_path, "model.pt"),
                            os.path.join(model_path, "whisper.pt"),
                            os.path.join(model_path, "model.pth"),
                            os.path.join(model_path, "model.safetensors")
                        ]
                        
                        model_file = None
                        for file_path in possible_files:
                            if os.path.exists(file_path):
                                model_file = file_path
                                break
                        
                        if model_file is None:
                            # 如果没找到标准文件，列出目录中的所有.pt和.bin文件
                            pt_files = [f for f in os.listdir(model_path) if f.endswith(('.pt', '.bin', '.pth'))]
                            if pt_files:
                                model_file = os.path.join(model_path, pt_files[0])
                                print(f"找到模型文件: {model_file}")
                            else:
                                # 列出目录中的所有文件，帮助用户调试
                                all_files = os.listdir(model_path)
                                raise FileNotFoundError(
                                    f"在目录 {model_path} 中未找到模型文件 (.pt, .bin, .pth, .safetensors)。\n"
                                    f"目录中的文件: {', '.join(all_files[:10])}{'...' if len(all_files) > 10 else ''}\n"
                                    f"请确保目录中包含以下格式之一的模型文件：pytorch_model.bin, model.pt, whisper.pt, model.pth, model.safetensors"
                                )
                        
                        model_path = model_file

                    if model_path.endswith('.safetensors'):
                        print("检测到.safetensors格式，使用transformers加载")
                        from transformers import AutoModelForSpeechSeq2Seq
                        model = AutoModelForSpeechSeq2Seq.from_pretrained(
                            os.path.dirname(model_path),
                            local_files_only=True
                        )
                    else:
                        model = whisper.load_model(model_path)
                    
                    print("模型加载成功")
                    
                except Exception as e2:
                    print(f"模型加载失败: {e2}")


                    model = whisper.load_model(model_path)
                    print("使用whisper.load_model加载成功")
                    
                except Exception as e2:
                    print(f"whisper.load_model加载失败: {e2}")
                    
                    # 方法3: 使用torch直接加载模型文件
                    if model_path.endswith(('.pt', '.bin', '.pth')):
                        try:
                            print(f"尝试使用torch.load加载: {model_path}")
                            model_state = torch.load(model_path, map_location='cpu')
                            
                            # 检查模型状态的结构
                            if isinstance(model_state, dict):
                                if 'model_state_dict' in model_state:
                                    # 这是一个训练checkpoint文件
                                    print("检测到训练checkpoint格式")
                                    # 需要先确定模型大小
                                    model_size = 'base'  # 默认使用base
                                    if 'config' in model_state and 'n_mels' in model_state['config']:
                                        # 根据配置推断模型大小
                                        n_mels = model_state['config']['n_mels']
                                        if n_mels == 80:
                                            model_size = 'base'
                                    
                                    model = whisper.load_model(model_size)
                                    model.load_state_dict(model_state['model_state_dict'])
                                elif 'dims' in model_state or any(k.startswith('encoder.') or k.startswith('decoder.') for k in model_state.keys()):
                                    # 这可能是直接的模型状态字典
                                    print("检测到模型状态字典格式")
                                    # 推断模型大小
                                    model_size = 'base'  # 默认
                                    model = whisper.load_model(model_size)
                                    model.load_state_dict(model_state)
                                else:
                                    raise ValueError(f"无法识别的模型格式: {list(model_state.keys())[:5]}")
                            else:
                                raise ValueError(f"模型文件格式不正确，期望字典类型，得到: {type(model_state)}")
                        
                        except Exception as e3:
                            print(f"torch.load也失败: {e3}")
                            raise ValueError(
                                f"无法加载本地模型 {model_path}。\n"
                                f"尝试的方法：\n"
                                f"1. transformers库: {str(e1)}\n"
                                f"2. whisper.load_model: {str(e2)}\n"
                                f"3. torch.load: {str(e3)}\n\n"
                                f"请确保模型文件是有效的Whisper模型格式。\n"
                                f"支持的格式：\n"
                                f"- Hugging Face格式模型目录（推荐）\n"
                                f"- 标准Whisper模型文件 (.pt)\n"
                                f"- 训练checkpoint文件 (.pt, .bin)\n"
                                f"- PyTorch模型状态字典 (.pth)"
                            )
                    else:
                        raise e2
        else:
            # 在线模型，需要下载
            print(f"加载在线模型: {model_name_or_path}，下载路径: {download_root}")
            model = whisper.load_model(model_name_or_path, download_root=download_root)
        
        model_cache[cache_key] = model
        return model
    except Exception as e:
        raise gr.Error(f"无法加载模型: {str(e)}")

def build_command(args, gpus):
    """构建训练命令"""
    list_command = []
    
    # 分布式训练
    if gpus > 1:
        # 多卡训练：使用torchrun，不需要python_exec参数
        list_command.extend([
            "torchrun", 
            f"--nproc_per_node={gpus}",
            "--master_port=29500"
        ])
    
    # 根据训练模式选择脚本
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if args['training_mode'] == "LoRA微调":
        finetune_script = os.path.join(current_dir, "finetune.py")
    else:  # 全参数微调
        finetune_script = os.path.join(current_dir, "finetune_all.py")
    
    # 确保脚本存在
    if not os.path.exists(finetune_script):
        raise FileNotFoundError(f"找不到训练脚本: {finetune_script}")
    
    # 添加脚本路径
    if gpus > 1:
        # 多卡训练：直接添加脚本路径
        list_command.append(finetune_script)
    else:
        # 单卡训练：使用当前Python解释器
        import sys
        python_exec = sys.executable
        list_command.extend([python_exec, finetune_script])
    
    # 数据参数
    list_command.extend([
        f"--train_data={args['train_data']}",
        f"--test_data={args['test_data']}",
        f"--min_audio_len={args['min_audio_len']}",
        f"--max_audio_len={args['max_audio_len']}",
    ])
    
    if args['augment_config_path']:
        list_command.append(f"--augment_config_path={args['augment_config_path']}")
    
    # 模型参数
    list_command.extend([
        f"--base_model={args['base_model']}",
        f"--language={args['language']}",
        f"--task={args['task']}",
        f"--timestamps={'True' if args['timestamps'] else 'False'}",
        f"--fp16={'True' if args['fp16'] else 'False'}",
        f"--use_8bit={'True' if args['use_8bit'] else 'False'}",
    ])
    
    # 根据训练模式添加特定参数
    if args['training_mode'] == "LoRA微调":
        list_command.append(f"--use_adalora={'True' if args['use_adalora'] else 'False'}")
    else:  # 全参数微调
        list_command.append(f"--freeze_encoder={'True' if args['freeze_encoder'] else 'False'}")
    
    # 训练参数
    list_command.extend([
        f"--output_dir={args['output_dir']}",
        f"--num_train_epochs={args['num_train_epochs']}",
        f"--learning_rate={args['learning_rate']}",
        f"--warmup_steps={args['warmup_steps']}",
        f"--per_device_train_batch_size={args['per_device_train_batch_size']}",
        f"--per_device_eval_batch_size={args['per_device_eval_batch_size']}",
        f"--gradient_accumulation_steps={args['gradient_accumulation_steps']}",
        f"--logging_steps={args['logging_steps']}",
        f"--eval_steps={args['eval_steps']}",
        f"--save_steps={args['save_steps']}",
        f"--num_workers={args['num_workers']}",
        f"--save_total_limit={args['save_total_limit']}",
    ])
    
    # 高级参数
    if args['resume_from_checkpoint']:
        list_command.append(f"--resume_from_checkpoint={args['resume_from_checkpoint']}")
    
    list_command.extend([
        f"--local_files_only={'True' if args['local_files_only'] else 'False'}",
        f"--use_compile={'True' if args['use_compile'] else 'False'}",
        f"--push_to_hub={'True' if args['push_to_hub'] else 'False'}",
    ])
    
    if args['hub_model_id']:
        list_command.append(f"--hub_model_id={args['hub_model_id']}")
    
    # 创建字符串命令用于显示
    str_command = " ".join(list_command)
    return str_command, list_command

def run_training(list_command, output_file, log_callback=None):
    """运行训练命令并捕获输出"""
    global TRAINING_PROCESS, TRAINING_ACTIVE
    
    try:
        logger.info(f"执行命令: {' '.join(list_command)}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 直接启动训练进程并实时捕获输出
        TRAINING_PROCESS = subprocess.Popen(
            list_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        
        logger.info(f"训练进程已启动，PID: {TRAINING_PROCESS.pid}")
        
        # 打开日志文件用于写入
        with open(output_file, "w", encoding="utf-8") as log_file:
            # 实时读取进程输出
            while TRAINING_ACTIVE and TRAINING_PROCESS.poll() is None:
                try:
                    line = TRAINING_PROCESS.stdout.readline()
                    if line:
                        line = line.rstrip('\n\r')
                        # 写入日志文件
                        log_file.write(line + '\n')
                        log_file.flush()
                        
                        # 回调函数处理
                        if log_callback:
                            log_callback(line)
                        
                        # 返回给界面显示
                        yield line + '\n'
                    else:
                        time.sleep(0.1)
                except Exception as e:
                    logger.error(f"读取输出时出错: {e}")
                    break
            
            # 读取剩余输出
            remaining_output = TRAINING_PROCESS.stdout.read()
            if remaining_output:
                log_file.write(remaining_output)
                log_file.flush()
                yield remaining_output
        
        # 等待进程结束
        return_code = TRAINING_PROCESS.wait()
        
        # 检查退出状态
        if return_code == 0:
            yield "\n\n✅ 训练成功完成!"
        else:
            yield f"\n\n❌ 训练失败，退出码: {return_code}"
    
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        yield f"\n\n❌ 发生错误: {str(e)}"
    finally:
        TRAINING_PROCESS = None
        TRAINING_ACTIVE = False

def save_command_script(command_str, output_dir, training_mode="LoRA微调"):
    """将训练命令保存为shell脚本文件"""
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 根据训练模式生成不同的文件名
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if training_mode == "LoRA微调":
            script_filename = f"train_lora_{timestamp}.sh"
            script_type = "LoRA微调"
        else:
            script_filename = f"train_full_{timestamp}.sh"
            script_type = "全参数微调"
        
        script_path = os.path.join(output_dir, script_filename)
        
        # 写入文件
        with open(script_path, "w", encoding="utf-8") as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Whisper {script_type}训练命令脚本\n")
            f.write(f"# 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 训练模式: {training_mode}\n\n")
            
            # 添加使用说明
            f.write("# 使用说明:\n")
            if training_mode == "LoRA微调":
                f.write("# 此脚本用于LoRA微调训练，参数效率高，显存占用少\n")
                f.write("# 训练完成后需要使用模型合并功能将LoRA适配器合并到基础模型\n")
            else:
                f.write("# 此脚本用于全参数微调训练，效果更好但显存占用大\n")
                f.write("# 训练完成后直接得到完整的微调模型，无需额外合并步骤\n")
            f.write("# 执行前请确保已安装所需依赖和配置好环境\n\n")
            
            f.write(command_str + "\n")
        
        # 添加执行权限（Windows下可能不支持，但不会报错）
        try:
            os.chmod(script_path, 0o755)
        except:
            pass  # Windows下忽略权限设置错误
        
        return script_path
    except Exception as e:
        print(f"保存训练脚本失败: {e}")
        return None

def start_training(
    train_data,
    test_data,
    augment_config_path,
    min_audio_len,
    max_audio_len,
    base_model,
    language,
    task,
    training_mode,
    timestamps,
    use_adalora,
    freeze_encoder,
    fp16,
    use_8bit,
    output_dir,
    num_train_epochs,
    learning_rate,
    warmup_steps,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    gradient_accumulation_steps,
    logging_steps,
    eval_steps,
    save_steps,
    num_workers,
    save_total_limit,
    resume_from_checkpoint,
    local_files_only,
    use_compile,
    push_to_hub,
    hub_model_id,
    gpus,
    save_command_script_flag  # 新增参数
):
    global TRAINING_THREAD, OUTPUT_FILE, TRAINING_ACTIVE
    
    if TRAINING_ACTIVE:
        yield "⚠️ 当前已有训练正在进行，请等待完成后再启动新训练。", "", None
        return
    
    # 准备参数
    args = {
        "train_data": train_data,
        "test_data": test_data,
        "augment_config_path": augment_config_path,
        "min_audio_len": min_audio_len,
        "max_audio_len": max_audio_len,
        "base_model": base_model,
        "language": language,
        "task": task,
        "training_mode": training_mode,
        "timestamps": timestamps,
        "use_adalora": use_adalora,
        "freeze_encoder": freeze_encoder,
        "fp16": fp16,
        "use_8bit": use_8bit,
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "logging_steps": logging_steps,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "num_workers": num_workers,
        "save_total_limit": save_total_limit,
        "resume_from_checkpoint": resume_from_checkpoint,
        "local_files_only": local_files_only,
        "use_compile": use_compile,
        "push_to_hub": push_to_hub,
        "hub_model_id": hub_model_id,
    }
    
    # 构建命令
    str_command, list_command = build_command(args, gpus)
    
    # 创建输出文件
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    OUTPUT_FILE = f"training_logs/train_{timestamp}.log"
    
    # 保存训练命令为脚本（如果用户选择）
    script_path = None
    if save_command_script_flag:
        script_path = save_command_script(str_command, output_dir, training_mode)
        script_info = f"\n\n📜 训练命令已保存为脚本: {script_path}" if script_path else "\n\n⚠️ 保存训练脚本失败"
    else:
        script_info = ""
    
    TRAINING_ACTIVE = True
    
    # 先返回命令预览
    initial_output = f"训练命令预览:\n{str_command}{script_info}\n\n开始训练...\n日志文件: {OUTPUT_FILE}\n"
    yield initial_output, str_command, script_path
    
    # 运行训练并实时返回输出
    accumulated_output = initial_output
    
    for line in run_training(list_command, OUTPUT_FILE):
        accumulated_output += line
        yield accumulated_output, str_command, script_path

def stop_training():
    global TRAINING_PROCESS, TRAINING_ACTIVE
    
    if TRAINING_PROCESS and TRAINING_ACTIVE:
        # 终止整个进程组
        try:
            if platform.system() == "Windows":
                subprocess.Popen(["taskkill", "/F", "/T", "/PID", str(TRAINING_PROCESS.pid)])
            else:
                os.killpg(os.getpgid(TRAINING_PROCESS.pid), signal.SIGTERM)
            TRAINING_ACTIVE = False
            return "🛑 训练已停止", ""
        except Exception as e:
            return f"停止训练失败: {str(e)}", ""
    return "⚠️ 没有正在进行的训练", ""

def get_log_content():
    if OUTPUT_FILE and os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            return f.read()
    return "暂无日志"

# ============================== 步骤3: 模型合并功能 ==============================
def merge_models(lora_model, output_dir, local_files_only):
    """合并LoRA模型到基础模型"""
    try:
        start_time = time.time()
        
        # 检查模型文件是否存在
        if not os.path.exists(lora_model):
            return f"错误：模型文件 {lora_model} 不存在！"
        
        # 获取Lora配置参数
        peft_config = PeftConfig.from_pretrained(lora_model)
        
        # 获取Whisper的基本模型
        base_model = WhisperForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path, 
            device_map={"": "cpu"},
            local_files_only=local_files_only
        )
        
        # 与Lora模型合并
        model = PeftModel.from_pretrained(base_model, lora_model, local_files_only=local_files_only)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            peft_config.base_model_name_or_path,
            local_files_only=local_files_only
        )
        tokenizer = WhisperTokenizerFast.from_pretrained(
            peft_config.base_model_name_or_path,
            local_files_only=local_files_only
        )
        processor = WhisperProcessor.from_pretrained(
            peft_config.base_model_name_or_path,
            local_files_only=local_files_only
        )
        
        # 合并参数
        model = model.merge_and_unload()
        model.train(False)
        
        # 保存的文件夹路径
        if peft_config.base_model_name_or_path.endswith("/"):
            peft_config.base_model_name_or_path = peft_config.base_model_name_or_path[:-1]
        save_directory = os.path.join(output_dir, f'{os.path.basename(peft_config.base_model_name_or_path)}-finetune')
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存模型到指定目录中
        model.save_pretrained(save_directory, max_shard_size='4GB')
        feature_extractor.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        processor.save_pretrained(save_directory)
        
        elapsed = time.time() - start_time
        return f"模型合并成功！保存在: {save_directory}\n耗时: {elapsed:.2f}秒"
    except Exception as e:
        traceback.print_exc()
        return f"模型合并失败: {str(e)}"

# ============================== 步骤4: 模型使用功能 ==============================
def transcribe_audio(model_name, file_path, task, language, custom_model_path, download_root, model_type, 
                     enable_vad=False, vad_method="silero", vad_aggressiveness=3, 
                     enable_multi_model=False, whisper_model_type="whisper", device="auto", compute_type="float16"):
    """执行语音识别或翻译，支持VAD过滤和多模型对比"""
    
    # 应用VAD过滤（如果启用）
    processed_audio_path = file_path
    if enable_vad:
        try:
            filtered_path = apply_vad_filter(file_path, vad_method, vad_aggressiveness)
            if filtered_path and filtered_path != file_path:
                processed_audio_path = filtered_path
                print(f"VAD过滤完成: {vad_method}")
            else:
                print("VAD过滤未生效，使用原始音频")
        except Exception as e:
            print(f"VAD过滤失败: {e}，使用原始音频")
    
    # 多模型对比模式
    if enable_multi_model:
        try:
            # 获取已加载的模型列表
            model_list = multi_model_manager.get_model_list()
            if not model_list:
                return "未加载任何模型进行对比", "", ""
            
            # 设置语言参数
            lang = None if language == "自动检测" else language
            
            # 使用所有模型进行转录
            results = multi_model_manager.transcribe_with_all_models(
                processed_audio_path, task, lang
            )
            
            # 格式化对比结果
            formatted_results = []
            detected_languages = []
            
            for model_name, result in results.items():
                if result["status"] == "success":
                    formatted_results.append(f"**{model_name}:**\n{result['text']}\n")
                    detected_languages.append(result["language"])
                else:
                    formatted_results.append(f"**{model_name}:** 错误 - {result.get('error', '未知错误')}\n")
            
            combined_text = "\n".join(formatted_results)
            
            # 返回最常见的检测语言
            if detected_languages:
                most_common_lang = max(set(detected_languages), key=detected_languages.count)
                detected_lang_name = LANGUAGES.get(most_common_lang, most_common_lang)
            else:
                most_common_lang = "unknown"
                detected_lang_name = "未知"
            
            return combined_text, detected_lang_name, most_common_lang
            
        except Exception as e:
            return f"多模型对比失败: {str(e)}", "", ""
    
    # 单模型模式
    # 根据模型类型选择加载方式
    if model_type == "本地模型":
        if not custom_model_path or not os.path.exists(custom_model_path):
            return "请提供有效的本地模型路径", "", ""
        model_path = custom_model_path
        is_local = True
    else:  # 在线模型
        model_path = model_name
        is_local = False
    
    try:
        model = load_model(model_path, download_root, is_local, whisper_model_type, device, compute_type)
    except Exception as e:
        return str(e), "", ""
    
    # 设置语言参数
    lang = None if language == "自动检测" else language
    
    # 根据模型类型处理
    try:
        if whisper_model_type == "faster_whisper":
            # Faster Whisper 模型
            segments, info = model.transcribe(processed_audio_path, task=task, language=lang)
            
            # 格式化结果
            formatted_segments = []
            for segment in segments:
                formatted_segments.append(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
            
            text = "\n".join(formatted_segments)
            detected_lang = info.language if hasattr(info, 'language') else "unknown"
            detected_lang_name = LANGUAGES.get(detected_lang, detected_lang)
            
            return text, detected_lang_name, detected_lang
            
        elif hasattr(model, 'pipe'):  # TransformersWhisperModel
            # 使用transformers pipeline
            generate_kwargs = {"task": task}
            if lang:
                generate_kwargs["language"] = lang
            
            result = model.pipe(processed_audio_path, return_timestamps=True, generate_kwargs=generate_kwargs)
            
            # 格式化结果
            if "chunks" in result:
                formatted_chunks = []
                for chunk in result["chunks"]:
                    # 安全处理时间戳
                    start = chunk['timestamp'][0] if chunk['timestamp'] and chunk['timestamp'][0] is not None else 0.0
                    end = chunk['timestamp'][1] if chunk['timestamp'] and chunk['timestamp'][1] is not None else 0.0
                    
                    # 确保时间戳是数字类型
                    try:
                        start = float(start)
                        end = float(end)
                    except (TypeError, ValueError):
                        start, end = 0.0, 0.0
                    
                    formatted_chunks.append(f"[{start:.2f}s - {end:.2f}s] {chunk['text']}")
                
                text = "\n".join(formatted_chunks)
            else:
                text = result.get("text", "")
            
            detected_lang = lang if lang else "auto"
            detected_lang_name = LANGUAGES.get(detected_lang, detected_lang) if detected_lang != "auto" else "自动检测"
            
            return text, detected_lang_name, detected_lang
            
        else:
            # 使用标准whisper模型
            result = model.transcribe(
                processed_audio_path,
                task=task,
                language=lang
            )
            
            # 获取检测到的语言
            detected_lang = result.get("language", "未知")
            detected_lang_name = LANGUAGES.get(detected_lang, detected_lang)
            
            return result["text"], detected_lang_name, detected_lang
            
    except Exception as e:
        return f"模型推理失败: {str(e)}", "", ""
    
    finally:
        # 清理临时VAD过滤文件
        if enable_vad and processed_audio_path != file_path:
            try:
                os.unlink(processed_audio_path)
            except:
                pass

def process_file(file, model_name, task, language, custom_model_path, download_root, model_type):
    """处理上传的文件"""
    audio_path = None
    
    try:
        # 如果是视频文件，先提取音频
        if isinstance(file, str) and file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            try:
                import ffmpeg
                # 在Gradio临时目录中创建音频文件
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=temp_dir) as tmpfile:
                    audio_path = tmpfile.name
                
                # 使用ffmpeg提取音频
                (
                    ffmpeg
                    .input(file)
                    .output(audio_path, ac=1, ar=16000)
                    .overwrite_output()
                    .run(quiet=True)
                )
                file_path = audio_path
            except ImportError:
                return "处理视频需要ffmpeg-python库，请安装: pip install ffmpeg-python", "", "", ""
            except Exception as e:
                return f"视频处理失败: {str(e)}", "", "", ""
        else:
            file_path = file
        
        # 执行识别/翻译
        text, detected_lang, lang_code = transcribe_audio(model_name, file_path, task, language, custom_model_path, download_root, model_type)
        
        # 返回文本、检测语言、语言代码、音频文件路径（用于保存）
        return text, detected_lang, lang_code, audio_path or file_path
    
    except Exception as e:
        return f"处理过程中出错: {str(e)}", "", "", ""
    finally:
        # 清理临时音频文件（如果是视频转换生成的）
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass

def update_language_visibility(task):
    """根据任务类型更新语言选择的可见性"""
    return gr.Dropdown(visible=task != "translate")

def update_model_interface(model_type):
    """根据模型类型更新界面显示"""
    if model_type == "在线模型":
        return (
            gr.Dropdown(label="选择在线模型", visible=True),  # model_choice
            gr.Textbox(label="本地模型路径", visible=False),  # custom_model_path
            gr.Textbox(label="模型下载路径 (仅在线模型)", visible=True)  # download_root_input
        )
    else:  # 本地模型
        return (
            gr.Dropdown(label="选择在线模型", visible=False),  # model_choice
            gr.Textbox(label="本地模型路径", visible=True),  # custom_model_path
            gr.Textbox(label="模型下载路径 (仅在线模型)", visible=False)  # download_root_input
        )

def update_training_mode_interface(training_mode):
    """根据训练模式更新界面显示"""
    if training_mode == "LoRA微调":
        return (
            gr.Checkbox(label="使用AdaLora", visible=True),  # use_adalora
            gr.Checkbox(label="冻结编码器", visible=False)   # freeze_encoder
        )
    else:  # 全参数微调
        return (
            gr.Checkbox(label="使用AdaLora", visible=False), # use_adalora
            gr.Checkbox(label="冻结编码器", visible=True)    # freeze_encoder
        )

def save_dataset_entry(text, audio_path, output_dir):
    """保存数据集条目到JSON文件"""
    if not audio_path or not isinstance(audio_path, str):
        return "音频文件路径无效，无法保存"
    
    if not os.path.exists(audio_path):
        return "音频文件不存在，无法保存"
    
    if not text.strip():
        return "文本内容为空，无法保存"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建数据集文件路径
    dataset_path = os.path.join(output_dir, "dataset.json")
    
    # 复制音频文件到输出目录
    audio_filename = os.path.basename(audio_path)
    target_audio_path = os.path.join(output_dir, audio_filename)
    
    try:
        # 复制音频文件
        import shutil
        shutil.copy(audio_path, target_audio_path)
        
        # 创建或更新数据集JSON
        entry = {
            "audio_filepath": audio_filename,
            "text": text.strip()
        }
        
        # 如果文件存在，直接追加；如果不存在，创建新文件
        with open(dataset_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # 计算条目总数
        count = 0
        if os.path.exists(dataset_path):
            with open(dataset_path, "r", encoding="utf-8") as f:
                count = sum(1 for _ in f)
            
        return f"✅ 已保存到数据集! 条目总数: {count}"
    except Exception as e:
        return f"保存失败: {str(e)}"

# ============================== Gradio界面 ==============================
# 可用的模型列表
MODEL_OPTIONS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"]

# 语言选项
language_names = sorted(LANGUAGES.values())
language_options = ["自动检测"] + language_names

# 创建Gradio界面
with gr.Blocks(theme=gr.themes.Ocean(), title="Whisper 训练工具套件") as demo:
    gr.Markdown("# 🎤 Whisper 语音识别模型训练平台")
    gr.Markdown("Whisper 模型训练工作台，数据集准备和训练")
    
    # ======================= 步骤1: 数据准备 =======================
    with gr.Tab("步骤1: 数据准备", id="data_preparation"):
        gr.Markdown("""### 数据准备
        **上传CSV/Excel文件，选择对应列生成Whisper训练所需的JSON格式**
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="上传数据文件 (CSV/Excel)", 
                    type="filepath",
                    file_types=[".csv", ".xls", ".xlsx"]
                )
                gr.Markdown("支持CSV和Excel格式")
                
                gr.Markdown("### 字段映射")
                audio_col = gr.Dropdown(
                    label="音频路径列", 
                    choices=["不选择"],
                    value="不选择",
                    interactive=True
                )
                text_col = gr.Dropdown(
                    label="文本列", 
                    choices=["不选择"],
                    value="不选择",
                    interactive=True
                )
                
                language_col = gr.Dropdown(
                    label="语言列", 
                    choices=["不选择"],
                    value="不选择",
                    interactive=True
                )
                
                gr.Markdown("### 分段信息 (可选)")
                with gr.Row():
                    start_col = gr.Dropdown(
                        label="开始时间列", 
                        choices=["不选择"],
                    value="不选择",
                    interactive=True
                )
                end_col = gr.Dropdown(
                    label="结束时间列", 
                        choices=["不选择"],
                    value="不选择",
                    interactive=True
                )
                segment_text_col = gr.Dropdown(
                    label="分段文本列", 
                    choices=["不选择"],
                    value="不选择",
                    interactive=True
                )
                
                output_dir = gr.Textbox(
                    label="输出目录", 
                    value="./whisper_dataset"
                )
                
            with gr.Column(scale=1):
                
                gr.Markdown("### 高级选项")
                with gr.Row():
                    include_sentences = gr.Checkbox(
                        label="包含sentences字段",
                        value=False
                    )
                    include_duration = gr.Checkbox(
                        label="包含duration字段",
                        value=True
                    )
                
                    add_punctuation = gr.Checkbox(
                        label="添加标点恢复",
                        value=False
                    )
                    auto_calc_duration = gr.Checkbox(
                        label="自动计算音频时长",
                        value=True
                    )
                    
                gr.Markdown("### 📊 数据集划分配置")
                gr.Markdown("💡 **注意**: 数据将自动随机打乱后生成JSONL文件，确保训练数据的随机性")
                
                # 数据集划分模式选择
                dataset_split_mode = gr.Radio(
                    label="📋 数据集划分模式",
                    choices=[
                        "不划分 - 生成单个完整数据集",
                        "二分割 - 训练集 + 验证集", 
                        "三分割 - 训练集 + 验证集 + 最终测试集"
                    ],
                    value="不划分 - 生成单个完整数据集",
                    info="选择数据集的划分方式"
                )
                
                # 二分割配置区域
                with gr.Group(visible=False) as two_split_config:
                    gr.Markdown("#### 🔄 二分割配置")
                    gr.Markdown("📈 **说明**: 数据将被划分为训练集和验证集两部分")
                    train_ratio = gr.Slider(
                        label="训练集比例",
                        minimum=0.5,
                        maximum=0.95,
                        value=0.8,
                        step=0.05,
                        info="训练集占总数据的比例，剩余部分为验证集"
                    )
                    with gr.Row():
                        gr.Markdown("**示例**: 1000条数据，训练集80% → 训练集800条，验证集200条")
                
                # 三分割配置区域
                with gr.Group(visible=False) as three_split_config:
                    gr.Markdown("#### 🎯 三分割配置")
                    gr.Markdown("📊 **说明**: 数据将被划分为训练集、验证集和最终测试集三部分")
                    
                    with gr.Row():
                        final_test_ratio = gr.Slider(
                            label="最终测试集比例",
                            minimum=0.05,
                            maximum=0.3,
                            value=0.1,
                            step=0.05,
                            info="从总数据中划分的比例，用于最终模型评估"
                        )
                        train_ratio_three = gr.Slider(
                            label="训练集比例（剩余数据中）",
                            minimum=0.5,
                            maximum=0.95,
                            value=0.8,
                            step=0.05,
                            info="从剩余数据中划分训练集的比例"
                        )
                    
                    gr.Markdown("**划分流程**:")
                    gr.Markdown("1️⃣ 首先从总数据中划分出最终测试集")
                    gr.Markdown("2️⃣ 剩余数据按训练集比例划分为训练集和验证集")
                    gr.Markdown("**示例**: 1000条数据，最终测试集10%，训练集80% → 最终测试集100条，训练集720条，验证集180条")
                
                # 隐藏的兼容性变量（保持原有接口）
                enable_split = gr.State(False)
                enable_final_test = gr.State(False)
                
                # 添加模式切换的交互逻辑
                def update_split_interface(mode):
                    """根据选择的模式更新界面显示"""
                    if mode == "不划分 - 生成单个完整数据集":
                        return (
                            gr.Group(visible=False),  # two_split_config
                            gr.Group(visible=False),  # three_split_config
                            False,  # enable_split
                            False   # enable_final_test
                        )
                    elif mode == "二分割 - 训练集 + 验证集":
                        return (
                            gr.Group(visible=True),   # two_split_config
                            gr.Group(visible=False),  # three_split_config
                            True,   # enable_split
                            False   # enable_final_test
                        )
                    else:  # 三分割模式
                        return (
                            gr.Group(visible=False),  # two_split_config
                            gr.Group(visible=True),   # three_split_config
                            True,   # enable_split
                            True    # enable_final_test
                        )
                
                # 绑定模式切换事件
                dataset_split_mode.change(
                    fn=update_split_interface,
                    inputs=[dataset_split_mode],
                    outputs=[two_split_config, three_split_config, enable_split, enable_final_test]
                )
                
                generate_btn = gr.Button("生成数据集", variant="primary", size="lg")

                gr.Markdown("### 数据预览")
                preview_table = gr.Dataframe(
                    interactive=False,
                    wrap=True
                )
                
                gr.Markdown("### 生成结果")
                output_result = gr.Textbox(
                    label="输出信息",
                    interactive=False,
                    lines=4
                )
        
        # 当文件上传时更新所有组件
        def update_all_components(input_file):
            preview, options = data_prep_update_preview(input_file)
            return [
                preview,
                gr.update(choices=options),
                gr.update(choices=options),
                gr.update(choices=options),
                gr.update(choices=options),
                gr.update(choices=options),
                gr.update(choices=options)
            ]
        
        # 文件上传时更新所有下拉框选项和预览
        file_input.change(
            fn=update_all_components,
            inputs=file_input,
            outputs=[
                preview_table,
                audio_col,
                text_col,
                language_col,
                start_col,
                end_col,
                segment_text_col
            ]
        )
        
        # 包装函数，根据模式选择正确的参数
        def generate_with_mode_selection(
            file_input, audio_col, text_col, language_col, start_col, end_col, 
            segment_text_col, output_dir, include_sentences, include_duration, 
            add_punctuation, auto_calc_duration, dataset_split_mode, 
            train_ratio_two, train_ratio_three, final_test_ratio, 
            enable_split_state, enable_final_test_state
        ):
            """根据选择的模式调用数据生成函数"""
            # 根据模式确定参数
            if dataset_split_mode == "不划分 - 生成单个完整数据集":
                actual_enable_split = False
                actual_enable_final_test = False
                actual_train_ratio = 0.8  # 默认值，不会被使用
            elif dataset_split_mode == "二分割 - 训练集 + 验证集":
                actual_enable_split = True
                actual_enable_final_test = False
                actual_train_ratio = train_ratio_two
            else:  # 三分割模式
                actual_enable_split = True
                actual_enable_final_test = True
                actual_train_ratio = train_ratio_three
            
            return data_prep_generate_json(
                file_input, audio_col, text_col, language_col, start_col, end_col,
                segment_text_col, output_dir, include_sentences, include_duration,
                actual_train_ratio, actual_enable_split, add_punctuation, auto_calc_duration,
                actual_enable_final_test, final_test_ratio
            )
        
        # 生成JSON文件
        generate_btn.click(
            fn=generate_with_mode_selection,
            inputs=[
                file_input,
                audio_col,
                text_col,
                language_col,
                start_col,
                end_col,
                segment_text_col,
                output_dir,
                include_sentences,
                include_duration,
                add_punctuation,
                auto_calc_duration,
                dataset_split_mode,
                train_ratio,
                train_ratio_three,
                final_test_ratio,
                enable_split,
                enable_final_test
            ],
            outputs=output_result
        )
    
    # ======================= 步骤2: 模型训练 =======================
    with gr.Tab("步骤2: 模型训练", id="model_training"):
        gr.Markdown("### 训练参数设置")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 数据设置")
                with gr.Row():
                    train_data = gr.Textbox(label="训练数据集路径", value="./whisper_dataset/train.jsonl")
                    test_data = gr.Textbox(label="测试数据集路径", value="./whisper_dataset/test.jsonl")
                augment_config_path = gr.Textbox(label="增强配置文件路径", placeholder="可选，留空则不使用增强")
                with gr.Row():
                    min_audio_len = gr.Number(label="最小音频长度(秒)", value=0.5)
                    max_audio_len = gr.Number(label="最大音频长度(秒)", value=30.0)
                
                gr.Markdown("#### 模型设置")
                base_model = gr.Textbox(label="基础模型", value="./whisper-large-v3-turbo")
                language = gr.Dropdown(label="语言", choices=["Chinese", "English", "Japanese", "Multilingual"], value="Chinese")
                task = gr.Dropdown(label="任务", choices=["transcribe", "translate"], value="transcribe")
                
                # 训练模式选择
                training_mode = gr.Radio(
                    label="训练模式",
                    choices=["LoRA微调", "全参数微调"],
                    value="LoRA微调",
                    info="LoRA微调：参数效率高，显存占用少；全参数微调：效果更好，显存占用大"
                )
                
                with gr.Row():
                    timestamps = gr.Checkbox(label="使用时间戳", value=False)
                    # LoRA相关参数，仅在LoRA模式下显示
                    use_adalora = gr.Checkbox(label="使用AdaLora", value=True, visible=True)
                
                # 全参数微调专用参数
                freeze_encoder = gr.Checkbox(
                    label="冻结编码器", 
                    value=True, 
                    visible=False,
                    info="仅训练解码器参数，减少显存占用"
                )
                
                with gr.Row():
                    fp16 = gr.Checkbox(label="FP16训练", value=True)
                    use_8bit = gr.Checkbox(label="8-bit量化", value=False)
            
                gr.Markdown("#### 高级设置")
                # 新增保存脚本选项
                save_command_script_flag = gr.Checkbox(
                   label="保存训练命令为脚本文件",
                   value=True,
                   info="将训练参数保存为可执行的Shell脚本")
                resume_from_checkpoint = gr.Textbox(label="恢复训练检查点", placeholder="可选，从检查点恢复训练")
                local_files_only = gr.Checkbox(label="仅使用本地模型", value=False)
                use_compile = gr.Checkbox(label="使用PyTorch编译", value=False)
                push_to_hub = gr.Checkbox(label="推送模型到HuggingFace Hub", value=False)
                hub_model_id = gr.Textbox(label="Hub模型ID", placeholder="HuggingFace Hub模型ID")
            
            with gr.Column(scale=1):
                gr.Markdown("#### 训练设置")
                output_dir = gr.Textbox(label="输出目录", value="./whisper_output/")
                num_train_epochs = gr.Number(label="训练轮数", value=3)
                learning_rate = gr.Number(label="学习率", value=1e-3)
                warmup_steps = gr.Number(label="预热步数", value=50)
                with gr.Row():
                    per_device_train_batch_size = gr.Number(label="训练batch size", value=8)
                    per_device_eval_batch_size = gr.Number(label="评估batch size", value=8)
                gradient_accumulation_steps = gr.Number(label="梯度累积步数", value=1)
                with gr.Row():
                    logging_steps = gr.Number(label="日志打印步数", value=100)
                    eval_steps = gr.Number(label="评估步数", value=1000)
                    save_steps = gr.Number(label="保存步数", value=1000)
                with gr.Row():
                    num_workers = gr.Number(label="数据加载线程数", value=8)
                    save_total_limit = gr.Number(label="保存的检查点数量", value=10)
                gpus = gr.Number(label="GPU数量", value=1, precision=0)
        
        
        with gr.Row():
            start_btn = gr.Button("🚀 开始训练", variant="primary")
            stop_btn = gr.Button("🛑 停止训练")
        
        with gr.Row():
            gr.Markdown("### 训练命令预览")
            command_preview = gr.Textbox(label="将执行的命令", lines=3, interactive=False)
        
        with gr.Row():
            output_log = gr.Textbox(label="训练输出", lines=20, interactive=False, autoscroll=True)
        # 新增训练脚本下载组件
        with gr.Row():
           script_output = gr.File(
              label="训练脚本",
              interactive=False,
              visible=False)
        
        # 设置训练模式切换事件
        training_mode.change(
            fn=update_training_mode_interface,
            inputs=[training_mode],
            outputs=[use_adalora, freeze_encoder]
        )
        
        # 存储所有输入组件
        all_input_components = [
            train_data, test_data, augment_config_path, min_audio_len, max_audio_len,
            base_model, language, task, training_mode, timestamps, use_adalora, freeze_encoder, fp16, use_8bit,
            output_dir, num_train_epochs, learning_rate, warmup_steps,
            per_device_train_batch_size, per_device_eval_batch_size,
            gradient_accumulation_steps, logging_steps, eval_steps, save_steps,
            num_workers, save_total_limit,
            resume_from_checkpoint, local_files_only, use_compile, push_to_hub, hub_model_id,
            gpus, save_command_script_flag
        ]
        
        # 设置开始按钮的点击事件
        start_btn.click(
            fn=start_training,
            inputs=all_input_components,
            outputs=[output_log, command_preview,script_output],
            show_progress=True
        )
        
        # 设置停止按钮的点击事件
        stop_btn.click(
            fn=stop_training,
            outputs=[output_log, command_preview,script_output]
        )
        
        # 添加日志查看按钮
        gr.Markdown("### 日志查看")
        with gr.Row():
            log_view_btn = gr.Button("查看完整日志")
            log_content = gr.Textbox(label="完整日志内容", lines=10, interactive=False)
        
        log_view_btn.click(
            fn=get_log_content,
            outputs=log_content
        )

    # ======================= 步骤3: 模型合并 =======================
    with gr.Tab("步骤3: 模型合并", id="model_merge"):
        gr.Markdown("### 🛠️ LoRA 模型合并")
        gr.Markdown("将LoRA适配器合并到基础模型中，生成可直接使用的完整模型")
        
        with gr.Row():
            with gr.Column():
                lora_model = gr.Textbox(
                    label="LoRA 模型路径", 
                    value="./whisper_output/"
                )
                output_dir = gr.Textbox(
                    label="输出目录", 
                    value="./whisper_merged_models/"
                )
                local_files_only_merge = gr.Checkbox(
                    label="仅使用本地文件", 
                    value=True
                )
                merge_btn = gr.Button("开始合并", variant="primary")
                
            with gr.Column():
                merge_output = gr.Textbox(
                    label="合并结果", 
                    lines=8, 
                    interactive=False
                )
        
        merge_btn.click(
            fn=merge_models,
            inputs=[lora_model, output_dir, local_files_only_merge],
            outputs=merge_output
        )
        
        gr.Markdown("### 使用说明")
        gr.Markdown("""
        1. **LoRA模型路径**：输入微调后的模型保存目录路径
        2. **输出目录**：指定合并后模型的保存位置
        3. **仅使用本地文件**：勾选后将不会尝试从HuggingFace Hub下载模型
        4. 点击"开始合并"按钮执行合并操作
        5. 合并后的模型将保存在指定输出目录中
        """)

    # ======================= 步骤4: 模型使用 =======================
    with gr.Tab("步骤4: 模型使用", id="model_usage"):
        gr.Markdown("### 🎧 语音识别与翻译")
        gr.Markdown("使用Whisper模型进行语音识别或翻译，支持Faster Whisper、VAD过滤、多模型对比等高级功能")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 🔧 模型配置")
                with gr.Group():
                    # 模型引擎选择
                    whisper_model_type = gr.Radio(
                        label="Whisper引擎类型",
                        choices=["whisper", "faster_whisper", "transformers"],
                        value="whisper",
                        info="whisper: 标准OpenAI Whisper | faster_whisper: 优化版本，速度更快 | transformers: HuggingFace版本"
                    )
                    
                    # 设备和计算类型设置（仅Faster Whisper可见）
                    with gr.Group(visible=False) as faster_whisper_settings:
                        gr.Markdown("**Faster Whisper设置**")
                        with gr.Row():
                            device_choice = gr.Dropdown(
                                label="设备选择",
                                choices=["auto", "cuda", "cpu"],
                                value="auto",
                                info="auto: 自动检测 | cuda: GPU加速 | cpu: CPU推理"
                            )
                            compute_type = gr.Dropdown(
                                label="计算精度",
                                choices=["float16", "float32", "int8"],
                                value="float16",
                                info="float16: 半精度，速度快 | float32: 全精度 | int8: 量化，内存占用小"
                            )
                    
                    # 模型类型和路径
                    model_type = gr.Radio(
                        label="模型来源",
                        choices=["在线模型", "本地模型"],
                        value="在线模型"
                    )
                    model_choice = gr.Dropdown(
                        label="选择在线模型", 
                        choices=MODEL_OPTIONS, 
                        value="large-v3",
                        visible=True
                    )
                    custom_model_path = gr.Textbox(
                        label="本地模型路径",
                        placeholder="例如: /path/to/your/model",
                        visible=False
                    )
                    download_root_input = gr.Textbox(
                        label="模型下载路径 (仅在线模型)",
                        placeholder="留空则默认当前工作目录",
                        value="/Whisper-Finetune/whisper-large-v3-turbo",
                        visible=True
                    )
                
                gr.Markdown("#### 🎯 任务设置")
                with gr.Group():
                    task_choice = gr.Radio(
                        label="任务类型",
                        choices=["transcribe", "translate"],
                        value="transcribe"
                    )
                    
                    language_choice = gr.Dropdown(
                        label="语言选择 (仅用于语音识别)",
                        choices=language_options,
                        value="自动检测",
                        visible=True
                    )
                
                gr.Markdown("#### 🔊 音频输入")
                with gr.Group():
                    # 音频输入方式选择
                    audio_input_mode = gr.Radio(
                        label="音频输入方式",
                        choices=["文件上传", "目录批量", "语音录制"],
                        value="文件上传",
                        info="文件上传: 单个文件 | 目录批量: 批量处理文件夹 | 语音录制: 实时录音"
                    )
                    
                    # 单文件上传
                    file_input = gr.File(
                        label="上传音频/视频文件", 
                        file_types=["audio", "video"],
                        visible=True
                    )
                    
                    # 目录批量上传
                    directory_input = gr.Textbox(
                        label="音频文件路径或目录路径",
                        placeholder="例如: /path/to/audio/files 或 /path/to/audio.wav",
                        visible=False,
                        info="支持目录路径或直接音频文件路径 | 格式: .wav, .mp3, .m4a, .flac, .ogg"
                    )
                    
                    # 语音录制
                    audio_recorder = gr.Audio(
                        label="语音录制",
                        sources=["microphone"],
                        type="filepath",
                        visible=False
                    )
                
                gr.Markdown("#### ⚙️ 高级设置")
                with gr.Group():
                    # VAD设置
                    enable_vad = gr.Checkbox(
                        label="启用VAD语音活动检测",
                        value=False,
                        info="过滤静音片段，提高识别质量"
                    )
                    
                    with gr.Row(visible=False) as vad_settings:
                        vad_method = gr.Dropdown(
                            label="VAD方法",
                            choices=["silero", "webrtc"],
                            value="silero",
                            info="silero: 深度学习方法 | webrtc: 传统信号处理"
                        )
                        vad_aggressiveness = gr.Slider(
                            label="VAD敏感度",
                            minimum=0,
                            maximum=3,
                            value=3,
                            step=1,
                            info="0: 最不敏感 | 3: 最敏感"
                        )
                    
                    # CT2模型转换（仅本地模型可见）
                with gr.Group(visible=False) as ct2_conversion_group:
                    gr.Markdown("**CT2模型转换**")
                    with gr.Row():
                        convert_to_ct2 = gr.Button(
                            "转换为CT2格式",
                            variant="secondary",
                            size="sm"
                        )
                        ct2_output_dir = gr.Textbox(
                            label="CT2输出目录",
                            placeholder="转换后的CT2模型保存路径",
                            scale=2
                        )
                
                with gr.Row():
                    submit_btn = gr.Button("🚀 开始转录", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ 清空结果", variant="secondary", scale=1)
            
            with gr.Column(scale=1):
                gr.Markdown("#### 📝 识别结果")
                output_text = gr.Textbox(
                    label="识别/翻译结果", 
                    lines=12, 
                    interactive=True,
                    placeholder="转录结果将在这里显示..."
                )
                
                with gr.Row():
                    detected_lang = gr.Textbox(
                        label="检测到的语言", 
                        interactive=False,
                        scale=1
                    )
                    processing_status = gr.Textbox(
                        label="处理状态",
                        interactive=False,
                        scale=1
                    )
                
                # 模型信息显示
                gr.Markdown("#### ℹ️ 模型信息")
                model_info_display = gr.Textbox(
                    label="当前模型信息",
                    lines=2,
                    interactive=False,
                    placeholder="选择模型后将显示相关信息"
                )
                
                # 音频预览和文件信息
                audio_preview = gr.Audio(
                    label="音频预览", 
                    interactive=False, 
                    visible=False
                )
                
                # 隐藏的文本框用于存储音频文件路径
                audio_file_path = gr.Textbox(
                    label="音频文件路径",
                    visible=False,
                    interactive=False
                )
                
                gr.Markdown("#### 💾 保存结果")
                with gr.Group():
                    dataset_output_dir = gr.Textbox(
                        label="数据集输出目录",
                        value="whisper_dataset"
                    )
                    with gr.Row():
                        save_btn = gr.Button(
                            "💾 保存当前结果", 
                            variant="secondary",
                            scale=1
                        )
                        export_btn = gr.Button(
                            "📤 导出所有结果",
                            variant="secondary",
                            scale=1
                        )
                    save_result = gr.Textbox(
                        label="保存状态", 
                        interactive=False
                    )
        
        # ======================= 事件处理函数 =======================
        
        # 音频输入方式切换
        def update_audio_input_interface(mode):
            """根据音频输入方式更新界面"""
            if mode == "文件上传":
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            elif mode == "目录批量":
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            elif mode == "语音录制":
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
        
        # VAD设置显示/隐藏
        def update_vad_settings(enable_vad):
            """根据VAD启用状态更新设置界面"""
            return gr.update(visible=enable_vad)
        
        # 更新引擎设置界面
        def update_engine_settings(engine_type, model_source):
            """根据引擎类型和模型来源更新设置界面"""
            # Faster Whisper设置可见性
            faster_visible = (engine_type == "Faster Whisper")
            # CT2转换功能可见性（仅本地模型且非Faster Whisper时可见）
            ct2_visible = (model_source == "本地模型" and engine_type != "Faster Whisper")
            
            return gr.update(visible=faster_visible), gr.update(visible=ct2_visible)
        
        # 更新模型信息显示
        def update_model_info(engine_type, model_source, model_name):
            """更新模型信息显示"""
            info_lines = []
            info_lines.append(f"引擎类型: {engine_type}")
            info_lines.append(f"模型来源: {model_source}")
            if model_name:
                info_lines.append(f"模型名称: {model_name}")
            return "\n".join(info_lines)
        
        # 简化的处理函数（移除多模型管理功能）
        
        # CT2模型转换
        def convert_model_to_ct2_handler(model_path, output_dir, model_type):
            """处理CT2模型转换"""
            if not CT2_CONVERTER_AVAILABLE:
                return "CT2转换器不可用，请安装: pip install ct2-transformers-converter"
            
            if model_type == "本地模型":
                if not model_path or not os.path.exists(model_path):
                    return "请提供有效的本地模型路径"
                source_path = model_path
            else:
                return "CT2转换仅支持本地模型"
            
            if not output_dir.strip():
                return "请指定CT2输出目录"
            
            try:
                result = convert_model_to_ct2(source_path, output_dir.strip())
                return result
            except Exception as e:
                return f"CT2转换失败: {str(e)}"
        
        # 处理音频文件（支持新功能）
        def process_audio_advanced(audio_input_mode, file_input, directory_input, audio_recorder,
                                 model_choice, custom_model_path, model_type, whisper_model_type,
                                 task_choice, language_choice, download_root_input,
                                 enable_vad, vad_method, vad_aggressiveness,
                                 device_choice, compute_type):
            """高级音频处理函数，支持多种输入方式和新功能"""
            
            # 确定音频文件路径
            audio_path = None
            if audio_input_mode == "文件上传" and file_input:
                audio_path = file_input.name if hasattr(file_input, 'name') else file_input
            elif audio_input_mode == "语音录制" and audio_recorder:
                audio_path = audio_recorder
            elif audio_input_mode == "目录批量" and directory_input.strip():
                # 目录批量处理 - 支持目录路径和直接音频文件路径
                input_path = directory_input.strip()
                
                # 检查输入是文件还是目录
                if os.path.isfile(input_path):
                    # 直接是音频文件路径
                    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
                    if any(input_path.lower().endswith(ext) for ext in audio_extensions):
                        audio_path = input_path
                    else:
                        return "不支持的音频文件格式", "", "格式错误", ""
                elif os.path.isdir(input_path):
                    # 是目录路径，查找其中的音频文件
                    import glob
                    audio_extensions = ['*.wav', '*.mp3', '*.m4a', '*.flac', '*.ogg']
                    audio_files = []
                    for ext in audio_extensions:
                        audio_files.extend(glob.glob(os.path.join(input_path, ext)))
                    
                    if audio_files:
                        audio_path = audio_files[0]  # 处理第一个文件
                    else:
                        return "目录中未找到支持的音频文件", "", "未找到音频文件", ""
                else:
                    return "输入的路径不存在或无法访问", "", "路径错误", ""
            
            if not audio_path:
                return "请提供音频输入", "", "无音频输入", ""
            
            try:
                # 调用增强的转录函数
                result_text, detected_language, lang_code = transcribe_audio(
                    model_name=model_choice,
                    file_path=audio_path,
                    task=task_choice,
                    language=language_choice,
                    custom_model_path=custom_model_path,
                    download_root=download_root_input,
                    model_type=model_type,
                    enable_vad=enable_vad,
                    vad_method=vad_method,
                    vad_aggressiveness=int(vad_aggressiveness),
                    enable_multi_model=False,
                    whisper_model_type=whisper_model_type,
                    device=device_choice,
                    compute_type=compute_type
                )
                
                status = "处理完成"
                if enable_vad:
                    status += f" (VAD: {vad_method})"
                
                return result_text, detected_language, status, audio_path
                
            except Exception as e:
                return f"处理失败: {str(e)}", "", "处理失败", ""
        
        # 清空结果
        def clear_results():
            """清空所有结果"""
            return "", "", "已清空", ""
        
        # ======================= 事件绑定 =======================
        
        # 基础事件
        task_choice.change(
            fn=update_language_visibility,
            inputs=task_choice,
            outputs=language_choice
        )
        
        model_type.change(
            fn=update_model_interface,
            inputs=model_type,
            outputs=[model_choice, custom_model_path, download_root_input]
        )
        
        # 新增事件
        audio_input_mode.change(
            fn=update_audio_input_interface,
            inputs=audio_input_mode,
            outputs=[file_input, directory_input, audio_recorder]
        )
        
        enable_vad.change(
            fn=update_vad_settings,
            inputs=enable_vad,
            outputs=vad_settings
        )
        
        # 引擎设置联动事件
        whisper_model_type.change(
            fn=update_engine_settings,
            inputs=[whisper_model_type, model_type],
            outputs=[faster_whisper_settings, ct2_conversion_group]
        )
        
        model_type.change(
            fn=update_engine_settings,
            inputs=[whisper_model_type, model_type],
            outputs=[faster_whisper_settings, ct2_conversion_group]
        )
        
        # 模型信息更新事件
        for component in [whisper_model_type, model_type, model_choice]:
            component.change(
                fn=update_model_info,
                inputs=[whisper_model_type, model_type, model_choice],
                outputs=model_info_display
            )
        
        # CT2转换事件
        convert_to_ct2.click(
            fn=convert_model_to_ct2_handler,
            inputs=[custom_model_path, ct2_output_dir, model_type],
            outputs=save_result
        )
        
        # 主要处理事件
        submit_btn.click(
            fn=process_audio_advanced,
            inputs=[
                audio_input_mode, file_input, directory_input, audio_recorder,
                model_choice, custom_model_path, model_type, whisper_model_type,
                task_choice, language_choice, download_root_input,
                enable_vad, vad_method, vad_aggressiveness,
                device_choice, compute_type
            ],
            outputs=[output_text, detected_lang, processing_status, audio_file_path]
        )
        
        # 清空结果事件
        clear_btn.click(
            fn=clear_results,
            outputs=[output_text, detected_lang, processing_status, audio_file_path]
        )
        
        # 保存结果事件
        save_btn.click(
            fn=save_dataset_entry,
            inputs=[output_text, audio_file_path, dataset_output_dir],
            outputs=save_result
        )

# 创建日志目录
os.makedirs("training_logs", exist_ok=True)

# 确保缓存目录存在
cache_dir = os.environ.get('GRADIO_CACHE_DIR')
if cache_dir and not os.path.exists(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)

if __name__ == "__main__":
    print(f"临时目录: {temp_dir}")
    print(f"模型将下载到: {os.getcwd()} (默认)")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        show_api=False,
        allowed_paths=["/root/sj-fs"]
    )

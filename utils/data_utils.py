import argparse
import functools
import gc
import os

import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from utils.data_utils import DataCollatorSpeechSeq2SeqWithPaddingforEval, remove_punctuation, to_simple, to_lower
from utils.reader import CustomDataset
from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("test_data",   type=str, default="dataset/test.json",  help="测试集的路径")
add_arg("out_data",   type=str, default="dataset/out.txt",    help="解码输出路径")
add_arg("model_path",  type=str, default="models/whisper-tiny-finetune", help="合并模型的路径，或者是huggingface上模型的名称")
add_arg("num_beams", type=int, default=4,         help="number of beams")
add_arg("batch_size",  type=int, default=16,        help="评估的batch size")
add_arg("num_workers", type=int, default=8,         help="读取数据的线程数量")
add_arg("language",    type=str, default="zh", help="设置语言，可全称也可简写，如果为None则评估的是多语言")
add_arg("remove_pun",  type=bool, default=True,     help="是否移除标点符号")
add_arg("to_simple",   type=bool, default=True,     help="是否转为简体中文")
add_arg("timestamps",  type=bool, default=False,    help="评估时是否使用时间戳数据")
add_arg("min_audio_len",     type=float, default=0.5,  help="最小的音频长度，单位秒")
add_arg("max_audio_len",     type=float, default=30,   help="最大的音频长度，单位秒")
add_arg("local_files_only",  type=bool,  default=True, help="是否只在本地加载模型，不尝试下载")
add_arg("task",       type=str, default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
add_arg("max_samples",  type=int, default=None,      help="从测试集中使用的最大样本数量，None表示使用全部数据")
add_arg("metric",     type=str, default="both",       choices=['cer', 'wer', 'both'],      help="评估方式，both表示同时计算CER和WER")
args = parser.parse_args()
print_arguments(args)

# 判断模型路径是否合法
assert 'openai' == os.path.dirname(args.model_path) or os.path.exists(args.model_path), \
    f"模型文件{args.model_path}不存在，请检查是否已经成功合并模型，或者是否为huggingface存在模型"
# 获取Whisper的数据处理器，这个包含了特征提取器、tokenizer
processor = WhisperProcessor.from_pretrained(args.model_path,
                                             language=args.language,
                                             task=args.task,
                                             no_timestamps=not args.timestamps,
                                             local_files_only=args.local_files_only)
forced_decoder_ids = processor.get_decoder_prompt_ids()
# 获取模型
model = WhisperForConditionalGeneration.from_pretrained(args.model_path,
                                                        device_map="auto",
                                                        local_files_only=args.local_files_only)
model.eval()

# 获取测试数据
test_dataset = CustomDataset(data_list_path=args.test_data,
                             processor=processor,
                             timestamps=args.timestamps,
                             min_duration=args.min_audio_len,
                             max_duration=args.max_audio_len)
print(f"测试数据：{len(test_dataset)}")

# 数据padding器
data_collator = DataCollatorSpeechSeq2SeqWithPaddingforEval(processor=processor)
eval_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, collate_fn=data_collator)

# 获取评估方法
if args.metric == 'both':
    cer_metric = evaluate.load(f'metrics/cer.py')
    wer_metric = evaluate.load(f'metrics/wer.py')
    cer_has_data = False
    wer_has_data = False
else:
    metric = evaluate.load(f'metrics/{args.metric}.py')
    metric_has_data = False

OUT = open(args.out_data, 'w', encoding='utf-8')

# 开始评估
processed_samples = 0
total_batches = len(eval_dataloader)
if args.max_samples is not None:
    # 估算需要处理的批次数量
    estimated_batches = min(total_batches, (args.max_samples // args.batch_size) + 1)
    print(f"预计需要处理 {estimated_batches} 个批次以达到 {args.max_samples} 个样本")

for step, batch in enumerate(tqdm(eval_dataloader, desc="评估进度")):
    # 检查是否达到最大样本数量限制
    if args.max_samples is not None and processed_samples >= args.max_samples:
        print(f"已达到最大样本数量限制 {args.max_samples}，停止处理")
        break
    
    # 检查batch是否为空或无效
    if not batch['id'] or len(batch['id']) == 0:
        continue
    
    # 检查input_features是否有效
    if batch["input_features"].size(0) == 0:
        continue
        
    with torch.amp.autocast('cuda'):
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].cuda(),
                    decoder_input_ids=batch["labels"][:, :4].cuda(),
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=255,
                    num_beams=args.num_beams).cpu().numpy())
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            # 将预测和实际的token转换为文本
            decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
            # 删除标点符号
            if args.remove_pun:
                decoded_preds = remove_punctuation(decoded_preds)
                decoded_labels = remove_punctuation(decoded_labels)
            # 将繁体中文总成简体中文
            if args.to_simple:
                decoded_preds = to_simple(decoded_preds)
                decoded_labels = to_simple(decoded_labels)
            if args.language == 'en':
                decoded_preds = to_lower(decoded_preds)
                decoded_labels = to_lower(decoded_labels)
            
            # 过滤空参考样本，同时考虑max_samples限制
            filtered_preds = []
            filtered_labels = []
            filtered_ids = []
            for i, (audio_id, pred_text, label_text) in enumerate(zip(batch['id'], decoded_preds, decoded_labels)):
                # 检查是否已达到样本数量限制
                if args.max_samples is not None and processed_samples >= args.max_samples:
                    break
                    
                # 过滤掉空的或只包含空白字符的参考文本
                if label_text.strip():
                    filtered_preds.append(pred_text)
                    filtered_labels.append(label_text)
                    filtered_ids.append(audio_id)
                    OUT.write('{}	{}\n'.format(audio_id, pred_text))
                    processed_samples += 1
            
            # 如果达到限制，提前退出
            if args.max_samples is not None and processed_samples >= args.max_samples:
                print(f"已处理 {processed_samples} 个样本，达到限制，提前结束")
                break
            
            # 添加到评估指标
            if filtered_preds:  # 只有在有有效样本时才添加
                if args.metric == 'both':
                    cer_metric.add_batch(predictions=filtered_preds, references=filtered_labels)
                    wer_metric.add_batch(predictions=filtered_preds, references=filtered_labels)
                    cer_has_data = True
                    wer_has_data = True
                else:
                    metric.add_batch(predictions=filtered_preds, references=filtered_labels)
                    metric_has_data = True
    
    # 删除计算的记录
    del generated_tokens, labels, batch
    if step % 10 == 0:  # 每10个批次清理一次内存
        gc.collect()
    
    # 如果已达到样本限制，退出循环
    if args.max_samples is not None and processed_samples >= args.max_samples:
        break
OUT.close()
# 计算评估结果
if processed_samples > 0:
    if args.metric == 'both':
        if cer_has_data and wer_has_data:
            cer_result = cer_metric.compute()
            wer_result = wer_metric.compute()
            print(f"评估结果：CER={round(cer_result, 5)}, WER={round(wer_result, 5)}")
        else:
            print("警告：评估指标中没有足够的数据进行计算")
    else:
        if metric_has_data:
            m = metric.compute()
            print(f"评估结果：{args.metric}={round(m, 5)}")
        else:
            print("警告：评估指标中没有足够的数据进行计算")
else:
    print("警告：没有有效样本被处理，无法计算评估结果")
print(f"实际处理的样本数量：{processed_samples}")

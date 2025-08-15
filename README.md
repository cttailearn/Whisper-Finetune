# 微调Whisper语音识别模型和加速推理

## 前言

OpenAI在开源了号称其英文语音辨识能力已达到人类水准的Whisper项目，且它亦支持其它98种语言的自动语音辨识。Whisper所提供的自动语音识与翻译任务，它们能将各种语言的语音变成文本，也能将这些文本翻译成英文。本项目主要的目的是为了对Whisper模型使用Lora进行微调，**支持无时间戳数据训练，有时间戳数据训练、无语音数据训练**。目前开源了好几个模型，具体可以在[openai](https://huggingface.co/openai)查看，下面列出了常用的几个模型。另外项目最后还支持CTranslate2加速推理和GGML加速推理，提示一下，加速推理支持直接使用Whisper原模型转换，并不一定需要微调。支持Windows桌面应用，Android应用和服务器部署。

## 目录
 - [项目主要程序介绍](#项目主要程序介绍)
 - [模型效果](#模型效果)
 - [安装环境](#安装环境)
 - [准备数据](#准备数据)
 - [Web界面训练工具](#Web界面训练工具)
   - [启动Web界面](#启动Web界面)
   - [数据准备功能](#数据准备功能)
   - [模型训练功能](#模型训练功能)
 - [微调模型](#微调模型)
   - [单卡训练](#单卡训练)
   - [多卡训练](#多卡训练)
 - [合并模型](#合并模型)
 - [评估模型](#评估模型)
 - [预测](#预测)
 - [加速预测](#加速预测)
 - [GUI界面预测](#GUI界面预测)
 - [Web部署](#Web部署)
   - [接口文档](#接口文档)
 - [Android部署](#Android部署)
 - [Windows桌面应用](#Windows桌面应用)

<a name='项目主要程序介绍'></a>

## 项目主要程序介绍

1. `aishell.py`：制作AIShell训练数据。
2. `finetune.py`：PEFT方式微调模型。
3. `finetune_all.py`：全参数微调模型。
4. `train_webui.py`：**Web界面训练工具**，提供可视化的数据准备和模型训练界面，支持LoRA微调和全参数微调。
5. `merge_lora.py`：合并Whisper和Lora的模型。
6. `evaluation.py`：评估使用微调后的模型或者Whisper原模型。
7. `infer_tfs.py`：使用transformers直接调用微调后的模型或者Whisper原模型预测，只适合推理短音频。
8. `infer_ct2.py`：使用转换为CTranslate2的模型预测，主要参考这个程序用法。
9. `infer_gui.py`：有GUI界面操作，使用转换为CTranslate2的模型预测。
10. `infer_server.py`：使用转换为CTranslate2的模型部署到服务器端，提供给客户端调用。
11. `convert-ggml.py`：转换模型为GGML格式模型，给Android应用或者Windows应用使用。
12. `AndroidDemo`：该目录存放的是部署模型到Android的源码。
13. `WhisperDesktop`：该目录存放的是Windows桌面应用的程序。

## 安装环境

- 首先安装的是Pytorch的GPU版本，以下介绍两种安装Pytorch的方式，只需要选择一种即可。

1. 以下是使用Anaconda安装Pytorch环境，如果已经安装过了，请跳过。
```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

2. 以下是使用Docker镜像，拉取一个Pytorch环境的镜像。
```shell
sudo docker pull pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
```

然后进入到镜像中，同时将当前路径挂载到容器的`/workspace`目录下。
```shell
sudo nvidia-docker run --name pytorch -it -v $PWD:/workspace pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel /bin/bash
```

- 安装所需的依赖库。

```shell
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- Windows需要单独安装bitsandbytes。
```shell
python -m pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.40.1.post1-py3-none-win_amd64.whl
```

<a name='准备数据'></a>

## 准备数据

训练的数据集如下，是一个jsonlines的数据列表，也就是每一行都是一个JSON数据，数据格式如下。本项目提供了一个制作AIShell数据集的程序`aishell.py`，执行这个程序可以自动下载并生成如下列格式的训练集和测试集，**注意：** 这个程序可以通过指定AIShell的压缩文件来跳过下载过程的，如果直接下载会非常慢，可以使用一些如迅雷等下载器下载该数据集，然后通过参数`--filepath`指定下载的压缩文件路径，如`/home/test/data_aishell.tgz`。

**小提示：**
1. 如果不使用时间戳训练，可以不包含`sentences`字段的数据。
2. 如果只有一种语言的数据，可以不包含`language`字段数据。
3. 如果训练空语音数据，`sentences`字段为`[]`，`sentence`字段为`""`，`language`字段可以不存在。
4. 数据可以不包含标点符号，但微调的模型会损失添加符号能力。

```json
{
   "audio": {
      "path": "dataset/0.wav"
   },
   "sentence": "近几年，不但我用书给女儿压岁，也劝说亲朋不要给女儿压岁钱，而改送压岁书。",
   "language": "Chinese",
   "sentences": [
      {
         "start": 0,
         "end": 1.4,
         "text": "近几年，"
      },
      {
         "start": 1.42,
         "end": 8.4,
         "text": "不但我用书给女儿压岁，也劝说亲朋不要给女儿压岁钱，而改送压岁书。"
      }
   ],
   "duration": 7.37
}
```

<a name='Web界面训练工具'></a>

## Web界面训练工具

`train_webui.py` 是一个基于Gradio的Web界面训练工具，为用户提供了可视化的数据准备和模型训练界面。相比命令行方式，Web界面更加直观易用，特别适合初学者和需要频繁调整参数的用户。

### 主要功能特点

- **可视化数据准备**：支持音频文件上传、数据集格式转换、数据预览等功能
- **智能数据划分**：自动随机划分训练集和测试集，支持自定义划分比例
- **双模式训练**：支持LoRA微调和全参数微调两种训练模式
- **实时训练监控**：提供训练进度显示、日志输出、训练曲线等实时监控功能
- **脚本自动生成**：可以将训练配置自动生成为shell脚本，方便后续批量训练
- **参数预设管理**：支持保存和加载常用的训练参数配置

<a name='启动Web界面'></a>

### 启动Web界面

使用以下命令启动Web界面：

```shell
python train_webui.py
```

启动后，在浏览器中访问显示的本地地址（通常是 `http://127.0.0.1:7860`）即可使用Web界面。

<a name='数据准备功能'></a>

### 数据准备功能

Web界面的数据准备模块提供以下功能：

1. **数据集上传**：支持批量上传音频文件和对应的标注文件
2. **格式转换**：自动将不同格式的数据转换为训练所需的jsonlines格式
3. **数据预览**：可以预览数据集内容，检查数据质量
4. **随机划分**：自动将数据集随机划分为训练集和测试集
   - 支持自定义训练集比例（默认90%）
   - 确保数据划分的随机性和均匀性
5. **数据统计**：显示数据集的基本统计信息，如总时长、样本数量等

<a name='模型训练功能'></a>

### 模型训练功能

Web界面的模型训练模块支持以下功能：

#### 训练模式选择

- **LoRA微调模式**：
  - 使用 `finetune.py` 脚本进行训练
  - 支持AdaLoRA自适应参数分配
  - 显存占用少，训练速度快
  - 生成的脚本文件以 `train_lora_` 开头

- **全参数微调模式**：
  - 使用 `finetune_all.py` 脚本进行训练
  - 支持编码器冻结选项
  - 训练效果更好，无需后续合并步骤
  - 生成的脚本文件以 `train_full_` 开头

#### 参数配置

界面提供了丰富的训练参数配置选项：

- **基础设置**：基础模型选择、输出目录、GPU设置等
- **训练参数**：学习率、批次大小、训练轮数、梯度累积等
- **优化选项**：FP16训练、8-bit量化、梯度检查点等
- **LoRA参数**：LoRA rank、alpha值、dropout等（仅LoRA模式）
- **冻结选项**：编码器冻结设置（仅全参数模式）

#### 训练监控

- **实时日志**：显示训练过程中的详细日志信息
- **进度跟踪**：显示当前训练进度和预计完成时间
- **命令预览**：实时显示生成的训练命令，方便调试
- **脚本下载**：支持将训练配置保存为shell脚本文件

#### 脚本生成功能

训练界面支持将当前配置的训练参数自动生成为可执行的shell脚本：

- **智能命名**：根据训练模式自动生成脚本文件名
  - LoRA微调：`train_lora_YYYYMMDD_HHMMSS.sh`
  - 全参数微调：`train_full_YYYYMMDD_HHMMSS.sh`
- **完整配置**：脚本包含所有训练参数和环境设置
- **使用说明**：脚本中包含详细的使用说明和注意事项
- **跨平台兼容**：支持Windows和Linux系统

生成的脚本可以直接在命令行中执行，方便进行批量训练或在服务器上运行。

<a name='微调模型'></a>

## 微调模型

准备好数据之后，就可以开始微调模型了。本项目支持两种微调方式：

1. **LoRA微调（推荐）**：使用参数高效微调技术，显存占用少，训练速度快
2. **全参数微调**：微调模型的所有参数，效果更好但需要更多显存

### 训练模式选择

**LoRA微调**（使用 `finetune.py`）：
- 适合显存有限的情况
- 训练速度快，显存占用少
- 支持AdaLoRA自适应参数分配
- 需要后续合并模型步骤

**全参数微调**（使用 `finetune_all.py`）：
- 适合显存充足的情况
- 微调效果更好，无需合并模型
- 支持编码器冻结选项（`--freeze_encoder`）
- 直接输出完整的微调模型

### 重要参数说明

训练最重要的参数包括：
- `--base_model`：指定微调的Whisper模型，需要在[HuggingFace](https://huggingface.co/openai)存在
- `--output_dir`：训练时保存模型的路径
- `--freeze_encoder`：（仅全参数微调）是否冻结编码器，只微调解码器
- `--use_8bit`：是否使用8位量化，设置为False可以提高训练速度

如果显存足够的话，建议将`--use_8bit`设置为False，这样训练速度会快很多。

<a name='单卡训练'></a>

### 单卡训练

单卡训练命令如下，Windows系统可以不添加`CUDA_VISIBLE_DEVICES`参数。

**LoRA微调：**
```shell
CUDA_VISIBLE_DEVICES=0 python finetune.py --base_model=openai/whisper-tiny --output_dir=output/
```

**全参数微调：**
```shell
CUDA_VISIBLE_DEVICES=0 python finetune_all.py --base_model=openai/whisper-tiny --output_dir=output/
```

**全参数微调（冻结编码器）：**
```shell
CUDA_VISIBLE_DEVICES=0 python finetune_all.py --base_model=openai/whisper-tiny --output_dir=output/ --freeze_encoder
```

<a name='多卡训练'></a>

### 多卡训练

多卡训练有两种方法，分别是torchrun和accelerate，开发者可以根据自己的习惯使用对应的方式。

1. 使用torchrun启动多卡训练，命令如下，通过`--nproc_per_node`指定使用的显卡数量。

**LoRA微调：**
```shell
torchrun --nproc_per_node=2 finetune.py --base_model=openai/whisper-tiny --output_dir=output/
```

**全参数微调：**
```shell
torchrun --nproc_per_node=2 finetune_all.py --base_model=openai/whisper-tiny --output_dir=output/
```

**全参数微调（冻结编码器）：**
```shell
torchrun --nproc_per_node=2 finetune_all.py --base_model=openai/whisper-tiny --output_dir=output/ --freeze_encoder
```

2. 使用accelerate启动多卡训练，如果是第一次使用accelerate，要配置训练参数，方式如下。

首先配置训练参数，过程是让开发者回答几个问题，基本都是默认就可以，但有几个参数需要看实际情况设置。
```shell
accelerate config
```

大概过程就是这样：
```
--------------------------------------------------------------------In which compute environment are you running?
This machine
--------------------------------------------------------------------Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]:
Do you wish to optimize your script with torch dynamo?[yes/NO]:
Do you want to use DeepSpeed? [yes/NO]:
Do you want to use FullyShardedDataParallel? [yes/NO]:
Do you want to use Megatron-LM ? [yes/NO]: 
How many GPU(s) should be used for distributed training? [1]:2
What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:
--------------------------------------------------------------------Do you wish to use FP16 or BF16 (mixed precision)?
fp16
accelerate configuration saved at /home/test/.cache/huggingface/accelerate/default_config.yaml
```

配置完成之后，可以使用以下命令查看配置。
```shell
accelerate env
```

开始训练命令如下。

**LoRA微调：**
```shell
accelerate launch finetune.py --base_model=openai/whisper-tiny --output_dir=output/
```

**全参数微调：**
```shell
accelerate launch finetune_all.py --base_model=openai/whisper-tiny --output_dir=output/
```

**全参数微调（冻结编码器）：**
```shell
accelerate launch finetune_all.py --base_model=openai/whisper-tiny --output_dir=output/ --freeze_encoder
```


输出日志如下：
```shell
{'loss': 0.9098, 'learning_rate': 0.000999046843662503, 'epoch': 0.01}                                                     
{'loss': 0.5898, 'learning_rate': 0.0009970611012927184, 'epoch': 0.01}                                                    
{'loss': 0.5583, 'learning_rate': 0.0009950753589229333, 'epoch': 0.02}                                                  
{'loss': 0.5469, 'learning_rate': 0.0009930896165531485, 'epoch': 0.02}                                          
{'loss': 0.5959, 'learning_rate': 0.0009911038741833634, 'epoch': 0.03}
```

<a name='合并模型'></a>

## 合并模型

**注意：** 只有LoRA微调需要合并模型，全参数微调直接输出完整模型，无需此步骤。

PEFT方式（LoRA）微调模型完成之后会有两个模型，第一个是Whisper基础模型，第二个是Lora模型，需要把这两个模型合并之后才能进行后续操作。这个程序只需要传递两个参数，`--lora_model`指定的是训练结束后保存的Lora模型路径，其实就是检查点文件夹路径，第二个`--output_dir`是合并后模型的保存目录。

**LoRA模型合并：**
```shell
python merge_lora.py --lora_model=output/whisper-tiny/checkpoint-best/ --output_dir=models/
```

<a name='评估模型'></a>

## 评估模型

执行以下程序进行评估模型，最重要的两个参数分别是。第一个`--model_path`指定的是合并后的模型路径，同时也支持直接使用Whisper原模型，例如直接指定`openai/whisper-large-v2`，第二个是`--metric`指定的是评估方法，例如有字错率`cer`和词错率`wer`。**提示：** 没有微调的模型，可能输出带有标点符号，影响准确率。其他更多的参数请查看这个程序。
```shell
python evaluation.py --model_path=models/whisper-tiny-finetune --metric=cer
```

<a name='预测'></a>

## 预测

执行以下程序进行语音识别，这个使用transformers直接调用微调后的模型或者Whisper原模型预测，只适合推理短音频，长语音还是参考`infer_ct2.py`的使用方式。第一个`--audio_path`参数指定的是要预测的音频路径。第二个`--model_path`指定的是合并后的模型路径，同时也支持直接使用Whisper原模型，例如直接指定`openai/whisper-large-v2`。其他更多的参数请查看这个程序。
```shell
python infer_tfs.py --audio_path=dataset/test.wav --model_path=models/whisper-tiny-finetune
```

<a name='加速预测'></a>

## 加速预测

众所周知，直接使用Whisper模型推理是比较慢的，所以这里提供了一个加速的方式，主要是使用了CTranslate2进行加速，首先要转换模型，把合并后的模型转换为CTranslate2模型。如下命令，`--model`参数指定的是合并后的模型路径，同时也支持直接使用Whisper原模型，例如直接指定`openai/whisper-large-v2`。`--output_dir`参数指定的是转换后的CTranslate2模型路径，`--quantization`参数指定的是量化模型大小，不希望量化模型的可以直接去掉这个参数。
```shell
ct2-transformers-converter --model models/whisper-tiny-finetune --output_dir models/whisper-tiny-finetune-ct2 --copy_files tokenizer.json --quantization float16
```

执行以下程序进行加速语音识别，`--audio_path`参数指定的是要预测的音频路径。`--model_path`指定的是转换后的CTranslate2模型。其他更多的参数请查看这个程序。
```shell
python infer_ct2.py --audio_path=dataset/test.wav --model_path=models/whisper-tiny-finetune-ct2
```

输出结果如下：
```shell
-----------  Configuration Arguments -----------
audio_path: dataset/test.wav
model_path: models/whisper-tiny-finetune-ct2
language: zh
use_gpu: True
use_int8: False
beam_size: 10
num_workers: 1
vad_filter: False
local_files_only: True
------------------------------------------------
[0.0 - 8.0]：近几年,不但我用书给女儿压碎,也全说亲朋不要给女儿压碎钱,而改送压碎书。
```

<a name='GUI界面预测'></a>

## GUI界面预测

这里同样是使用了CTranslate2进行加速，转换模型方式看上面文档。`--model_path`指定的是转换后的CTranslate2模型。其他更多的参数请查看这个程序。

```shell
python infer_gui.py --model_path=models/whisper-tiny-finetune-ct2
```

启动后界面如下：

<div align="center">
<img src="./docs/images/gui.jpg" alt="GUI界面" width="600"/>
</div>

<a name='Web部署'></a>

## Web部署

Web部署同样是使用了CTranslate2进行加速，转换模型方式看上面文档。`--host`指定服务启动的地址，这里设置为`0.0.0.0`，即任何地址都可以访问。`--port`指定使用的端口号。`--model_path`指定的是转换后的CTranslate2模型。`--num_workers`指定是使用多少个线程并发推理，这在Web部署上很重要，当有多个并发访问是可以同时推理。其他更多的参数请查看这个程序。

```shell
python infer_server.py --host=0.0.0.0 --port=5000 --model_path=models/whisper-tiny-finetune-ct2 --num_workers=2
```

### 接口文档

目前提供两个接口，普通的识别接口`/recognition`和流式返回结果`/recognition_stream`，注意这个流式是指流式返回识别结果，同样是上传完整的音频，然后流式返回识别结果，这种方式针对长语音识别体验非常好。他们的文档接口是完全一致的，接口参数如下。

|     字段     | 是否必须 |   类型   |    默认值     |              说明               |
|:----------:|:----:|:------:|:----------:|:-----------------------------:|
|   audio    |  是   |  File  |            |           要识别的音频文件            |
| to_simple  |  否   |  int   |     1      |            是否繁体转简体            |
| remove_pun |  否   |  int   |     0      |           是否移除标点符号            |
|    task    |  否   | String | transcribe | 识别任务类型，支持transcribe和translate |
|  language  |  否   | String |     zh     |    设置语言，简写，如果为None则自动检测语言     |


返回结果：

|   字段    |  类型  |      说明       |
|:-------:|:----:|:-------------:|
| results | list |    分割的识别结果    |
| +result | str  |   每片分隔的文本结果   |
| +start  | int  | 每片分隔的开始时间，单位秒 |
|  +end   | int  | 每片分隔的结束时间，单位秒 |
|  code   | int  |  错误码，0即为成功识别  |

示例如下：
```json
{
  "results": [
    {
      "result": "近几年,不但我用书给女儿压碎,也全说亲朋不要给女儿压碎钱,而改送压碎书。",
      "start": 0,
      "end": 8
    }
  ],
  "code": 0
}
```

为了方便理解，这里提供了调用Web接口的Python代码，下面的是`/recognition`的调用方式。
```python
import requests

response = requests.post(url="http://127.0.0.1:5000/recognition", 
                         files=[("audio", ("test.wav", open("dataset/test.wav", 'rb'), 'audio/wav'))],
                         json={"to_simple": 1, "remove_pun": 0, "language": "zh", "task": "transcribe"}, timeout=20)
print(response.text)
```

下面的是`/recognition_stream`的调用方式。
```python
import json
import requests

response = requests.post(url="http://127.0.0.1:5000/recognition_stream",
                         files=[("audio", ("test.wav", open("dataset/test_long.wav", 'rb'), 'audio/wav'))],
                         json={"to_simple": 1, "remove_pun": 0, "language": "zh", "task": "transcribe"}, stream=True, timeout=20)
for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
    if chunk:
        result = json.loads(chunk.decode())
        text = result["result"]
        start = result["start"]
        end = result["end"]
        print(f"[{start} - {end}]：{text}")
```


提供的测试页面如下：

首页`http://127.0.0.1:5000/` 的页面如下：

<div align="center">
<img src="./docs/images/web.jpg" alt="首页" width="600"/>
</div>

文档页面`http://127.0.0.1:5000/docs` 的页面如下：

<div align="center">
<img src="./docs/images/api.jpg" alt="文档页面" width="600"/>
</div>


<a name='Android部署'></a>
## Android部署

安装部署的源码在[AndroidDemo](./AndroidDemo)目录下，具体文档可以到该目录下的[README.md](AndroidDemo/README.md)查看。
<br/>
<div align="center">
<img src="./docs/images/android2.jpg" alt="Android效果图" width="200">
<img src="./docs/images/android1.jpg" alt="Android效果图" width="200">
<img src="./docs/images/android3.jpg" alt="Android效果图" width="200">
<img src="./docs/images/android4.jpg" alt="Android效果图" width="200">
</div>


<a name='Windows桌面应用'></a>
## Windows桌面应用

程序在[WhisperDesktop](./WhisperDesktop)目录下，具体文档可以到该目录下的[README.md](WhisperDesktop/README.md)查看。

<br/>
<div align="center">
<img src="./docs/images/desktop1.jpg" alt="Windows桌面应用效果图">
</div>



## 参考资料

1. https://github.com/huggingface/peft
2. https://github.com/guillaumekln/faster-whisper
3. https://github.com/ggerganov/whisper.cpp
4. https://github.com/Const-me/Whisper

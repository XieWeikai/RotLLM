# Evaluation
## 环境安装
首先安装conda: [conda installation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) 或者和conda兼容使用一致的mamba: [mamba](https://github.com/mamba-org/mamba).

安装环境需要的包
```bash
conda env create -f environment.yml
```
注意由于该`environment.yml`是我直接用我的环境导出的，里面含有很多该项目并不会用到的package，建议自己新建一个环境，在跑代码时碰到缺失的包再安装`environment.yml`中对应版本的包。其中比较重要的`transformers`包版本如下:
```
transformers==4.51.3
```
---
创建好虚拟环境后使用如下命令安装`rotate`包
```bash
pip install -e .
```
## 测试ShowUI-2B
### 下载ShowUI-2B
ShowUI-2B是一个专为GUI任务设计的模型，huggingface模型链接见 https://huggingface.co/showlab/ShowUI-2B.
该链接下有load model对应代码，如下
```python
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

processor = AutoProcessor.from_pretrained("showlab/ShowUI-2B", min_pixels=min_pixels, max_pixels=max_pixels)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "showlab/ShowUI-2B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```
第一次运行上面代码会自动下载`ShowUI-2B`并缓存在本地，后续可以直接通过`showlab/ShowUI-2B`来加载模型。

你也可以通过如下命令将模型下载到指定目录
```bash
huggingface-cli download --repo-type model --local-dir path/to/ShowUI-2B --local-dir-use-symlinks False --resume showlab/ShowUI-2B
```
后续可以用如下代码加载模型
```python
processor = AutoProcessor.from_pretrained("path/to/ShowUI-2B", min_pixels=min_pixels, max_pixels=max_pixels)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "path/to/ShowUI-2B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

### 下载ScreenSpot数据集
ScreenSpot数据集链接为 https://huggingface.co/datasets/rootsautomation/ScreenSpot.
通过如下命令下载
```bash
huggingface-cli download --repo-type dataset --local-dir path/to/ScreenSpot --local-dir-use-symlinks False --resume rootsautomation/ScreenSpot
```
或者也可以像前面使用模型一样，直接使用`rootsautomation/ScreenSpot`来加载数据集，让`transformers`自动缓存该数据集。

### evaluation
在项目根目录下，使用如下命令测试ShowUI-2B旋转量化后在ScreenSpot上的准确率
```bash
CUDA_VISIBLE_DEVICES=... python eval/evaluate_screen_dataset_qkvnobias.py \
--dataset_path path/to/ScreenSpot \
--model_name path/to/ShowUI-2B \
--scale_file eval/scales/showui_lm_vit_rot_screenqa_nobias.json \
--quantize_vit --rotate_vit --online_rotation --R_path eval/R.bin \
--clip_all --save_clip_info
```
各个参数的具体含义可以查看`args.py`，其中`--dataset_path`和`--model_name`为之前下载数据集和模型的位置（如果缓存过了可以直接使用`rootsautomation/ScreenSpot`和`showlab/ShowUI-2B`进行加载）.


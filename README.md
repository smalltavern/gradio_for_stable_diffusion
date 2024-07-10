# gradio构建的SDv3
### 1、安装依赖

```
pip install transformers==4.42.3
pip install diffusers==0.29.2
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install gradio==4.37.2
```
### 2、模型安装
```python
Helsinki-NLP/opus-mt-zh-en # 中文转英文
stabilityai/stable-diffusion-3-medium-diffusers # stable diffusion v3模型
```
### 3、模型运行
```
python app.py
```
from PIL import Image
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from diffusers import DiffusionPipeline
import torch
import IPython.display as display
import argparse

def get_params():
    parser = argparse.ArgumentParser()
    # 使用"Helsinki-NLP/opus-mt-zh-en"预训练模型中->英
    parser.add_argument('--opus_mt_zh_cn', type=str, default="Helsinki-NLP/opus-mt-zh-en",
                        help='Directory path to a batch of content images')
    parser.add_argument('--SDv3', type=str, default="stabilityai/stable-diffusion-3-medium-diffusers",
                        help='Directory path to a batch of style images')
    
    args = parser.parse_args()
    return args

args = get_params()

tokenizer = AutoTokenizer.from_pretrained(args.opus_mt_zh_cn)
model = AutoModelForSeq2SeqLM.from_pretrained(args.opus_mt_zh_cn)
pipeline = DiffusionPipeline.from_pretrained(args.SDv3, torch_dtype=torch.float16)


def translate_zh_en(text):
  inputs = tokenizer(text, return_tensors="pt", padding=True)
  outputs = model.generate(**inputs)
  en_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
  return en_text
 

device = "cuda" if torch.cuda.is_available() else "cpu"
# model = model.to(device)
pipeline = pipeline.to(device)
 
 
def get_completion(prompt):
    res = pipeline(prompt).images[0]
    return res
 
 
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size
 
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
 
    return grid


def generate(prompt, negative_prompt, steps, guidance, width, height):
    prompt = translate_zh_en(prompt)
    negative_prompt = translate_zh_en(negative_prompt)
 
    params = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height
    }
 
    output = pipeline(prompt, **params).images[0]
 
    return output
 
gr.close_all()
demo = gr.Interface(fn=generate,
              inputs=[
                  gr.Textbox(label="", placeholder="Enter your prompt"),
                  gr.Textbox(label="", placeholder="Enter a negative prompt"),
                  gr.Slider(label="Inference Steps", minimum=10, maximum=150, value=28,
                            info="In how many steps will the denoiser denoise the image?"),
                  gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=7.5,
                            info="Controls how much the text prompt influences the result"),
                  gr.Slider(label="Width", minimum=64, maximum=512, step=64, value=512),
                  gr.Slider(label="Height", minimum=64, maximum=512, step=64, value=512),
              ],
              outputs=[gr.Image(label="Result")],
              title="Image Generation with Stable Diffusion v3",
              description="Generate any image with Stable Diffusion v3",
              allow_flagging="never",
            #   examples=["A 3D render of an astronaut walking in a green desert", "A hand-drawn sailboat circled by birds on the sea at sunrise"]
              )
 
 
# debug=True 为设置在单元格输出信息，若不需要去掉也并无影响
demo.launch(share=True, debug=True, server_port=10055)


    
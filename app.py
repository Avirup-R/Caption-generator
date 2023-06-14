import torch
import gradio as gr 
import re 
from transformers import AutoTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel
from bardapi import Bard


device='cpu'
encoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
decoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
model_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint).to(device)

def listtostr(li):
  s=''
  for i in li:
    s+=i
    s+='\n'
  return s


def get_text(prompt):
  token = 'WghfRvtVO_W5Xh8J_q0gM-xlln4FspcCVg-qGbgxvNa0CSayl6eCwxykYGXoBg3Srbu78g.'
  bard = Bard(token=token)
  ans=bard.get_answer("create 5 catchy captions for '"+ prompt+"' with numbering as caption1 and so on with the caption on the same line")['content']
  return ans


def predict(image,max_length=64, num_beams=4):
  image = image.convert('RGB')
  image = feature_extractor(image, return_tensors="pt").pixel_values.to(device)
  clean_text = lambda x: x.replace('<|endoftext|>','').split('\n')[0]
  caption_ids = model.generate(image, max_length = max_length)[0]
  caption_text = clean_text(tokenizer.decode(caption_ids))
  value=get_text(caption_text)
  text=value.splitlines()
  count=0
  no=1
  content=[]
  for i in text:
     new=str(no)
     temp=str('Caption '+ new)
     if temp in i and count<5:
        content.append(i)
        count+=1
        no+=1
  captions=listtostr(content)
  return captions

    
  # return all_captions 
def set_example_image(example: list) -> dict:
    return gr.Image.update(value=example[0])

css = '''
h1#title {
  text-align: center;
}
h3#header {
  text-align: center;
}
img#overview {
  max-width: 1400px;
  max-height: 1200px;
}
img#style-image {
  max-width: 1000px;
  max-height: 600px;
}
'''
demo = gr.Blocks(css=css)
with demo:
  gr.Markdown('''<h1 id="title">Image Caption Generator.</h1>''')
  gr.Markdown('''Made by : Avirup Rakshit''')
  with gr.Column():
        input = gr.inputs.Image(label="Upload your Image", type = 'pil', optional=True)
        output = gr.outputs.Textbox(type="text", label="Captions")
  btn = gr.Button("Genrate Caption")
  btn.click(fn=predict, inputs=input, outputs=output)
demo.launch()

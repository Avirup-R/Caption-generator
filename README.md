## This is a Caption Generator from Image model.

### This Uses a caption generator model (gpt2 image captioning) for converitng image to text(captions).

### Then it uses bard api to change the basic caption to various interesting prompts.(ChatGPT was not taken beacuse the wait time for the response was too long.)

### The Captions are a little less interactive as compared to chatGPT but are good enough.(with the timetrade off I think bard does a better job.)

### It uses gradio as an user interface.

## To Run

pip install -r requirements.txt <br>
If you do not have nvidia gpu change the device from cuda to cpu. <br>
python3 app.py(in linux and anaconda)  <br>
python app.py (in windows) <br>

#importing
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex,GPTSimpleVectorIndex, PromptHelper
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LLMPredictor
import torch
from langchain.llms.base import LLM
from transformers import pipeline
import gradio as gr
import pandas as pd
import pytesseract
from PIL import Image
import fitz
from llama_index import Document
import os
from langchain.document_loaders import CSVLoader
import moviepy.editor as mp

#initlizing
class FlanLLM(LLM):
    model_name = "google/flan-t5-small"
    pipeline = pipeline("text2text-generation", model=model_name)

    def _call(self, prompt, stop=None):
        return self.pipeline(prompt, max_length=9999)[0]["generated_text"]
 
    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    def _llm_type(self):
        return "custom"

llm_predictor = LLMPredictor(llm=FlanLLM())
hfemb = HuggingFaceEmbeddings()
embed_model = LangchainEmbedding(hfemb)
#audio to text conversion
# whisper = pipeline('automatic-speech-recognition', model = 'openai/whisper-medium')

# set number of output tokens
num_output = 250
# set maximum input size
max_input_size = 512
# set maximum chunk overlap
max_chunk_overlap = 20


prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

#Define functions
def build_the_bot(input_text):
  text_list = [input_text]
  documents = [Document(t) for t in text_list]
  # print(documents)
  global index
  index = GPTSimpleVectorIndex(documents, embed_model=embed_model, llm_predictor=llm_predictor)
  index.save_to_disk("index.json")
  return('Index saved successfull!!!')

def chat(chat_history, user_input):
  # global history
  # history=chat_history
  bot_response = index.query(user_input)
  #print(bot_response)
  response = ""
  for letter in ''.join(bot_response.response): #[bot_response[i:i+1] for i in range(0, len(bot_response), 1)]:
      response += letter + ""
      yield chat_history + [(user_input, response)]

def read_text_file(file):
    with open(file.name, "r") as f:
        text = f.read()
        build_the_bot(text)
    return text

def read_image_file(file):
    print(file.name)
    text = pytesseract.image_to_string(Image.open(file.name))
    build_the_bot(text)
    return text

def read_pdf(file):
  doc = fitz.open(file.name)
  text = ""
  for page in doc:
    text+=page.get_text()
  build_the_bot(text)
  return text

def find_file_type(input,file):
    if input:
      return build_the_bot(input)
    
    if file:
        filename, file_extension = os.path.splitext(file.name)
        if file_extension.lower()=='.txt':
          text=read_text_file(file)
        elif file_extension.lower() in ['.png','.jpg','.jpeg']:
          text=read_image_file(file)
        elif file_extension.lower()=='.pdf':
          text=read_pdf(file)
        # elif file_extension.lower() in ['.mp3','.wav','.m4a']:
        #   text = transcribe2(file.name)
        # elif file_extension=='.mp4':
        #   text = video(file.name)
        else:
          text='Try with Text, Image and PDF Document only.'  
        return text
  
def get_history(history):
  with open('Chat_Histroy.txt','w') as file:
    for c in history:
      file.write(c[0]+':'+c[1]+'\n')
  return 'Chat_Histroy.txt'

# def transcribe2(audio):
#     text = whisper(audio)
#     build_the_bot(text['text'])
#     return text['text']

# def video(file):
#   clip = mp.VideoFileClip(file)
#   clip.audio.write_audiofile('audio.wav')
#   text = whisper('audio.wav')
#   build_the_bot(text['text'])
#   return text['text']

#Gradio
with gr.Blocks() as demo:
    f=0
    gr.Markdown('# <center>INTELLIGENT ANALYSIS OF DOCUMENT DRIVEN QA CHATBOT</center>')
    with gr.Column(variant="panel"):
      gr.Markdown("## Feed the Bot ")
      with gr.Row(variant="compact"):    
          input=gr.Textbox(
              label='Enter the Content',
              placeholder='Prompt engineering is a concept in artificial intelligence (AI), particularly natural language processing (NLP). In prompt engineering, the description of the task that the AI is supposed to accomplish is embedded in the input, e.g. as a question, instead of it being implicitly given. Prompt engineering typically works by converting one or more tasks to a prompt-based dataset and training a language model with what has been called "prompt-based learning" or just "prompt learning".'
          )
          with gr.Column(variant="panel"): 
            gr.Markdown("## <center>Or Upload your Document <center>")
            with gr.Row(variant="compact"):
              files = gr.File()
      text_output = gr.Textbox(
          label="Verify the content",
          placeholder="Start Building the Bot to view the content")
      text_button = gr.Button("Build the Bot!!!")
      text_button.click(find_file_type, [input,files], text_output)
      
      gr.Markdown("# <hr><center>Knowledge Bot</center>")
      gr.Markdown("### Chat with our Knowledge Bot and get answer for your queries")
      chatbot = gr.Chatbot()
      message = gr.Textbox(
          label='AskMe',
          value="What is this document about?")
      message.submit(chat, [chatbot, message], chatbot)

      with gr.Column(variant="panel"):
        with gr.Row(variant="compact"):
          get=gr.Button('Get History')
          clear=gr.Button('Clear History')
        outputs=gr.outputs.File("txt")
        # show=gr.Textbox("Get the history")
        get.click(get_history,chatbot,outputs)
        
        clear.click(lambda: None, None, chatbot)

    demo.queue(max_size=200)
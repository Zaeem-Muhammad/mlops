# app.py
import gradio as gr
from model import predict

def inference(text):
    return predict(text)

iface = gr.Interface(fn=inference, inputs="text", outputs="text")
iface.launch()

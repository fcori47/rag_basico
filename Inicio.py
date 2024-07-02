import gradio as gr
from common.rag_pago import respuesta


gr.ChatInterface(respuesta).launch()
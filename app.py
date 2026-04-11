import gradio as gr
import asyncio
import os
import sys
from inference import run_inference

# Create Gradio interface
with gr.Blocks(title="🎫 OpenEnv Ticket Triage") as demo:
    gr.Markdown("# 🎫 OpenEnv Ticket Triage Classification")
    gr.Markdown("Classifies support tickets into categories: Billing, Technical, General")
    
    with gr.Row():
        run_button = gr.Button("Run Classification", size="lg")
        result_output = gr.JSON(label="Results")
    
    def run_classification():
        try:
            asyncio.run(run_inference())
            return {"status": "success", "message": "Classification complete"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    run_button.click(fn=run_classification, outputs=result_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
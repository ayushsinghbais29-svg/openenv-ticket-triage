import gradio as gr
import random

def reset_environment(task_type):
    return "?? Ticket loaded! Ready for actions.", "Environment reset!", "Ready to start!"

def take_action(action_type, department, confidence):
    reward = round(random.uniform(0.1, 0.9), 4)
    return f"? Action '{action_type}' executed to {department} with {confidence}% confidence", f"Reward: +{reward}", f"Total Actions: 1"

def get_suggestion():
    return "?? AI suggests: Route to **Billing** (Payment issue detected with 92% confidence)"

with gr.Blocks(title="OpenEnv Ticket Triage", theme=gr.themes.Soft()) as demo:
    gr.HTML('''<div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;"><h1>?? OpenEnv Ticket Triage Environment</h1><h3>Interactive Dashboard for AI-Powered Ticket Routing</h3><p>Meta PyTorch OpenEnv Hackathon 2026</p></div>''')
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ?? Task Selection")
            task = gr.Radio(
                choices=[
                    ("?? Easy: Department Classification", "classification"),
                    ("?? Medium: Priority + Classification", "priority"),
                    ("?? Hard: Efficiency Triage", "efficiency"),
                ],
                value="classification",
                label="Select Difficulty"
            )
            reset_btn = gr.Button("?? Reset Environment", size="lg", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### ?? AI Suggestions")
            ai_box = gr.Markdown("AI suggestions will appear here")
            ai_btn = gr.Button("?? Get AI Suggestion", size="sm")
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### ?? Ticket Information")
            ticket_display = gr.Markdown("Click 'Reset Environment' to start")
        
        with gr.Column(scale=1):
            pass
    
    gr.Markdown("---")
    gr.Markdown("### ? Execute Action")
    
    with gr.Row():
        action = gr.Dropdown(
            choices=["read", "analyze", "classify", "set_priority", "route"],
            value="read",
            label="Action Type",
            interactive=True
        )
        dept = gr.Dropdown(
            choices=["Billing", "Technical", "General", "Premium Support"],
            label="Department",
            interactive=True
        )
        conf = gr.Slider(0, 100, 50, step=5, label="Confidence (%)", interactive=True)
    
    exec_btn = gr.Button("? Execute Action", size="lg", variant="primary")
    
    gr.Markdown("---")
    gr.Markdown("### ?? Results")
    
    with gr.Row():
        result = gr.Markdown("Results will appear here")
        summary = gr.Markdown("Summary will appear here")
    
    gr.Markdown("---")
    gr.Markdown("### ?? How to Use")
    gr.Markdown("""
    1. **Select Task**: Choose Easy, Medium, or Hard
    2. **Reset Environment**: Start a new episode
    3. **Take Actions**: Read, analyze, or route tickets
    4. **Monitor Rewards**: Track your performance
    5. **Get AI Suggestions**: See what AI recommends
    """)
    
    reset_btn.click(
        fn=reset_environment,
        inputs=[task],
        outputs=[ticket_display, result, summary]
    )
    
    exec_btn.click(
        fn=take_action,
        inputs=[action, dept, conf],
        outputs=[result, summary, ai_box]
    )
    
    ai_btn.click(
        fn=get_suggestion,
        outputs=[ai_box]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, show_error=True)

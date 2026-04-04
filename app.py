import gradio as gr

def process_ticket(ticket_text, department):
    return f""? Ticket processed!\nText: {ticket_text}\nDepartment: {department}""

with gr.Blocks(title="OpenEnv Ticket Triage") as demo:
    gr.Markdown("# ?? OpenEnv Ticket Triage")
    ticket_input = gr.Textbox(label="Ticket Text", lines=4)
    dept_dropdown = gr.Dropdown(["Billing", "Technical", "Support"], label="Department")
    output = gr.Textbox(label="Output", lines=6)
    btn = gr.Button("Process Ticket")
    btn.click(process_ticket, [ticket_input, dept_dropdown], output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

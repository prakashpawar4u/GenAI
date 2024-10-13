import gradio as gr

def add_numbers(Num1, Num2):
    return Num1 + Num2

#Define the interface
demo = gr.Interface(fn = add_numbers,
                    inputs = [gr.Number(), gr.Number()],
                    outputs=gr.Number()
                    )
demo.launch(server_name="127.0.0.1", server_port =7860)

import yaml
import argparse
import os 
import gradio as gr

from func import predict, re_generate, remove_last_turn, like, select_model_change

# Argument parser setup
parser = argparse.ArgumentParser(description='Chatbot Interface with Customizable Parameters')
parser.add_argument('--stop-token-ids', type=str, default='', help='Comma-separated stop token IDs')
parser.add_argument("--host", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, default=8101)
parser.add_argument("--default_model", type=str, default="")
parser.add_argument("--default_model_url", type=str, default="")
parser.add_argument("--default_api_key", type=str, default="EMPTY")
# Parse the arguments
args = parser.parse_args()

def load_models_from_yaml(path):
    if os.path.isfile(path) and path.endswith('.yaml'):
        with open(path, 'r') as f:
            models = yaml.safe_load(f)
        # print("models:", models)
        supported_models = {}
        for model in models['models']:
            for key, value in model.items():
                # print("key:", key, "value:", value)
                supported_models[key] = value
        # print("supported_models:", supported_models)
        return supported_models
    elif os.path.isdir(path):
        supported_models = {}
        for file in os.listdir(path):
            if file.endswith('.yaml'):
                with open(os.path.join(path, file), 'r') as f:
                    models = yaml.safe_load(f)
                for model in models['models']:
                    for key, value in model.items():
                        supported_models[key] = value
        return supported_models
    else:
        raise ValueError(f"Invalid path: {path}")
                    

multimodaltextbox = gr.MultimodalTextbox(label="Submit your message here")
with gr.Blocks(fill_height=True, theme=gr.themes.Ocean()) as demo:
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("## Minecraft Chatbot@CraftJarvis")
            supported_models = load_models_from_yaml('configs/api')
            # print("supported_models:", supported_models)
            model_choices = [(key, value) for key, value in supported_models.items()]
            # print("model_choices:", model_choices)
            select_model = gr.Dropdown(choices=model_choices, value="custom", interactive=True, label="Select Model", allow_custom_value=True)
            model_name = gr.Textbox(value=args.default_model, label="Model name")
            model_url = gr.Textbox(value=args.default_model_url, label="Model URL")
            api_key = gr.Textbox(value=args.default_api_key, label="API Key", type="password")

            select_model.change(select_model_change, inputs=[select_model], outputs=[model_name, model_url])

            with gr.Accordion("Parameters", open=True) as parameter_row:
                temp = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, interactive=True, label="Temperature",)
                max_output_tokens = gr.Slider(minimum=0, maximum=8196, value=2048, step=128, interactive=True, label="Max output tokens",)
                # repetition_penalty = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.1, interactive=True, label="Repetition penalty (future)",)
                # gradio checkbox for stream mode or not 
                stream = gr.Checkbox(label="Streaming", value = True)

            system_prompt = gr.Textbox(placeholder="You are a helpful assistant that can answer questions and help with tasks in Minecraft.", label="System Prompt", lines = 4)

        with gr.Column(scale=8):
            chatbot = gr.Chatbot(
                height=650, 
                type="messages", 
                render_markdown=True,
                sanitize_html=False,
                show_copy_button=True,
                show_share_button=True,
                # additional_inputs=[model, model_url, api_key, temp, max_output_tokens, stream],
                # fill_height=True
                allow_tags=["thinking", "think"]
                )
            multimodaltextbox.render()
            # gr.ChatInterface(predict, type="messages", chatbot=chatbot, textbox=multimodaltextbox, multimodal=True, additional_inputs=[model, model_url, api_key, temp, max_output_tokens, stream], fill_height=True)
            multimodaltextbox.submit(predict, [multimodaltextbox, chatbot, model_name, model_url, api_key, system_prompt, temp, max_output_tokens, stream], [multimodaltextbox, chatbot])
            chatbot.like(like)
            # add three buttons in one row 
            with gr.Row():
                # add a button to submit the chat
                # submit = gr.Button(value = "🚀 Submit")
                # submit.click(predict, [multimodaltextbox, chatbot, model, model_url, api_key, temp, max_output_tokens, stream], [multimodaltextbox, chatbot])
                # add a button to re-generate the response
                regenerate = gr.Button(value = "🔄  Regenerate")
                regenerate.click(re_generate, inputs=[multimodaltextbox, chatbot, model_name, model_url, api_key, system_prompt, temp, max_output_tokens, stream], outputs=[multimodaltextbox, chatbot])
                # # add a button to download the conversation
                # download = gr.Button(value = "📥 Download")
                # add a button to remove the last run
                remove_last = gr.Button(value = "🧺  Remove Last Turn")
                remove_last.click(remove_last_turn, inputs=[chatbot], outputs=[chatbot])
                # add a clear button to clear the chatbot
                clear = gr.ClearButton([multimodaltextbox, chatbot], value="🗑  Clear History", )
                clear.click(inputs=[multimodaltextbox, chatbot], outputs=[multimodaltextbox, chatbot])

    with gr.Row():
        with gr.Accordion("Examples", open=False) as conv_examples:
            gr.Examples(examples=[
                {"files":[], "text": "Can diamond be mined with a stone pickaxe in Minecraft?"}, 
                {"files":[], "text": "Give you nothing in the inventory, generate a step-by-step plan to obtain diamonds."},
                {"files":[], "text": "What is the recipe for the enchanting table in Minecraft?"},
                {"files":[], "text": "You are a Minecraft expert. You can finish tasks by strict and correct reasoning. Now you face a task: obtain a diamond pickaxe in Minecraft. Reasoning what should you do first when you have nothing in the inventory."}
            ], inputs=multimodaltextbox, label="Chat")


            gr.Examples(examples=[
                {"files":["data/images/007-dark_forest.png"], "text": "What are the red structures visible in the background?"},
                {"files":["data/images/030-villager.png"], "text": "What is the role of the villager seen in the image?"},
                {"files":["data/images/038-inventory.png"], "text": "What type of armor is the player wearing?"},
            ], inputs=multimodaltextbox, label="Visual Question Answering")

            gr.Examples(examples=[
                {"files":["data/images/004-forest.png"], "text": "Caption the image and answer what biome is this."},
                {"files":["data/images/042-ender_dragen.png"], "text": "Caption this image in details."},
                {"files":["data/images/033-enderman.png"], "text": "Caption this image."},
            ], inputs=multimodaltextbox, label="Visual Captioning")

            gr.Examples(examples=[
                {"files":["data/images/005-diamond_ore.png"], "text": "Pinpoint the diamond ore."},
                {"files":["data/images/009-pig.png"], "text": "Pinpoint the pig."},
                {"files":["data/images/035-cow.png"], "text": "Pinpoint the cow."},
                {"files":["data/images/045-crafting.png"], "text": "Pinpoint the oak_planks."},
                {"files":["data/images/023-chicken.png"], "text": "Pinpoint the chicken."},
                {"files":["data/images/014-inventory.png"], "text": "Pinpoint the apple."},
            ], inputs=multimodaltextbox, label="Visual Point Grounding")

            gr.Examples(examples=[
                {"files":["data/images/022-zombie.png"], "text": "Please provide the bounding box coordinate of the region this sentence describes: zombie."},
                {"files":["data/images/037-cow.png"], "text": "Please provide the bounding box coordinate of the region this sentence describes: cow."},
                {"files":["data/images/031-sheep.png"], "text": "Output the bounding box of the sheep in the image."},
                {"files":["data/images/039-villager.png"], "text": "Output the bounding box of the villager in the image."},
            ], inputs=multimodaltextbox, label="Visual Box Grounding")

            gr.Examples(examples=[
                {"files":["data/images/020-ender_dragen.png"], "text": "Generate the caption with grounding:"},
                {"files":["data/images/028-savannah_biome.png"], "text": "Generate the caption with grounding:"},
                {"files":["data/images/043-layout.png"], "text": "Generate the caption with grounding:"},
                {"files":["data/images/052-inventory.jpg"], "text": "Generate the caption with grounding:"},
            ], inputs=multimodaltextbox, label="Visual Captioning and Box Grounding")

demo.queue().launch(server_name=args.host, server_port=args.port, share=True)
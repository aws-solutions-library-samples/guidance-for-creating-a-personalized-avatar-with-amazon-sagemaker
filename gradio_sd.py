import tarfile
import gradio as gr
import boto3
import random
import time
from PIL import Image
import numpy

# Upload the file
s3_client = boto3.client("s3")

bucket = "sagemaker-us-west-2-376678947624"
prefix = "personalized-avatar-builder"

sample_files = [[["statics/titanic.csv", "statics/std_diff_mme.jpg", "statics/gradio_screenshot.jpg"]]]

# funtion that takes a file_name, a bucket_name, and a prefix and upload the file to the bucket
def upload_file(s3_client, file_name, bucket, prefix):

    object_name = f"{prefix}/{file_name}"

    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except Exception as e:
        print(e)
        return False
    return True

# write a function that takes a list of files and tar them into tar.gz file
# then return the path of the tar.gz file
def tar_n_upload_files(files):
    tarfile_name = "images.tar.gz"
    with tarfile.open(tarfile_name, "w:gz") as tar:
        for file in files:
            tar.add(file.name, arcname=file.name.split("/")[-1])
            print(f"file {file.name} added")

    if upload_file(s3_client, tarfile_name, bucket, prefix):
        return f"{tarfile_name} uploaded to s3://{bucket}/{prefix}/{tarfile_name} \n\n Training kicked off..."
    return "Failed to upload"


with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald")) as demo:
    gr.Markdown("# Personalized Avatar Builder")
    with gr.Tab("Upload Image"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("Upload 5-10 image of yourself")
                input_files = gr.File(file_count="multiple", file_types=["text", ".json", ".csv"])
                input_examples = gr.Examples(sample_files, input_files)
                upload_button = gr.Button("Upload")
            with gr.Column():
                output_text = gr.TextArea("...", interactive=False, show_label=False)
    with gr.Tab("Create Your Avatar"):
        with gr.Row():
            with gr.Column(scale=3):
                models = gr.Dropdown(choices=["base.tar.gz", "pikachu.tar.gz"], type="value",
                                     info="Choose a model", show_label=False)
            with gr.Column(scale=1):
                refresh_models = gr.Button(value="Refresh").style(full_width=True)
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(show_label=False,
                                    info="Prompt:",
                                    placeholder="Enter a prompt for your avatar")
                nprompt = gr.Textbox(show_label=False,
                                     info="Negative prompt:",
                                     placeholder="ugly, tiling, poorly drawn hands, " +
                                                  "poorly drawn feet, out of frame, " +
                                                  "extra limbs, disfigured, deformed, body out of frame, blurry," +
                                                  "blurred, watermark, grainy, signature").style(show_copy_button=True,
                                                                                          container=True)

            with gr.Column(scale=1):
                create = gr.Button(value="Create").style(full_width=True)
        with gr.Row():
            with gr.Column():
                inference_step = gr.Slider(1, 100, value=50, show_label=False, info="Number of inference steps")
                guidance_scale = gr.Slider(0, 20, value=7.5, show_label=False,
                                           info="Guidance scale (choose between 0 and 20)")
                seed = gr.Slider(1, 10000000, value=10, show_label=False, info="Random Seed")
            with gr.Column():
                output_img = gr.Image(label="Output Image", type="pil").style(height=400)

    # refresh the model list from a S3 location
    def refresh_model_list():
        model_list = []
        for i in range(3):
            num = random.randint(0, 10)
            model_list.append(f"model-{num}.tar.gz")
        models = gr.Dropdown.update(choices=model_list)
        return models

    def generate_avatar(model_name, p, np, inf_steps=50, scale=10, s=1):
        time.sleep(5)
        image_array = numpy.random.random((600, 600, 3))*255
        image = Image.fromarray(image_array.astype('uint8'), 'RGB')
        return image

    # events
    upload_button.click(tar_n_upload_files, input_files, output_text)
    refresh_models.click(fn=refresh_model_list, inputs=None, outputs=models, api_name="refresh")
    create.click(generate_avatar, [models, prompt, nprompt, inference_step, guidance_scale, seed], output_img)


if __name__ == "__main__":
    demo.launch(share=True)

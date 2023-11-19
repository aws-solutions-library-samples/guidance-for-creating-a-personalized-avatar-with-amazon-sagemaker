import json
import base64
import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch_neuronx
import triton_python_backend_utils as pb_utils
from PIL import Image
from io import BytesIO
import os

from torch import autocast
from torch.utils.dlpack import to_dlpack, from_dlpack
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

model_id = "stabilityai/stable-diffusion-2-1"

DTYPE = torch.bfloat16

class NeuronTypeConversionWrapper(nn.Module):
    def __init__(self, post_quant_conv):
        super().__init__()
        self.network = post_quant_conv

    def forward(self, x):
        return self.network(x.float())

class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        out_tuple = self.unet(sample, timestep, encoder_hidden_states, return_dict=False)
        return out_tuple

class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.device = unetwrap.unet.device

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None, return_dict=False):
        sample = self.unetwrap(sample, timestep.to(dtype=DTYPE).expand((sample.shape[0],)), encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)

class NeuronTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = text_encoder.dtype
        self.device = text_encoder.device

    def forward(self, emb, attention_mask = None):
        return [self.neuron_text_encoder(emb)['last_hidden_state']]
                
class TritonPythonModel:

    def decode_latents(self, latents):
        # latents = latents.to(torch.float)
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def initialize(self, args):
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(json.loads(args["model_config"]),
                                               "generated_image")["data_type"])
        
        self.model_dir = args['model_repository']
        self.model_ver = args['model_version']
        self.model_nam = args['model_name']
        
        sd_file_dir = f"{self.model_dir}/{self.model_ver}/sd2_compile_dir_512"
                
        print(f"check compiled model files at {sd_file_dir}:\n\n")
        print(os.listdir(sd_file_dir))

        text_encoder_filename = os.path.join(sd_file_dir, 'text_encoder/model.pt')
        decoder_filename = os.path.join(sd_file_dir, 'vae_decoder/model.pt')
        unet_filename = os.path.join(sd_file_dir, 'unet/model.pt')
        post_quant_conv_filename = os.path.join(sd_file_dir, 'vae_post_quant_conv/model.pt')

        
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        # Replaces StableDiffusionPipeline's decode_latents method with our custom decode_latents method defined above.
        StableDiffusionPipeline.decode_latents = self.decode_latents

        # Load the compiled UNet onto two neuron cores.
        self.pipe.unet = NeuronUNet(UNetWrap(self.pipe.unet))
        device_ids = [0, 1]
        self.pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_filename), device_ids, set_dynamic_batching=False)
        
        # Load other compiled models onto a single neuron core.
        self.pipe.text_encoder = NeuronTextEncoder(self.pipe.text_encoder)
        self.pipe.text_encoder.neuron_text_encoder = torch.jit.load(text_encoder_filename)
                
        self.pipe.vae.decoder = NeuronTypeConversionWrapper(torch.jit.load(decoder_filename))
        self.pipe.vae.post_quant_conv = NeuronTypeConversionWrapper(torch.jit.load(post_quant_conv_filename))

        # self.pipe.enable_xformers_memory_efficient_attention()

        
        # This line of code does offload of model parameters to the CPU and only pulls them into the GPU as they are needed
        # Not tested with MME, since it will likely provoke CUDA OOM errors.
        #self.pipe.enable_sequential_cpu_offload()
    def encode_image(self, img): 
        # Convert the image to bytes
        with BytesIO() as output:
            img.save(output, format="JPEG")
            img_bytes = output.getvalue()

        return base64.b64encode(img_bytes).decode()

    def execute(self, requests):
        responses = []
        for request in requests:
            prompt_object = pb_utils.get_input_tensor_by_name(request, "prompt")
            prompt_text = prompt_object.as_numpy()[0][0].decode()
            
            nprompt_object = pb_utils.get_input_tensor_by_name(request, "negative_prompt")
            nprompt_text = nprompt_object.as_numpy()[0][0].decode()
            
            gen_args = pb_utils.get_input_tensor_by_name(request, "gen_args")
            gen_args_decoded = json.loads(gen_args.as_numpy()[0][0].decode())

            generator = [torch.Generator().manual_seed(gen_args_decoded['seed'])]
            
            with torch.no_grad():
                generated_image = self.pipe(
                    prompt = prompt_text,
                    negative_prompt = nprompt_text,
                    num_inference_steps=gen_args_decoded['num_inference_steps'],
                    guidance_scale=gen_args_decoded['guidance_scale'],
                    generator=generator
                ).images[0]
                
            output_img_bytes = self.encode_image(generated_image)
            
            output_image_obj = np.array([output_img_bytes], dtype="object").reshape((-1, 1))
            
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    "generated_image",
                    output_image_obj
                )
            ])
            
            responses.append(inference_response)
        
        return responses
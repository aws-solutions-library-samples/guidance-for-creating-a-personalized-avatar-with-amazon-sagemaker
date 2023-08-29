
import pathlib
import shlex
import subprocess
from typing import Optional, Tuple, Union, List, Dict
from random import randint
import os
from pathlib import Path

# bind to a random port so torch dist doesn't complain when running multiple jobs
os.environ["MASTER_PORT"] = str(randint(10000, 60000))



class Trainer:
    def __init__(self, instance_data_dir:Union[pathlib.Path, str], output_dir:Union[pathlib.Path, str]):
        self.is_running = False
        self.is_running_message = "Another training is in progress."

        self.output_dir = pathlib.Path(output_dir)
        self.instance_data_dir = pathlib.Path(instance_data_dir)

    def run(
        self,
        base_model: str = "stabilityai/stable-diffusion-2-1-base",
        resolution: int = 512,
        n_steps: int = 1000,
        concept_prompt: str = "photo of <<TOK>>",
        learning_rate: float = 1e-4,
        gradient_accumulation: int = 1,
        fp16: bool = True,
        use_8bit_adam: bool = True,
        gradient_checkpointing: bool = True,
        train_text_encoder: bool = True,
        with_prior_preservation: bool = True,
        prior_loss_weight: float = 1.0,
        class_prompt: str = "a photo of person",
        num_class_images: int = 50,
        class_data_dir: Union[pathlib.Path, str] = Path("/tmp/priors"),
        lora_r: int = 128,
        lora_alpha: int = 1,
        lora_bias: str = "none",
        lora_dropout: float = 0.05,
        lora_text_encoder_r: int = 64,
        lora_text_encoder_alpha: int = 1,
        lora_text_encoder_bias: str = "none",
        lora_text_encoder_dropout: float = 0.05,
    ) -> Tuple[Dict, List[pathlib.Path]]:
        
        if not class_data_dir.exists():
            class_data_dir.mkdir(parents=True)
      

        command = f"""
        accelerate launch train_dreambooth.py \
            --pretrained_model_name_or_path={base_model}  \
            --instance_data_dir={self.instance_data_dir} \
            --output_dir={self.output_dir} \
            --train_text_encoder \
            --instance_prompt="{concept_prompt}" \
            --resolution={resolution} \
            --gradient_accumulation_steps={gradient_accumulation} \
            --learning_rate={learning_rate} \
            --max_train_steps={n_steps} \
            --train_batch_size=1 \
            --lr_scheduler=constant \
            --lr_warmup_steps=100 \
            --num_class_images={num_class_images} \
            
        """
        if train_text_encoder:
            command += f" --train_text_encoder"
        if with_prior_preservation:
            command += f""" --with_prior_preservation \
                --prior_loss_weight={prior_loss_weight} \
                --class_prompt="{class_prompt}" \
                --class_data_dir={class_data_dir}
                """

        command += f""" --use_lora \
            --lora_r={lora_r} \
            --lora_alpha={lora_alpha} \
            --lora_bias={lora_bias} \
            --lora_dropout={lora_dropout} 
            """

        if train_text_encoder:
            command += f""" --lora_text_encoder_r={lora_text_encoder_r} \
                --lora_text_encoder_alpha={lora_text_encoder_alpha} \
                --lora_text_encoder_bias={lora_text_encoder_bias} \
                --lora_text_encoder_dropout={lora_text_encoder_dropout}
                """
        if fp16:
            command += " --mixed_precision fp16"
        if use_8bit_adam:
            command += " --use_8bit_adam"
        if gradient_checkpointing:
            command += " --gradient_checkpointing"

        with open(self.output_dir / "train.sh", "w") as f:
            command_s = " ".join(command.split())
            f.write(command_s)

        self.is_running = True
        res = subprocess.run(shlex.split(command))
        self.is_running = False

        if res.returncode == 0:
            result_message = "Training Completed!"
        else:
            result_message = "Training Failed!"

        return result_message
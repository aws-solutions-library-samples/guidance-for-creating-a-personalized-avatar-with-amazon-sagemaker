from trainer import Trainer
import argparse
from io import BytesIO
from pathlib import Path
import os
from utils import preprocess_training_images
import shutil
import tarfile
import uuid
import subprocess

def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument

    parser.add_argument("--input_data", 
                        type=str, 
                        default="/opt/ml/input/data/training", 
                        help="Path to dataset.")

    parser.add_argument("--model_dir", 
                        type=str, 
                        default="/opt/ml/model", 
                        help="Model directory")
    
    parser.add_argument("--resolution", 
                        type=int, 
                        default= 512, 
                        help="Resolution of generated image")
    
    parser.add_argument("--num_steps", 
                        type=int, 
                        default=1000, 
                        help="Number of training iterations")
    
    parser.add_argument("--concept_prompt", 
                        type=str, 
                        default="photo of <<TOK>>", 
                        help="Expanded concept vocabulary")
    
    parser.add_argument("--class_prompt", 
                        type=str, 
                        default="a photo of person", 
                        help="Class of the instance to fine-tune")
    
    parser.add_argument("--lr", 
                        type=float, 
                        default=1e-4, 
                        help="Learning rate to use for training")
    
    parser.add_argument("--grad_accum", 
                        type=int, 
                        default=1, 
                        help="Number of mini-batch to accumlate gradient")
    
    parser.add_argument("--bf16",
                        type=bool,
                        default=True,
                        help="Whether to use bf16")
    
    parser.add_argument("--eight_bit_adam",
                        type=bool,
                        default=True,
                        help="Whether to use 8bit Adam optimization")
    
    parser.add_argument("--gradient_checkpointing",
                        type=bool,
                        default=True,
                        help="Whether to save checkpoints")
    
    parser.add_argument("--train_text_encoder",
                        type=bool,
                        default=True,
                        help="Whether to fine-tune text encoder")
    
    parser.add_argument("--prior_preservation",
                        type=bool,
                        default=True,
                        help="Whether to generate class images")
    
    parser.add_argument("--prior_loss_weight", 
                        type=float, 
                        default=1.0, 
                        help="Prior preservation loss weight")
    
    parser.add_argument("--num_class_images", 
                        type=int, 
                        default=50, 
                        help="Number of prior pres images to generate")
    
    parser.add_argument("--lora_r", 
                        type=int, 
                        default=128, 
                        help="LoRA rank parameter")
        
    parser.add_argument("--lora_alpha", 
                        type=int, 
                        default=1, 
                        help="LoRA alpha parameter")
    
    parser.add_argument("--lora_bias", 
                        type=str, 
                        default="none", 
                        help="LoRA bias parameter")
    
    parser.add_argument("--lora_dropout", 
                        type=float, 
                        default=0.05, 
                        help="LoRA dropout parameter")

    parser.add_argument("--lora_text_encoder_r", 
                        type=int, 
                        default=64, 
                        help="LoRA text encoder rank parameter")
    
    parser.add_argument("--lora_text_encoder_alpha", 
                        type=int, 
                        default=1, 
                        help="LoRA text encoder alpha parameter")

    parser.add_argument("--lora_text_encoder_bias", 
                        type=str, 
                        default="none", 
                        help="LoRA text encoder bias parameter")
    
    parser.add_argument("--lora_text_encoder_dropout", 
                        type=float, 
                        default=0.05, 
                        help="LoRA text encoder dropout parameter")
    
    parser.add_argument("--face_cropping", 
                        type=bool, 
                        default=True, 
                        help="Whether to crop faces")

    args, _ = parser.parse_known_args()

    return args


def main():
    
    args = parse_arge()
    
    print("print all arguments for debugging ===================================")
    
    print(f"input_data: {args.input_data} and data type: {type(args.input_data)}")
    print(f"face_cropping: {args.face_cropping} and data type: {type(args.face_cropping)}")
    print(f"resolution: {args.resolution} and data type: {type(args.resolution)}")
    print(f"num_steps: {args.num_steps} and data type: {type(args.num_steps)}")
    print(f"concept_prompt: {args.concept_prompt} and data type: {type(args.concept_prompt)}")
    print(f"lr: {args.lr} and data type: {type(args.lr)}")
    print(f"grad_accum: {args.grad_accum} and data type: {type(args.grad_accum)}")
    print(f"bf16: {args.bf16} and data type: {type(args.bf16)}")
    print(f"eight_bit_adam: {args.eight_bit_adam} and data type: {type(args.eight_bit_adam)}")
    print(f"gradient_checkpointing: {args.gradient_checkpointing} and data type: {type(args.gradient_checkpointing)}")
    print(f"train_text_encoder: {args.train_text_encoder} and data type: {type(args.train_text_encoder)}")
    print(f"prior_preservation: {args.prior_preservation} and data type: {type(args.prior_preservation)}")
    print(f"prior_loss_weight: {args.prior_loss_weight} and data type: {type(args.prior_loss_weight)}")
    print(f"class_prompt: {args.class_prompt} and data type: {type(args.class_prompt)}")
    print(f"num_class_images: {args.num_class_images} and data type: {type(args.num_class_images)}")
    print(f"lora_r: {args.lora_r} and data type: {type(args.lora_r)}")
    print(f"lora_alpha: {args.lora_alpha} and data type: {type(args.lora_alpha)}")
    print(f"lora_bias: {args.lora_bias} and data type: {type(args.lora_bias)}")
    print(f"lora_dropout: {args.lora_dropout} and data type: {type(args.lora_dropout)}")
    print(f"lora_text_encoder_r: {args.lora_text_encoder_r} and data type: {type(args.lora_text_encoder_r)}")
    print(f"lora_text_encoder_alpha: {args.lora_text_encoder_alpha} and data type: {type(args.lora_text_encoder_alpha)}")
    print(f"lora_text_encoder_bias: {args.lora_text_encoder_bias} and data type: {type(args.lora_text_encoder_bias)}")
    print(f"lora_text_encoder_dropout: {args.lora_text_encoder_dropout} and data type: {type(args.lora_text_encoder_dropout)}")
    
    
    print("end of all arguments for debugging ===================================")
    
    train_path = preprocess_training_images(args.input_data) #, args.face_cropping)
 
    print("list image files for trianing input: \n\n")
    print(os.listdir(str(train_path)))
    
    class_data_dir = Path("/tmp/priors")
    if not class_data_dir.exists():
        class_data_dir.mkdir(parents=True)
    
    model_dir = Path(args.model_dir)
    # prepare the mme model directory
    mme_dir = model_dir / "sd_lora"
    shutil.copytree("sd_lora", mme_dir)
    
    
    output_dir = mme_dir / "1" / "output"
    output_dir.mkdir(exist_ok=True)
    
    print(f"output directory is here: {str(output_dir)}")
    
    trn = Trainer(train_path, output_dir)
    
    status = trn.run(base_model="stabilityai/stable-diffusion-2-1-base",
        resolution=args.resolution,
        n_steps=args.num_steps,
        concept_prompt=args.concept_prompt,
        learning_rate=args.lr,
        gradient_accumulation=args.grad_accum,
        fp16=args.bf16,
        use_8bit_adam=args.eight_bit_adam,
        gradient_checkpointing=args.gradient_checkpointing,
        train_text_encoder=args.train_text_encoder,
        with_prior_preservation=args.prior_preservation,
        prior_loss_weight=args.prior_loss_weight,
        class_prompt=args.class_prompt,
        num_class_images=args.num_class_images,
        class_data_dir=class_data_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_bias=args.lora_bias,
        lora_dropout=args.lora_dropout,
        lora_text_encoder_r=args.lora_text_encoder_r,
        lora_text_encoder_alpha=args.lora_text_encoder_alpha,
        lora_text_encoder_bias=args.lora_text_encoder_bias,
        lora_text_encoder_dropout=args.lora_text_encoder_dropout
    )
    
    print("list file in model directory: \n\n")
    print(os.listdir(str(model_dir)))
    
    print("list file in output directory: \n\n")
    print(os.listdir(str(output_dir)))
    
    print("====training complete =====")

if __name__ == "__main__":
    main()
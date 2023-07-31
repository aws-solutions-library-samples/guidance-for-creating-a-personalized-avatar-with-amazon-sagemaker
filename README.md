## Personalized Avatar Workshop With Generative AI On Amazon SageMaker

Welcome to the "Building Personalized Avatar With Generative AI Using Amazon SageMaker" workshop! This workshop aims to demonstrate how to leverage generative AI models, specifically Stable Diffusion (SD), with Amazon SageMaker. You will learn how to fine-tune a personalized model using your own images and generate avatars based on text prompts. The workshop also showcases cost-saving techniques using [Multi Model Endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html).

By the end of this workshop, you will be able to upload your own images, fine-tune a Stable Diffusion model, and generate personalized avatars based on text prompts. You will gain hands-on experience with SageMaker's model training, inference orchestration, and Multi-Model Endpoints. Additionally, you can apply these techniques to other creative art generation projects.

The entire example takes about 1 hour to complete. At the end, you will use your models to build a simple Gradio application which you can experiment with different prompt and generate avatar images of yourself.

Input Images          |  Personalized Output
:-------------------------:|:-------------------------:
![Inputs](statics/demo_inputs.jpg)  |  ![DEMO OUPUT](statics/avatar.gif)


## Solution Architecture

![solution architecture](statics/solution_architecture.png)

The architecture diagram above outlines the end-to-end solution. This workshop will only focus on the model training and inference portion of this solution. You can use this as a reference and build on top of the examples we provide. 

Steps covered in the workshop:

1. Set Up
2. Prepare Image Data
3. Run LoRA Finetuning
4. Host Multi-Model Endpoints
5. Invoke Model
6. Run The Gradio App
7. Clean Up

## Usage
Make sure that your AWS identity has the requisite permissions which includes ability to create SageMaker Resources (Model, EndpointConfigs, Endpoints, and Training Jobs) in addition to S3 access to upload model artifacts. Alternatively, you can attach the [AmazonSageMakerFullAccess](https://docs.aws.amazon.com/sagemaker/latest/dg/security-iam-awsmanpol.html#security-iam-awsmanpol-AmazonSageMakerFullAccess) managed policy to your IAM User or Role.

Clone this repo into a Jupyter environment and run [personalized_avatar_solution.ipynb](personalized_avatar_solution.ipynb) notebook. It will take you through the each of the step mentioned above.

We recommend to run this workshop on **Data Science 3.0 kernel in SageMaker Studio with a ml.m5.large instance.**

## Additional Modules and Utilities
Additional modules and utilities are provided within subdirectories.

```
|-- models
|   └── model_setup         A Triton Python backend model that prepares the common stable diffusion components on hosting container
|       |-- 1
|       |   └── model.py
|       └── config.pbtxt
|-- src                      Training code directory for the fine tuning job
|   |--launch.py             Entry script for the training job
|   |--requirements.txt      Python modules to extend the container
|   |--trainer.py            LoRA fine tuning script
|   |--train_dreambooth.py   Dreambooth script
|   |--utils.py              Utility functions
    └── sd_lora              A Triton Python backend model template directory for LoRA fine-tuned Stable Diffusion models
        |-- 1
        |   └── model.py
        └── config.pbtxt
```

## Example Inputs

To achieve the best results from fine-tuning Stable Diffusion to generate images of yourself, it is typically need a large quantity and variety of photos of yourself from different angles, with different expressions, and in different backgrounds.

![Input Sample Pictures](statics/input_examples.jpg)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
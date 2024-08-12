---
title: Llama-2-7b-chat-finetune
---
## Model Description

This repository contains the fine-tuned Llama 2 7B chat model, trained on the [mlabonne/guanaco-llama2-1k](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k) dataset using QLoRA. The model is intended for chat-based interactions and exhibits improved performance compared to the base Llama 2 7B chat model.

## Fine-tuning Details

The model was fine-tuned using the following settings:

- **QLoRA Parameters:**
    - Rank (lora_r): 64
    - Alpha (lora_alpha): 16
    - Dropout (lora_dropout): 0.1
- **4-bit Quantization:**
    - Enabled (use_4bit): True
    - Compute dtype (bnb_4bit_compute_dtype): float16
    - Quantization type (bnb_4bit_quant_type): nf4
    - Nested quantization (use_nested_quant): False
- **Training Parameters:**
    - Output directory: ./results
    - Number of epochs (num_train_epochs): 1
    - Batch size per device (per_device_train_batch_size): 4
    - Gradient accumulation steps (gradient_accumulation_steps): 1
    - Gradient checkpointing: True
    - Maximum gradient norm (max_grad_norm): 0.3
    - Learning rate (learning_rate): 2e-4
    - Optimizer: paged_adamw_32bit
    - Learning rate scheduler: cosine
    - Warmup ratio (warmup_ratio): 0.03
    - Group sequences by length (group_by_length): True

## Usage

To use the model, you can load it using the `transformers` library:

## Limitations and Biases

This model is based on the Llama 2 7B chat model and may inherit its limitations and biases. It is important to use the model responsibly and be aware of potential issues such as generating incorrect or misleading information, offensive or discriminatory language, and perpetuating harmful stereotypes.

## Acknowledgements

This project builds upon the work of the Llama 2 team and the contributors to the QLoRA technique. We thank them for their valuable contributions to the field of large language models.

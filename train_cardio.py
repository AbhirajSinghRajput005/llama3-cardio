
# Fine-Tuning Llama-3 for Cardiology FHIR Extraction
# This script includes the 'SafeSFTTrainer' interceptor to bypass the .mean() bug.

from trl import SFTTrainer
from transformers import TrainingArguments


class SafeSFTTrainer(SFTTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        return super().training_step(model, inputs)


trainer = SafeSFTTrainer(
    model = model,
    tokenizer = tokenizer, 
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, 
        warmup_steps = 10,
        num_train_epochs = 3, 
        learning_rate = 5e-5, 
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "cardio_outputs",
        report_to = "wandb", 
    ),
)

trainer.train()

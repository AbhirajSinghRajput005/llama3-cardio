
# Fine-Tuning Llama-3 for Cardiology FHIR Extraction
# This script includes the 'SafeSFTTrainer' interceptor to bypass the .mean() bug.

from trl import SFTTrainer
from transformers import TrainingArguments

class SafeSFTTrainer(SFTTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        return super().training_step(model, inputs)

# Training logic goes here...

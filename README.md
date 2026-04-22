
# Cardiology Entity Extraction to FHIR JSON
Fine-tuning Llama-3-8B to extract clinical entities from cardiology reports.

## 🚀 Model Access
The fine-tuned LoRA adapters are hosted on Hugging Face due to their size:
[**Download Weights here**](https://huggingface.co/abhirajs005/llama3-cardio-fhir-v1)

## 🛠️ How to Use
1. Clone this repo.
2. Install Unsloth.
3. Load the model using:
   `model = FastLanguageModel.from_pretrained("abhirajs005/llama3-cardio-fhir-v1")`

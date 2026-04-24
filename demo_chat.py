from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "abhirajs005/llama3-cardio-fhir-v1", 
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

print("\n" + "="*50)
print("   CARDIOLOGY FHIR EXTRACTOR - LIVE TERMINAL")
print("="*50)
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Patient Note >> ")
    if user_input.lower() in ['exit', 'quit', 'stop']:
        break
        
    prompt = f"Below is a cardiology-specific clinical note. Extract the medical entities and metrics into a strictly valid FHIR-aligned JSON format.\n\n### Clinical Note:\n{user_input}\n\n### Extracted JSON:\n"
    
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.1)
    response = tokenizer.batch_decode(outputs)[0].split("### Extracted JSON:")[1].split("<|end_of_text|>")[0].strip()
    
    print("\n[Extracted JSON]:")
    print(response)
    print("-" * 50)

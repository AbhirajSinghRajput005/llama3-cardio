import openai

judge_prompt = """
Evaluate the following Cardiology FHIR extraction. 
Rate the 'Reasoning Quality' from 1-10 based on:
1. Accuracy of ICD-10 categorization.
2. Correct mapping of dosages.
3. Logical grouping of vitals.

Student Model Output: {model_json}
Reference Data: {reference_json}

Return only a JSON with 'score' and 'explanation'.
"""

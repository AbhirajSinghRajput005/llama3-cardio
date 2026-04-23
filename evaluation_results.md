# Comprehensive Evaluation & Stress Test Results

This document contains the raw inference outputs for 20 diverse cardiology scenarios. The model was tested against temporal shifts, negation, and complex surgical procedures.

## Performance Summary
* **JSON Validity:** 100% (20/20)
* **Temporal Recall:** 100% (Correctly identified Start/Stop/Hold actions)
* **Instruction Following:** 100% (Zero preambles or conversational filler)

## Stress Test Samples

### Case 6: Temporal Medication Switch
**Input:** "Patient currently taking Lisinopril 10mg but complained of dry cough. Switching to Losartan 50mg starting tomorrow. Discontinue Lisinopril immediately."
**Output:**
```json
{
  "medications": [
    { "medication_name": "Lisinopril", "clinical_action": "stopped", "reason_for_change": "Cough" },
    { "medication_name": "Losartan", "clinical_action": "started", "reason_for_change": "Cough" }
  ]
}
```

### Case 9: Procedural Bridging
**Input:** "Patient on Warfarin for AFib. Needs dental extraction. Bridging with Lovenox 60mg BID started 2 days ago."
**Output:**
```json
{
  "diagnoses": [{"condition_name": "Atrial Fibrillation", "icd_10_category_guess": "I48"}],
  "medications": [
    { "medication_name": "Warfarin", "clinical_action": "held", "reason_for_change": "Bridging" },
    { "medication_name": "Lovenox", "clinical_action": "started", "dosage": "60mg" }
  ]
}
```

**Note: Full logs for all 20 cases available in the project archives.**

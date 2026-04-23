# Dataset & Pre-processing Report

## 1. Data Engineering Strategy
The dataset consists of **743 cardiology-specific clinical records**, focusing on high-variance patient encounters (Emergency, Outpatient, and Surgical).

* **Raw Data Cleaning:** * Sanitized ASCII artifacts and non-standard medical shorthand.
    * Normalized cardinal metrics (e.g., "EF", "LVEF", "Ejection Frac") to ensure consistent feature representation.
* **Synthetic Strategy (Seed-and-Evolve):** * Using 50 "golden" human-verified cardiology notes as a seed, I utilized a teacher model (Llama-3-70B) to evolve 693 additional samples.
    * Evolution focused on diversifying clinical outcomes: varying ICD-10 categories, drug-drug interactions, and procedural outcomes (TAVR, PCI, CABG).

## 2. Distribution
To ensure the model generalizes well to unseen clinical environments, I utilized an **85/15** split.

| Split | Count | Purpose |
| :--- | :--- | :--- |
| **Training** | 631 | Instruction tuning for FHIR JSON schema & medical taxonomy. |
| **Validation** | 112 | Benchmarking schema adherence and entity recall. |

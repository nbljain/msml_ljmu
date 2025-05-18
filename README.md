# Flan-T5 Fine-Tuning and PEFT with LoRA

This project demonstrates fine-tuning and parameter-efficient fine-tuning (PEFT) using LoRA on the Flan-T5 model for text summarization, specifically on the CNN/DailyMail dataset.

## Features

- Fine-tune Flan-T5 on summarization tasks.
- Apply LoRA (Low-Rank Adaptation) for efficient PEFT.
- Evaluate model performance using ROUGE, BLEU, BERTScore, and FactScore.
- Compare original and fine-tuned model results.

## Files

- `flan_t5_peft_lora.py`: Main script for PEFT with LoRA.
- `msml_1_finetune_flant5_small_sumx_50k_data.py`: Script for standard fine-tuning.

## Example ROUGE Comparison

| Metric    | Original Model | FineTuned Model |
|-----------|---------------|----------------|
| rouge1    | 0.2345        | 0.2699         |
| rouge2    | 0.0857        | 0.0922         |
| rougeL    | 0.1787        | 0.1984         |
| rougeLsum | 0.2103        | 0.2376         |

## Usage

1. Install dependencies:
    ```bash
    pip install torch transformers datasets peft sentence-transformers bert-score rouge-score nltk
    ```

2. Run the fine-tuning or PEFT script:
    ```bash
    python msml_1_finetune_flant5_small_sumx_50k_data.py
    # or
    python flan_t5_peft_lora.py
    ```

3. Evaluate results using the generated CSV files.

## Notes

- Model checkpoints and results are saved in the specified output directories.
- For Google Colab usage, adjust file paths as needed.

---

**Author:** Nitin Jain
**License:** MIT

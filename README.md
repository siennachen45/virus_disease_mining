# Virus-Disease Association Mining

This project extracts virus-disease evidence from PubTator3/PubMed abstracts and uses a fine-tuned PubMedBERT model to predict likely associations.

The workflow starts from PubTator3 BioC XML abstracts, disease annotations, and species annotations. Sentences containing both viral species and disease mentions are extracted as candidate evidence.

`01Preprocess.ipynb` downloads/prepares PubTator3 data and extracts candidate virus-disease sentences.

`02finetune_bert_save.py` fine-tunes PubMedBERT/BiomedBERT for virus-disease association classification.

`03predict.py` applies the fine-tuned model to candidate sentences and outputs prediction probabilities.

`04filter_result.ipynb` filters abstract predictions by `virus_taxid` and `mesh_id` and prepares grouped evidence for review.

`05filter_result_onFULL.ipynb` applies the same `virus_taxid` / `mesh_id` filtering logic to full-text predictions.

`06ai_predict_Round2.py` uses an external LLM to re-check grouped virus-disease evidence pairs.

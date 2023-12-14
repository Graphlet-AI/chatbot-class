# All vs Paraphrase Mini-LM Model Comparisons

This experiment compares multiple methods of sentence encoding on people's names - including across character sets - using the following models:

* [`sentence-transformers/all-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
* [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)

## Notes

Compared to the names, JSON tends to compress scores together owing to overlapping text in formatting: field names, quotes and brackets. You can see in the name pairs name length is a source of error. The dates behave well in the JSON records.

# Amazon Fine-Food Reviews Sentiment Suite

Ever wondered what hundreds of thousands of Amazon food reviews really say? This repo walks through five progressively stronger NLP setups, from good-old TF-IDF to cutting-edge BERT with LoRA. To spot whether a review is positive or negative using sentiment analysis.

## Dataset  

Amazon Fine Food Reviews (≈ 568 k rows, 10+ years of data).
Label rule: `Score>3 = positive`, `Score≤3 = negative`.

## Contents  

`data/`
- `dataset.md` - Source and description for the Amazon Fine-Food dataset.

`notebooks/`   
- `notebook.md` - Link to the Colab notebook that runs every experiment.

## Top Results  

| Model / Approach                 | Text Representation | Accuracy    |
| -------------------------------- | ------------------- | ----------- |
| BERT (zero-shot, pipeline)       | Transformer         | **88.24 %** |
| TF-IDF + Random Forest           | Sparse Vectors      | 87.32 %     |
| BERT (fine-tuned, 3 epochs)      | Transformer         | 87.17 %     |
| TF-IDF + Logistic Regression     | Sparse Vectors      | 85.69 %     |
| BERT + LoRA (2 epochs)           | Transformer + LoRA  | 85.71 %     |
| TF-IDF + Multinomial Naive Bayes | Sparse Vectors      | 84.10 %     |
| Word2Vec + Random Forest         | Dense Embeddings    | 83.27 %     |
| Word2Vec + Logistic Regression   | Dense Embeddings    | 78.63 %     |
| Word2Vec + Linear SVC            | Dense Embeddings    | 78.62 %     |

## Takeaways    

- Pre-trained BERT already beats most classical baselines even without a single gradient step.
- Fine-tuning (standard or LoRA) moves BERT closer to the TF-IDF + Random Forest champ, but the zero-shot model still outperforms them all.
- TF-IDF continues to shine for linear / tree models – cheap, fast, surprisingly strong.
- Word2Vec lags here; dense averages lose nuance compared to sparse TF-IDF or contextual BERT.
- LoRA gives ~BERT-base quality with a fraction of trainable parameters, ideal when GPU time is precious.  


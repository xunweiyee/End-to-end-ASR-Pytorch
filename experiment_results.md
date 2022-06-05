# Experiment Results

## TedSrt

> 200 MB of audio data was scraped from [Ted2Srt](https://ted2srt.org/). We tried training on 1 GB of dataset but the results were roughly similar.

Baseline Model

| Extractor | Classifier | CTC-WER | CTC-Loss |
|-----------|------------|---------|----------|
| CNN       | RNN        | 0.92    | 6.56     |

Comparing performance of Extractor

| Extractor | Classifier | CTC-WER | CTC-Loss |
|-----------|------------|---------|----------|
| MLP       | RNN        | 0.99    | 6.58     |
| ANN       | RNN        | 1       | 6.92     |
| RNN       | RNN        | 0.95    | 6.53     |


Comparing performance of Classifier

| Extractor | Classifier | CTC-WER | CTC-Loss |
|-----------|------------|---------|----------|
| CNN       | MLP        | 0.90    | 5.75     | 
| CNN       | CNN        | 0.95    | 6.75     |
| CNN       | ANN        | 0.99    | 6.98     |

---

## LibriSpeech

> This is a public dataset by [OpenSLR](https://www.openslr.org/12/). The repository is already designed to accommodate this dataset.

Baseline Model

| Extractor | Classifier | CTC-WER | CTC-Loss |
|-----------|------------|---------|----------|
| CNN       | RNN        | 0.21926 | 1.32976  |

Comparing performance of Extractor

| Extractor | Classifier | CTC-WER | CTC-Loss |
|-----------|------------|---------|----------|
| MLP       | RNN        | 0.99    | 6.63263  |
| ANN       | RNN        | 0.9954  | 6.87     |
| RNN       | RNN        | 0.263   | 1.62     |


Comparing performance of Classifier

| Extractor | Classifier | CTC-WER | CTC-Loss |
|-----------|------------|---------|----------|
| CNN       | MLP        | 0.5884  | 4.092    |
| CNN       | CNN        | 1       | 7.031    |
| CNN       | ANN        | 0.9982  | 7.065    |

# Experiment Results

### TedSrt

> Data scraped from [Ted2Srt](https://ted2srt.org/). We tried training on 1Gb of dataset but the result is almost similar.

Comparing performance of Extractor

| Extractor | Classifier | CTC-WER | CTC-Loss |
|-----------|------------|---------|----------|
| MLP       | RNN        | 0.99    | 6.58     |
| CNN       | RNN        | 0.92    | 6.56     |
| ANN       | RNN        | 1       | 6.92     |
| RNN       | RNN        | 0.95    | 6.53     |


Comparing performance of Classifier

| Extractor | Classifier | CTC-WER | CTC-Loss |
|-----------|------------|---------|----------|
| CNN       | MLP        | 0.90    |5.746     | 
| CNN       | CNN        | 0.95    |6.745     |
| CNN       | ANN        | 0.99    |6.98      |

---

### LibriSpeech

> Public dataset by [OpenSLR](https://www.openslr.org/12/). We train models on it and compare the results.

Comparing performance of Extractor

| Extractor | Classifier | CTC-WER | CTC-Loss |
|-----------|------------|---------|----------|
| MLP       | RNN        | 0.99    | 6.63263  |
| CNN       | RNN        | 0.21926 | 1.32976  |
| ANN       | RNN        | 0.9954  | 6.87     |
| RNN       | RNN        | 0.263   | 1.62     |


Comparing performance of Classifier

| Extractor | Classifier | CTC-WER | CTC-Loss |
|-----------|------------|---------|----------|
| CNN       | MLP        | 0.5884  | 4.092    |
| CNN       | CNN        | 1       | 7.031    |
| CNN       | ANN        | 0.9982  | 7.065    |

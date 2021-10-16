Investigating performance of Feature Extractor

| Extractor | Classifier |CTC-WER|ATT-WER|CTC-Loss
|---------|-------------|-------|-----|---|
| MLP    | RNN|(step = 208.0K, wer = 0.99)| //|6.63263 
| CNN  | RNN|(step = 605.0K, wer = 0.13)|(step = 565.0K, wer = 0.16)|//
| ANN  | RNN| | |
| RNN   | RNN |


Investigating performance of Classifier

| Extractor | Classifier |CTC-WER|ATT-WER|CTC-Loss
|---------|-------------|-------|-----|---|
| CNN    | MLP|(step = 488.0K, wer = 0.13)| //|3.30768| 
| CNN  | CNN|
| CNN  | ANN|

//Traning time too long, to be calculated on our own data set



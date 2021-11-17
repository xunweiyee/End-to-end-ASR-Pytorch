Investigating performance of Feature Extractor

TedSrt =====================================================================


| Extractor | Classifier |CTC-WER|ATT-WER|CTC-Loss
|---------|-------------|-------|-----|---|
| MLP    | RNN|0.27276| |0.46318 |
| CNN  | RNN|0.26054|//|0.00942|
| ANN  | RNN| | |
| RNN   | RNN |


Investigating performance of Classifier

| Extractor | Classifier |CTC-WER|ATT-WER|CTC-Loss
|---------|-------------|-------|-----|---|
| CNN    | MLP|0.26984 |//| 2.61415  | 
| CNN  | CNN|
| CNN  | ANN|

LibriSpeech ===================================================================


| Extractor | Classifier |CTC-WER|CTC-Loss
|---------|-------------|-------|---|
| MLP    | RNN|0.99|6.63263 
| CNN  | RNN|0.2226|1.336
| ANN  | RNN| | |
| RNN   | RNN | 0.263| 1.62 |


Investigating performance of Classifier

| Extractor | Classifier |CTC-WER|CTC-Loss
|---------|-------------|------------|---|
| CNN    | MLP|(step = 488.0K, wer = 0.13) (retraining with reduced libri)|3.30768| 
| CNN  | CNN|1|7.031
| CNN  | ANN|

//Traning time too long, to be calculated on our own data set
1. Metrics are saved in two dictionaries: Solver.eval_stats and Solver.train_stats
2. Solver.print_msg is called during every validation, change "valid_step" parmas in the config file to change the printing frequency
3. ctc_weight is set to 1.0 to omit the training of attention networks, to speed up training process
4. Training Command: ```python main.py --config config/libri/asr_example.yaml --njobs 8``` 


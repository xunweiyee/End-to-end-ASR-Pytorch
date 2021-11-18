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
| CNN  | RNN|0.21926|1.32976
| ANN  | RNN|0.9954 |6.87 |
| RNN   | RNN | 0.263| 1.62 |


Investigating performance of Classifier

| Extractor | Classifier |CTC-WER|CTC-Loss
|---------|-------------|------------|---|
| CNN    | MLP|0.5884|4.092| 
| CNN  | CNN|1|7.031|
| CNN  | ANN|0.9982|7.065

//Traning time too long, to be calculated on our own data set
1. Metrics are saved in two dictionaries: Solver.eval_stats and Solver.train_stats
2. Solver.print_msg is called during every validation, change "valid_step" parmas in the config file to change the printing frequency
3. ctc_weight is set to 1.0 to omit the training of attention networks, to speed up training process
4. Training Command: ```python main.py --config config/libri/asr_example.yaml --njobs 8``` 


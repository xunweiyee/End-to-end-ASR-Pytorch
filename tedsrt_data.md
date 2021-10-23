# Using TedSrt data

Run `python scraper/preprocess.py` to scrape and generate data. Raw data saved at `scraper/`, processed data at `data/`.

To train using TedSrt `python main.py --config config/ted/asr_example.yaml --njobs 8`.
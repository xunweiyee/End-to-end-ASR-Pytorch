# Using TedSrt data

Run `python scraper/preprocess.py` to scrape and generate data. Raw data saved at `scraper/`, processed data at `data/`.

To train using TedSrt `python main.py --config config/ted/asr_example.yaml --njobs 8`.

![Run preprocess.py](assets/ss-preprocessing-data.png)

141 urls scraped /200

using 23 scraped data /140
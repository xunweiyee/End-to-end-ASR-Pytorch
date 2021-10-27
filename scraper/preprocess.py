import os
from joblib import Parallel, delayed
from os.path import join
from pathlib import Path
import random
import shutil

import scraper
import re
import subprocess
import num2words
import pydub
from pydub import AudioSegment
from tqdm import tqdm
import logging

logger = logging.getLogger()
# logging.basicConfig(level="INFO", format="%(levelname)s: %(filename)s: %(message)s")
# logging.basicConfig(level='WARNING', format="%(levelname)s: %(filename)s: %(message)s")
# logger.setLevel(logging.WARNING)


random.seed(42)
AUDIO_EXTENSION = 'mp3'
READ_FILE_THREADS = 4
corpus_path = '../data/TedSrt'
txt_norm = 'tedsrt-norm.txt'
txt_vocab = 'tedsrt-vocab.txt'
processed_data_path = join(corpus_path, 'processed_data') # store all data in this folder before splitting
txt_norm_path = join(corpus_path, txt_norm)
txt_vocab_path = join(corpus_path, txt_vocab)

src_path = 'data/' # raw data source
ratio=[.6, .2, .2] # train test split

# filter audio and srt
duration_diff = 7 # seconds
repeated_occurrence = 50



def clean_text(text):
    '''
    Text processing to clean text before saving as label
    to lowercase, convert years to words, convert digits to words, remove symbols
    '''
    text = text.lower().strip('\n')
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join([num2words.num2words(i, to='year') if (i.isdigit() & (len(i) == 4)) else i for i in text.split()]) # year to words
    text = ' '.join([num2words.num2words(i) if i.isdigit() else i for i in text.split()]) # num to words
    text = re.sub(' +', ' ', text) # remove redundant spaces
    text = text.replace('-', ' ')
    return text

def normalize_text(text):
    '''
    Text processing to normalize text for language model training
    should transform word numeral to numeric, normalize date formats, lemmatization; most not implemented currently
    '''
    text = text.lower().strip('\n')
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(' +', ' ', text) # remove redundant spaces
    text = text.replace('-', ' ')
    return text

def to_ms(string):
    '''
    Convert string '00:00:00,000' to milliseconds
    to be used for audio slicing
    '''
    string = string.replace(',','')
    hour, minute, second = string.split(':')
    second = int(second)
    second += int(hour) * 3600 * 1000
    second += int(minute) * 60 * 1000
    second = second
    return second

def txt_to_trans(txt_file, file_name):
    '''
    Convert txt file to transcript format ready to be read into Dataset
    lines formatted as 'filename-idx text_label'
    return lines and time_slices
    '''
    file = open(txt_file, 'r')
    lines = file.readlines()
    file.close()
    
    transcript = [] # label for audio
    txt_src = [] # label for language model
    time_slices = []

    for i in range(len(lines)):
        idx = re.search('^[\d]+$', lines[i].strip('\ufeff'))
        if idx:
            idx = idx[0]
            time_frame = re.findall('[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3}', lines[i+1])
            if time_frame:
                start, end = to_ms(time_frame[0]), to_ms(time_frame[1])
                time_slices.append((idx, (start, end)))

                audio_label = lines[i+2]
                audio_label = clean_text(audio_label)
                new_line = f"{file_name}-{idx} {audio_label}"
                transcript.append(new_line)
                
                lm_label = normalize_text(audio_label)
                txt_src.append(lm_label)
                
    return transcript, time_slices, txt_src

def save_txt(txt, output_path):
    '''
    Save transcript to output_path
    '''
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, 'a+') as f:
        for line in txt:
            f.write(f"{line}\n")
        f.close()

def check_ms_accuracy(time_slices, occurrence_threshold=20):
    '''
    Check whether srt is accurate, remove those not accurate to milliseconds, check repeated occurrence of 820
    '''
    for time_slice in time_slices:
        occurrence_820 = sum(1 for elem in time_slices if elem[1][0] % 1000 == 820)
    if occurrence_820 > repeated_occurrence:
        return occurrence_820
    return 0

def check_intro_timing(audio_duration, audio_srt_duration):
    '''
    Check whether srt matches audio, remove those without taking water drop intro into account
    '''
    duration_diff = audio_duration - audio_srt_duration
    if duration_diff > duration_diff:
        return duration_diff
    return 0

def build_dataset(src_path=src_path):
    '''
    Build dataset from raw data scraper/data/
    split audio into slices
    convert srt to trans.txt ready for training
    '''
    # logger.disabled = True
    shutil.rmtree(corpus_path, ignore_errors=True)
    folders = os.listdir(src_path)[:10]
    for idx, curr_folder in enumerate(tqdm(folders, desc="Building dataset from scraped data")):
        # logging.info('\n')
        file_name = str(idx) #save the transcript as num, can be changed to folder name
        output_path = join(processed_data_path, file_name)
        txt_output_path = join(output_path, file_name + '.trans.txt')
                
        # logging.info(f"{idx}. Creating transcript for {curr_folder}...")
        txt_path = list(Path(join(src_path, curr_folder)).rglob('*.txt'))[0]
        transcript, time_slices, txt_src = txt_to_trans(txt_path, file_name)
        
        # logging.info(f"{idx}. Slicing audio for {curr_folder}...")
        audio_path = list(Path(join(src_path, curr_folder)).rglob('*.' + AUDIO_EXTENSION))[0]
        audio_file = AudioSegment.from_file(audio_path, AUDIO_EXTENSION)
        
        # check timing accuracy by milliseconds
        ms_not_accurate = check_ms_accuracy(time_slices)
        if ms_not_accurate:
            # logging.warning(f"{idx}. Srt not accurate with {ms_not_accurate} 820s. Deleting entry {curr_folder}")
            shutil.rmtree(output_path, ignore_errors=True)
            continue
        
        # check timing of audio intro
        audio_duration = audio_file.duration_seconds
        audio_srt_duration = time_slices[-1][-1][-1] / 1000
        intro_not_matched = check_intro_timing(audio_duration, audio_srt_duration)
        if intro_not_matched:
            # logging.warning(f"{idx}. Srt not matching with time slices. Deleting entry {curr_folder}")
            shutil.rmtree(output_path, ignore_errors=True)
            continue
        
        # writing output
        save_txt(transcript, txt_output_path)
        save_txt(txt_src, txt_norm_path)
        for idx, time_slice in time_slices:
            audio_slice = audio_file[time_slice[0]:time_slice[1]]
            audio_output_path = join(output_path, f"{file_name}-{idx}.{AUDIO_EXTENSION}")
            audio_slice.export(audio_output_path, format=AUDIO_EXTENSION)
        
        tqdm.write(f'Successfully created {idx} {curr_folder}')
        # print(f'Successfully created {idx} {curr_folder}')

def move_folders(folders, source, destination):
    '''
    Move a list of folders from source to destination
    '''
    logger.info(f'{destination.split("/")[-1]}: {folders}')
    for folder in folders:
        shutil.move(join(source, folder), join(destination, folder))

def split_dataset(ratio=[.6, .2, .2]):
    '''
    Split dataset to train dev test according to ratio
    '''
    folders = os.listdir(processed_data_path)
    folders.sort()
    random.shuffle(folders)

    split_1 = int(ratio[0] * len(folders))
    split_2 = int((ratio[0] + ratio[1]) * len(folders))

    move_folders(folders[:split_1], processed_data_path, join(corpus_path, 'train'))
    move_folders(folders[split_1:split_2], processed_data_path, join(corpus_path, 'dev'))
    move_folders(folders[split_2:], processed_data_path, join(corpus_path, 'test'))
    shutil.rmtree(processed_data_path, ignore_errors=True)



def main():
    print('Scraping data...')
    scraper.main(number_of_talks=200, starting_video_id=100)
    print()
    print('Start preprocessing...')
    build_dataset(src_path)
    split_dataset(ratio=ratio)


if __name__ == "__main__":
    main()
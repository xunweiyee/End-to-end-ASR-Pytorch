#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path

text_paths = list(Path('../data/TedSrt').rglob("*.trans.txt"))
# print(text_paths)

for text_path in text_paths:
    file = open(text_path, "r")
    to_remove_list = ['laughter', 'applause']

    for line in file:
        is_in = [ele for ele in to_remove_list if(ele in line)]
        if bool(is_in):
            # print(line)
            audio_file = line.split(' ')[0] + '.mp3'
            directory = os.path.dirname(text_path)
            audio_path = os.path.join(directory, audio_file)
            try:
                os.remove(audio_path)
                print('removing', audio_path)
            except:
                continue
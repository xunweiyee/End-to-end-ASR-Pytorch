#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path



def main(src_path='../data/TedSrt'):
    text_paths = list(Path(src_path).rglob("*.trans.txt"))

    for text_path in text_paths:
        file = open(text_path, "r")
        to_remove_list = ['laughter', 'applause', 'music']

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


if __name__ == "__main__":
    
    main()
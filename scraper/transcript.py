import re

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

def extract_time_slices(txt_file):
    '''
    Extract time slices from raw text file
    '''
    file = open(txt_file, 'r')
    lines = file.readlines()
    file.close()
    
    time_slices = []

    for i in range(len(lines)):
        idx = re.search('^[\d]+$', lines[i].strip('\ufeff'))
        if idx:
            idx = idx[0]
            time_frame = re.findall('[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3}', lines[i+1])
            if time_frame:
                start, end = to_ms(time_frame[0]), to_ms(time_frame[1])
                time_slices.append((idx, (start, end)))
               
    return time_slices

def check_ms_accuracy(time_slices, occurrence_threshold=20):
    '''
    Check whether srt is accurate, remove those not accurate to milliseconds, check repeated occurrence of 820
    '''
    for time_slice in time_slices:
        occurrence_820 = sum(1 for elem in time_slices if elem[1][0] % 1000 == 820)
    if occurrence_820 > occurrence_threshold:
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

def check_srt_accuracy(srt_path, duration_diff=7, occurrence_threshold=50):
    time_slices = extract_time_slices(srt_path)

    # check timing accuracy by milliseconds
    ms_not_accurate = check_ms_accuracy(time_slices, duration_diff)
    if ms_not_accurate:
        # logging.warning(f"{idx}. Srt not accurate with {ms_not_accurate} 820s. Deleting entry {curr_folder}")
        return False
    
    # # check timing of audio intro
    # audio_duration = audio_file.duration_seconds
    # audio_srt_duration = time_slices[-1][-1][-1] / 1000
    # intro_not_matched = check_intro_timing(audio_duration, audio_srt_duration, occurrence_threshold)
    # if intro_not_matched:
    #     # logging.warning(f"{idx}. Srt not matching with time slices. Deleting entry {curr_folder}")
    #     return False
    return True



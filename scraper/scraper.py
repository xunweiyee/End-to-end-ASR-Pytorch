"""
References:
- https://wamiller0783.github.io/TED-scraping-post.html
- https://github.com/The-Gupta/TED-Scraper/blob/master/Scraper.ipynb
"""

from bs4 import BeautifulSoup
from moviepy import editor
import requests
from urllib import request, error

from ast import literal_eval
import json
import logging
import os
import re
import shutil

# Logger setup
logger = logging.getLogger()
logging.basicConfig(level="INFO", format="%(levelname)s: %(filename)s: %(message)s")

# Storage locations
DATA_STORAGE_ROOT = os.path.join(os.getcwd(), "data")

# Scraper constants
TED_URL_HOMEPAGE = "https://www.ted.com"
TED_URL_PREFIX = "https://www.ted.com/talks/"
TED_POPULAR_PAGE_URL = "https://www.ted.com/talks?sort=popular&page="

TED_SRT_HOMEPAGE = "https://ted2srt.org/"
TED_SRT_TALKS = "talks/"
TED_SRT_TALKPAGE = TED_SRT_HOMEPAGE + TED_SRT_TALKS


def main(number_of_talks=200, starting_video_id=100):
    if not os.path.isdir(DATA_STORAGE_ROOT):
        try:
            os.makedirs(DATA_STORAGE_ROOT)
        except FileNotFoundError:
            logging.error("Unable to create data storage folder.")
            raise Exception("Ensure that the current working directory is set to the project root.")

    # Uncomment to parse transcripts and audio from the TEDTalks website
    # logging.info("Parsing audio files with transcripts from TEDTalks website...")
    # video_urls = get_all_video_urls(TED_POPULAR_PAGE_URL)
    # for url in video_urls:
    #     scrape_data_from_url(url)

    # Uncomment to parse SRT transcripts and audio from the TED2SRT website

    _number_of_talks = number_of_talks  # Number of talks to attempt to scrape from
    _starting_video_id = starting_video_id  # Start retrieving from this video id

    logging.info("Parsing audio files with SRT transcripts from TED2SRT website...")
    # video_urls = get_all_srt_video_urls(TED_SRT_HOMEPAGE)
    video_urls = generate_all_srt_video_urls(TED_SRT_TALKPAGE, _number_of_talks, starting_video_id=_starting_video_id)
    for url in video_urls:
        scrape_data_from_srt_url(url)


## Code that handles scraping from the TED2SRT website


def generate_all_srt_video_urls(base_url, number_of_urls, starting_video_id=1):
    """
    Generates a list of urls to attempt to scrape from. This is designed to work specifically for the TED2SRT page.
    This method will not check whether the links are valid. It will only generate the required links.

    This method works because the TED2SRT page also stores videos mapped to their id. For example, visiting the webpage
    "homepage/talks/1" returns the 1st video in id.
    :param base_url: The base url of the TED2SRT talks page.
    :param number_of_urls: The number of urls to generate.
    :param starting_video_id: The starting video id, which defaults to the 1st video.
    :return: Returns a list of urls.
    """
    logging.info(f"Generating {number_of_urls} starting from video id {starting_video_id}...")
    logging.warning("Many of these links might not work. Consider changing the value of _number_of_talks in line 52.")
    return [base_url + str(video_id) for video_id in range(starting_video_id, starting_video_id + number_of_urls)]


# TED2SRT website constants
SUB_URL_KEY = "slug"


def get_all_srt_video_urls(base_url):
    """
    Gets all video urls, given the base url page. This is designed to work specifically for the TED2SRT page.
    :param base_url: The base url page (e.g. front page, 'all videos' page) to start searching from.
    :return: Returns a list of all video urls found.
    """
    response = requests.get(base_url)
    page_soup = BeautifulSoup(response.text, "html.parser")

    all_talks_object = page_soup.select("script")[0].string
    all_talks_list = literal_eval(all_talks_object.split(" = ")[1])

    all_sub_urls = [talk_dict[SUB_URL_KEY] for talk_dict in all_talks_list]

    logging.info(f"Found {len(all_sub_urls)} URLs. Depending on website format, this may only be the first screen.")

    urls = [TED_SRT_HOMEPAGE + TED_SRT_TALKS + sub_url for sub_url in all_sub_urls]
    logging.debug(f"URLs found are: {urls}")
    return urls


# TED2SRT specific page constants
VIDEO_NAME_KEY = "slug"

VIDEO_DOWNLOAD_SUB_URL_KEY = "mediaSlug"
START_OF_AUDIO_DOWNLOAD_URL = "https://download.ted.com/talks/"
END_OF_AUDIO_DOWNLOAD_URL = "-320k.mp4"

VIDEO_SRT_ID_SUB_URL_KEY = "id"
START_OF_SRT_TRANSCRIPT_URL = "https://ted2srt.org/api/talks/"
END_OF_SRT_TRANSCRIPT_URL = "/transcripts/download/srt?lang=en"


def scrape_data_from_srt_url(url):
    """
    Attempts to scrape the audio file and SRT transcript from the given webpage.
    Specifically written to scrape data from the TED2SRT page (root: https://ted2srt.org/).

    If either the audio file or the SRT transcript does not exist, then this method will not scrape anything.
    Otherwise, the data will be saved to a folder marked by the url.
    :param url: The url to scrape data from.
    :return: No return value.
    """
    logger.info(f"\n\nScraping from url: <{url}>")
    response = requests.get(url)
    page_soup = BeautifulSoup(response.text, "html.parser")

    # Identify the metadata object and parse the key phrases needed for video and SRT download
    talk_metadata_script = page_soup.select("script")
    if not talk_metadata_script:
        logging.warning(f"The url {url} is invalid or has no metadata object. Continuing...")
        return

    talk_metadata_string = talk_metadata_script[0].string
    talk_metadata_object = literal_eval(talk_metadata_string.split(" = ")[1])

    srt_download_keyword = str(talk_metadata_object[VIDEO_SRT_ID_SUB_URL_KEY])
    video_download_keyword = str(talk_metadata_object[VIDEO_DOWNLOAD_SUB_URL_KEY])
    name_of_video = str(talk_metadata_object[VIDEO_NAME_KEY])

    srt_download_url = START_OF_SRT_TRANSCRIPT_URL + srt_download_keyword + END_OF_SRT_TRANSCRIPT_URL
    video_download_url = START_OF_AUDIO_DOWNLOAD_URL + video_download_keyword + END_OF_AUDIO_DOWNLOAD_URL

    saved_folder_path = os.path.join(DATA_STORAGE_ROOT, name_of_video)

    success_status = download_and_save_srt_transcript_text(srt_download_url, saved_folder_path)
    if not success_status:
        return

    download_and_save_video_file(video_download_url, saved_folder_path)


# Video constants
VIDEO_FILE_EXTENSION = "video.mp4"
AUDIO_FILE_EXTENSION = "audio.mp3"
SRT_TRANSCRIPT_FILE_EXTENSION = "srt_transcript.txt"


def download_and_save_video_file(video_download_url, path_to_save_to):
    """
    Downloads a video file, given its url link. Converts the video file to audio. Saves the file to the required folder.
    Specifically designed for the TED2SRT website. Note that no audio download exists. Therefore, the video must be
    downloaded first, and then converted into audio. To save space, the video file is then deleted.
    :param video_download_url: The url to download the video from.
    :param path_to_save_to: The folder name to save this file to.
    :return: Returns True if the download was successful, and False otherwise.
    """
    if not os.path.isdir(path_to_save_to):
        os.mkdir(path_to_save_to)

    video_file_saved_location = os.path.join(path_to_save_to, VIDEO_FILE_EXTENSION)
    try:
        _, _ = request.urlretrieve(video_download_url, filename=video_file_saved_location)
        logger.info("Video file saved.")
    except error.HTTPError:
        logger.warning("No video file of lowest quality found. Deleting this specific data folder...")
        srt_transcript_file_saved_location = os.path.join(path_to_save_to, SRT_TRANSCRIPT_FILE_EXTENSION)

        if os.path.isfile(srt_transcript_file_saved_location):
            shutil.rmtree(path_to_save_to)

        if os.path.isdir(path_to_save_to):
            os.rmdir(path_to_save_to)

        return False

    logger.info("Converting video.mp4 file to audio.mp3 file...")
    audio_file_saved_location = os.path.join(path_to_save_to, AUDIO_FILE_EXTENSION)

    video = editor.VideoFileClip(video_file_saved_location)
    audio = video.audio
    audio.write_audiofile(audio_file_saved_location, verbose=False, logger=None)
    video.close()

    os.remove(video_file_saved_location)
    logging.info("Audio file saved. Video file deleted.")
    return True


def download_and_save_srt_transcript_text(srt_transcript_url, path_to_save_to):
    """
    Saves the transcript text to the required folder as a file. This is specifically designed to work with the TED2SRT
    website.
    :param srt_transcript_url: The url to download the srt transcript from.
    :param path_to_save_to: The folder name to save this file to.
    :return: Returns True if the download was successful, and False otherwise.
    """
    if not os.path.isdir(path_to_save_to):
        os.mkdir(path_to_save_to)

    srt_transcript_file_saved_location = os.path.join(path_to_save_to, SRT_TRANSCRIPT_FILE_EXTENSION)
    try:
        _, _ = request.urlretrieve(srt_transcript_url, filename=srt_transcript_file_saved_location)
        logger.info("SRT transcript saved.")
        return True
    except error.HTTPError:
        logger.warning("No SRT transcript found. Deleting this specific data folder...")
        audio_file_saved_location = os.path.join(path_to_save_to, VIDEO_FILE_EXTENSION)

        if os.path.isfile(audio_file_saved_location):
            shutil.rmtree(path_to_save_to)

        if os.path.isdir(path_to_save_to):
            os.rmdir(path_to_save_to)

        return False


## Code that handles scraping from the TEDTalks website


def get_all_video_urls(base_url):
    """
    Gets all video urls, given the base url page. It is assumed that the base url page is in page format (see comments).
    :param base_url: The base url page (e.g. front page, 'all videos' page) to start searching from.
    :return: Returns a list of all video urls found.
    """
    urls = []
    page_number = 0

    while True:
        # i.e. search from "page=1", "page=2", "page=3"...
        page_number += 1

        response = requests.get(base_url + str(page_number))

        page_soup = BeautifulSoup(response.text, "html.parser")
        video_elements = page_soup.select("div.container.results div.col")

        # If no more video urls exist, we have reached the end of the pages
        if len(video_elements) == 0:
            break

        for element in video_elements:
            url_object = element.select("div.media__image a.ga-link")
            url = TED_URL_HOMEPAGE + url_object[0].get("href")
            urls.append(url)
            logger.debug(f"Url found: {url}")

    return urls


def scrape_data_from_url(url):
    """
    Attempts to scrape the audio file and transcript text from the given webpage.
    Specifically written to scrape data from the TEDTalks page (root: https://www.ted.com/talks).

    If either the audio file or the transcript does not exist, then this method will not scrape anything.
    Otherwise, the data will be saved to a folder marked by the url.
    :param url: The url to scrape data from.
    :return: No return value.
    """
    assert url.startswith(TED_URL_PREFIX), f"Url provided does not start with expected url: <{url}>"
    logger.info(f"\n\nScraping from url: <{url}>")

    saved_folder_name = url.replace(TED_URL_PREFIX, "")
    saved_folder_path = os.path.join(DATA_STORAGE_ROOT, saved_folder_name)

    # Check if transcript exists, by attempting to access the page tab
    url = url + "/transcript"
    response = requests.get(url)
    if not response.ok:
        logger.warning(f"No transcript exists: <{url}>")
        return

    page_text = response.text
    transcript_text = get_page_transcript_text(page_text)
    if not transcript_text:
        return

    # Check if audio download exists
    audio_download_link = get_page_audio_download_link(page_text)
    if not audio_download_link:
        return

    # Save to file
    save_transcript_text(transcript_text, saved_folder_path)
    download_and_save_audio_file(audio_download_link, saved_folder_path)


# Audio constants
# Notice that the opening and closing parentheses are escaped
START_OF_METADATA = '<script data-spec="q">q\("talkPage.init",'
END_OF_METADATA = '\)</script>'
# Allow "." to match newlines using re.DOTALL
METADATA_REGEX_PATTERN = re.compile(f"{START_OF_METADATA}(.*?){END_OF_METADATA}", re.DOTALL)
METADATA_PATH_TO_AUDIO = ["__INITIAL_DATA__", "talks", 0, "downloads", "audioDownload"]
AUDIO_FILE_NAME = "audio.mp3"


def get_page_audio_download_link(page_text):
    """
    Obtains the audio data download link of a page, given its page text.
    Specifically written to scrape data from the TEDTalks page (root: https://www.ted.com/talks).

    Workflow:
    - Isolate and extract the audio download link from the page text using tag delimiters

    :param page_text: The raw page text from the website, as parsed by the requests module.
    :return: Returns the audio download link, or None if no link exists.
    """
    if not page_text:
        logger.error("No page text was passed to audio download link parser.")
        return None

    raw_metadata = METADATA_REGEX_PATTERN.search(page_text)
    if not raw_metadata:
        logger.warning("No metadata object was found.")
        return None

    json_metadata = json.loads(raw_metadata.group(1))

    # Locate the audio download link by traversing down the JSON object
    audio_metadata_url = json_metadata
    for key in METADATA_PATH_TO_AUDIO:
        try:
            # Note: This deliberately does not distinguish between accessing between:
            #           - a list, using an integer (e.g. 0)
            #           - a dict, using a key (e.g. "some_name").
            # This is because the metadata object combines both nested lists and nested dicts.
            audio_metadata_url = audio_metadata_url[key]
        except (KeyError, IndexError):
            logger.warning("The specified path to the audio download does not exist.")
            return None

    if not audio_metadata_url:
        logger.warning("No audio download link exists.")
        return None

    return audio_metadata_url


def download_and_save_audio_file(audio_download_url, path_to_save_to):
    """
    Downloads an audio file, given its url link. Saves the file to the required folder.
    :param audio_download_url: The url to download the audio from.
    :param path_to_save_to: The folder name to save this file to.
    :return: No return value.
    """
    if not os.path.isdir(path_to_save_to):
        os.mkdir(path_to_save_to)

    audio_file_saved_location = os.path.join(path_to_save_to, AUDIO_FILE_NAME)
    _, _ = request.urlretrieve(audio_download_url, filename=audio_file_saved_location)
    logger.info("Audio file saved.")


# Transcript constants
START_OF_TRANSCRIPT = "<!-- Transcript text -->"
END_OF_TRANSCRIPT = "<!-- /Transcript text -->"
# Allow "." to match newlines using re.DOTALL
TRANSCRIPT_REGEX_PATTERN = re.compile(f"{START_OF_TRANSCRIPT}(.*?){END_OF_TRANSCRIPT}", re.DOTALL)
TRANSCRIPT_TEXT_PARAGRAPH_TAG = "p"
TRANSCRIPT_FILE_NAME = "transcript.txt"


def get_page_transcript_text(page_text):
    """
    Obtains the speech transcript of a page, given its page text.
    Specifically written to scrape data from the TEDTalks page (root: https://www.ted.com/talks).

    Workflow:
    - Isolate and extract the transcript from the page text using the front and back comment delimiters.
    - Use BeautifulSoup to tidy up the transcript by removing unnecessary HTML tags.

    :param page_text: The raw page text from the website, as parsed by the requests module.
    :return: Returns the transcript text, or None if the text does not exist.
    """
    if not page_text:
        logger.error("No page text was passed to transcript parser.")
        return None

    raw_transcript = TRANSCRIPT_REGEX_PATTERN.search(page_text)
    if not raw_transcript:
        logger.warning("No transcript was found.")
        return None

    page_soup = BeautifulSoup(raw_transcript.group(1), "html.parser")

    # Extract transcript elements using the 'p' tag
    transcript_paragraphs = page_soup.find_all(TRANSCRIPT_TEXT_PARAGRAPH_TAG)

    # Basic data cleaning
    transcript = []
    for paragraph in transcript_paragraphs:
        paragraph = str(paragraph)

        # Remove the 'p' tags at the start and end
        paragraph = paragraph.replace(f"<{TRANSCRIPT_TEXT_PARAGRAPH_TAG}>", "")
        paragraph = paragraph.replace(f"</{TRANSCRIPT_TEXT_PARAGRAPH_TAG}>", "")

        # Remove excess spaces
        paragraph = re.sub("\\s+", " ", paragraph).strip()

        transcript.append(paragraph)

    return " ".join(transcript)


def save_transcript_text(transcript_text, path_to_save_to):
    """
    Saves the transcript text to the required folder as a file.
    :param transcript_text: The text to write to file.
    :param path_to_save_to: The folder name to save this file to.
    :return: No return value.
    """
    if not os.path.isdir(path_to_save_to):
        os.mkdir(path_to_save_to)

    transcript_file_saved_location = os.path.join(path_to_save_to, TRANSCRIPT_FILE_NAME)
    with open(transcript_file_saved_location, "w") as file:
        file.writelines(transcript_text)
        logger.info("Transcript text saved.")


if __name__ == "__main__":
    main()

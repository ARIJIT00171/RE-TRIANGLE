"""
Code to get the data in the DiDeMo video dataset using the videos stored on AWS.

Usage:

python download_videos_AWS.py  --download --video_directory DIRECTORY

will download videos from flickr to DIRECTORY

python download_videos_AWS.py  
"""

import requests
import tqdm
import argparse
import os
import json


def read_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data


parser = argparse.ArgumentParser()
parser.add_argument("--video_directory", type=str, default='videos/',
                    help="Indicate where you want downloaded videos to be stored")
parser.add_argument("--download", dest="download", action="store_true")
parser.set_defaults(download=False)
args = parser.parse_args()
if args.download:
    assert os.path.exists(args.video_directory)

multimedia_template = 'https://multimedia-commons.s3-us-west-2.amazonaws.com/data/videos/mp4/%s/%s/%s.mp4'

splits = ['test', 'val']
data_template = 'data/%s_data.json'
caps = [] 
for split in splits:
     caps.extend(read_json(data_template.format(split)))
videos = set([cap['video'] for cap in caps])


def read_hash(hash_file):
    lines = open(hash_file).readlines()
    yfcc100m_hash = {}
    for line_count, line in tqdm.tqdm(enumerate(lines)):
        line = line.strip().split('\t')
        yfcc100m_hash[line[0]] = line[1]
    return yfcc100m_hash


def get_aws_link(h):
    return multimedia_template % (h[:3], h[3:6], h)


yfcc100m_hash = read_hash('data/yfcc100m_hash.txt')

missing_videos = []

for video_count, video in enumerate(videos):
    video_id = video.split('_')[1]
    link = get_aws_link(yfcc100m_hash[video_id])
    if args.download:
        try:
            r = requests.get(link, allow_redirects=True)
            final_url = r.url

            with requests.get(final_url, stream=True) as resp:
                resp.raise_for_status()
                output_path = f"/content/drive/MyDrive/MyIR2/DiDeMo/videos/{video}.mp4"

                with open(output_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        except Exception:
            print(f"Could not find link for {link}")
    else:
        try:
            r = requests.get(link, allow_redirects=True)
        except Exception:
            missing_videos.append(video)
            print(f"Could not find link for {link}")

if len(missing_videos) > 0:
    write_txt = open('missing_videos.txt', 'w')
    for video in missing_videos:
        write_txt.writelines('%s\n' %video)
    write_txt.close()

from opustools import OpusRead
import os


output_dir = 'open_subtitles_en_pl'
os.makedirs(output_dir, exist_ok=True)


reader = OpusRead(
    source='en',
    target='pl',
    directory='OpenSubtitles',
    release='v2018',
    download_dir=output_dir,
    preprocess='xml',
    write_mode='moses',
    suppress_prompts=True
)

reader.write_files(output_dir)

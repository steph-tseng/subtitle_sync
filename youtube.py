# %%
dict = {'1': 'hi', '2': 'bye'}

print(dict.keys())

# %%
from pytube import YouTube
import os
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

if __package__ is None:                  
    DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(DIR.parent))
    __package__ = DIR.name

from .audio_converter import *

# Dictionary of links to Youtube videos to be downloaded
links = {'1MythicalKitchen': 'https://www.youtube.com/watch?v=OfrMqoyOJVE', 
         '2GameTheorists': 'https://www.youtube.com/watch?v=Ow3bMlScrzs&t=93s', 
         '3LastWeekTonight': 'https://www.youtube.com/watch?v=abuQk-6M5R4&t=215s', 
         '4LateShow': 'https://www.youtube.com/watch?v=kwcy6nLaguY'}

DATA_DIR = 'DATA'
if os.path.isdir(DATA_DIR) != True:
  os.mkdir(DATA_DIR) # Create the DATA folder if there isn't one already
for link in links.keys():
  yt = YouTube(link) # Link to the Youtube video
  t = yt.streams.filter(only_audio=True) # Filter for just audio, remove .filter for whole video
  t[0].download('DATA')

  print('Caption options: ', yt.captions) # To get the language codes
  try:
    if yt.captions['en']:
      caption = yt.captions['en']
    elif yt.captions['a.en']:
      caption = yt.captions['a.en']
    else:
      caption = yt.captions['en.ehkg1hFWq8A']
  except:
    print('Please check caption options above')

  f = open(f"{os.path.join(DATA_DIR, yt.title)}.srt", "w")
  f.write(caption.generate_srt_captions())

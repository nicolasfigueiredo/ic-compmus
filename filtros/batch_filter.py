import os
import librosa
from my_filters import *

audiofiles = [x for x in os.listdir() if (x.endswith('.wav') or x.endswith('.mp3') or x.endswith('.flac') or x.endswith('.aiff')) or x.endswith('.aac')]

for file in audiofiles:
  y, sr = librosa.load(file, sr=44100)
  y_filtered = my_filter(y, sr)
  path = file + 'filt_exp1.wav'

  librosa.output.write_wav(path, y_filtered, sr)
  print('Filtered ' + file)
import os
import librosa
import pr_util as util

from my_filters import *

data_dirs = ['/home/nicolas/documentos/compmus/birds/db-tcc-felipe/']

for data_dir in data_dirs:
  for subdir, dirs, files in os.walk(data_dir):
    for file in files:
      if util.is_audio(file) and file.count('filtered') == 0:
        file_dir = subdir + '/' + file
        print(file_dir)
        y, sr = librosa.load(file_dir, sr=44100)
        y_filtered = my_filter(y, sr)
        path = file_dir + '.filtered.wav'
        print("arquivo filtrado: {}".format(path))
        librosa.output.write_wav(path, y_filtered, sr)
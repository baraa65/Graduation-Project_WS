import numpy as np
import subprocess as sp
import os, shutil

folder = 'audio'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

command = ['ffmpeg.exe',
    "-i", r"rtsp://192.168.1.109:8080/h264_ulaw.sdp",
    '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
    "-f", "segment", '-segment_time', '3', 'audio/out%03d.wav']

pipe = sp.Popen(command, stdout=sp.PIPE)

n = 44100 * 3  # # of samples (sampling rate * duration)
nbytes = n * 2 * 2  # (#samples * #ch  * 2 bytes/sample)

while True:
    raw_audio = np.frombuffer(pipe.stdout.read(nbytes), shape=(n, 2), dtype=np.int16)

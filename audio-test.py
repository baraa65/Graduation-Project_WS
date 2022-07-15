import ffmpeg
import numpy as np
import subprocess as sp

# packet_size = 4096
# process = ffmpeg.input('rtsp://192.168.1.109:8080/h264_ulaw.sdp')
# process = ffmpeg.input('rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4')

# print(process.audio)
# file = process.output("test.png", vframes=1)
# print(file)
# testfile = file.run(capture_stdout=True, capture_stderr=True)
# print(testfile)
# process = process.output('-', format='mulaw')
# process = process.run_async(pipe_stdout=True)

# print(process)
# packet = process.stdout.read(packet_size)
#
# print(packet)
# while process.poll() is None:
#     packet = process.stdout.read(packet_size)
#     print(packet)


command = ['ffmpeg.exe',
    "-i", r"rtsp://192.168.1.109:8080/h264_ulaw.sdp",
    '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
    "-f", "segment", '-segment_time', '3', 'audio/out%03d.wav']

pipe = sp.Popen(command, stdout=sp.PIPE)

n = 44100 * 3  # # of samples (sampling rate * duration)
nbytes = n * 2 * 2  # (#samples * #ch  * 2 bytes/sample)

raw_audio = np.frombuffer(pipe.stdout.read(nbytes), shape=(n, 2), dtype=np.int16)


while True:
    # raw_audio = np.frombuffer(pipe.stdout.read(nbytes), shape=(n, 2), dtype=np.int16)
    print('----test------')
    # print(raw_audio)
    print('----test------')

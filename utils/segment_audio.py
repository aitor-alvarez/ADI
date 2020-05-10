from pydub import AudioSegment
import uuid
import os
import random


def segment(t, audio_file, name):
    cante = AudioSegment.from_wav(audio_file)
    #seg = cante[t[0][0]*1000:t[-1][-1]*1000]
    try:
        seg = cante[t[0][0] * 1000:t[-1][-1] * 1000]
        #seg = cante[t[0][0] * 1000:(t[0][0] * 1000)+3000]
        if seg.duration_seconds>=0.3:
            seg.set_channels(1)
            seg.export(name + '_' + str(uuid.uuid4()) + '.wav', format="wav", bitrate="192k")
    except:
        print("NO")


def save_sample_audio(dir_in, dir_out):
    for d in os.listdir(dir_in):
        if '.wav' in d:
            sound = AudioSegment.from_file(dir_in+d)
            dur = sound.duration_seconds
            begin = random.uniform(0.24, dur - 0.45)
            length = random.uniform(0.24, 0.35)
            sound = sound[begin * 1000:(begin * 1000) + (length * 1000)]
            sound.set_channels(1)
            sound.export(dir_out + d, format="wav", bitrate="192k")




def split_audio(audio_file, audio_dur, dirout, name):
        sound=AudioSegment.from_file(audio_file)
        sound = sound[:audio_dur*1000]
        sound.export(dirout+name, format="wav", bitrate="192k")

from pydub import AudioSegment
import uuid

def segment(t, audio_file, name):
    cante = AudioSegment.from_wav(audio_file)
    #seg = cante[t[0][0]*1000:t[-1][-1]*1000]
    try:
        seg = cante[t[0][0] * 1000:t[-1][-1] * 1000]
        #seg = cante[t[0][0] * 1000:(t[0][0] * 1000)+3000]
        if seg.duration_seconds>=0.5:
            seg.set_channels(1)
            seg.export(name + '_' + str(uuid.uuid4()) + '.wav', format="wav", bitrate="192k")
    except:
        print("NO")




def split_audio(audio_file, audio_dur, dirout, name):
        sound=AudioSegment.from_file(audio_file)
        sound = sound[:audio_dur*1000]
        sound.export(dirout+name, format="wav", bitrate="192k")

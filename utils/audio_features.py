from __future__ import print_function
import vamp
import librosa
import numpy as np
from utils.contour_approximation import contour_approximation
from utils.curve_fitting import curve_fitting
from utils.gapbide import Gapbide
from utils.segment_audio import segment
from utils.pattern_utils import create_dictionary, find_sublist
import os


def run_main(pat=False):
    path = './data/train/MSA/'
    out_path = './patterns_train/MSA/'
    audio_list = []
    if pat==False:
        dictionary = create_dictionary(out_path)
    for root, dirs, files in os.walk(path):
        for aud in files:
            if aud.endswith('.wav'):
                audio_file = os.path.join(root, aud)
                audio_list.append(audio_file)
                audio, sr = librosa.load(audio_file, sr=44100, mono=True)
                data = vamp.collect(audio, sr, "mtg-melodia:melodia", parameters={"voicing": 0.1})
                hop, melody = data['vector']
                melody=1200*np.log2(melody/55.0)
                #melody = melody[~np.isnan(melody)]
                timestamps = 8 * 128/44100.0 + np.arange(len(melody)) * (128/44100.0)
                new_melody, time = curve_fitting(melody, timestamps, 60)
                phrases=[]
                phrases_time=[]
                k=0
                for m in range(0, len(new_melody)-1):
                    if k ==0:
                        phrase=[]
                        tim=[]
                    if np.isnan(new_melody[m]):
                        continue
                    elif not np.isnan(new_melody[m+1]):
                        phrase.append(new_melody[m])
                        tim.append(time[m])
                        k+=1
                    else:
                        phrase.append(new_melody[m])
                        tim.append(time[m])
                        k=0
                        phrases.append(phrase)
                        phrases_time.append(tim)


            #contour phrases
            cont=[]
            ts=[]
            for i in range(0, len(phrases)):
                c, t = contour_approximation(phrases[i], phrases_time[i])
                cont.append(c)
                ts.append(t)

            if pat==True:
                pattern_length=5
                patterns_file = aud.replace('.wav', '')
                patterns_file = os.path.join(root, aud.replace('.wav', '')).replace('/data/train/', '/patterns_train/')
                patt = Gapbide(cont, 3, 0, 0, pattern_length, patterns_file)
                patt.run()
            else:
                for d in dictionary:
                    for i, c in enumerate(cont):
                        if len(c)<5 or len(d)>len(c):
                            continue
                        else:
                            ids = find_sublist(d, c)
                        if ids:
                            segment(ts[i][ids[0][0]:ids[0][1]], audio_file, out_path+aud.replace('.wav', '')+'_'+('').join(d))
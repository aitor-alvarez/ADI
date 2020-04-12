import urllib
import os


filesdir='./data/'

for line in open('./data/audio_list'):
    dir = line.rsplit('/', 2)[1]
    name = line.rsplit('/', 2)[2].replace('\n', '')

    dirout = os.path.join(filesdir, dir)
    fileout= os.path.join(dirout, name)

    if not os.path.isdir(dirout):
        os.mkdir(dirout)
    if not os.path.isfile(fileout):
        urllib.request.urlretrieve(line, fileout)
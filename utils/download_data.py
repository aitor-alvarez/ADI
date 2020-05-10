import urllib
import os


filesdir='./data/test_MGB3/'

for line in open('./data/test_data_MGB3'):
    url='http://crowdsource.cloudapp.net/MGB3_ADI_testset/wav/'
    dir = line.rsplit(' ')[1]
    name = line.rsplit(' ')[0]

    dirout = os.path.join(filesdir, dir)
    fileout= os.path.join(dirout, name+'.wav')

    if not os.path.isdir(dirout):
        os.mkdir(dirout)
    if not os.path.isfile(fileout):
        urllib.request.urlretrieve(url+name+'.wav', fileout)
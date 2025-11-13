"""
The data is provided by 
https://physionet.org/physiobank/database/html/mitdbdir/mitdbdir.htm

The recordings were digitized at 360 samples per second per channel with 11-bit resolution over a 10 mV range.
Two or more cardiologists independently annotated each record; disagreements were resolved to obtain the computer-readable
reference annotations for each beat (approximately 110,000 annotations in all) included with the database.

    Code		Description
    N		Normal beat (displayed as . by the PhysioBank ATM, LightWAVE, pschart, and psfd)
    L		Left bundle branch block beat
    R		Right bundle branch block beat
    B		Bundle branch block beat (unspecified)
    A		Atrial premature beat
    a		Aberrated atrial premature beat
    J		Nodal (junctional) premature beat
    S		Supraventricular premature or ectopic beat (atrial or nodal)
    V		Premature ventricular contraction
    r		R-on-T premature ventricular contraction
    F		Fusion of ventricular and normal beat
    e		Atrial escape beat
    j		Nodal (junctional) escape beat
    n		Supraventricular escape beat (atrial or nodal)
    E		Ventricular escape beat
    /		Paced beat
    f		Fusion of paced and normal beat
    Q		Unclassifiable beat
    ?		Beat not classified during learning
"""

from __future__ import division, print_function
import os
from tqdm import tqdm
import numpy as np
import random
import h5py
from utils import *
from config import get_config

def preprocess( split ):
    nums = ['cu01','cu02','cu03','cu04','cu05','cu06','cu07','cu08','cu09','cu10',
        'cu11','cu12','cu13','cu14','cu15','cu16','cu17','cu18','cu19','cu20',
        'cu21','cu22','cu23','cu24','cu25','cu26','cu27','cu28','cu29','cu30',
        'cu31','cu32','cu33','cu34','cu35']
    features = ['MLII', 'V1'] 

    if split :
        testset = ['cu03','cu07','cu12','cu18','cu22','cu28','cu34']
        trainset = [x for x in nums if x not in testset]

    def dataSaver(dataSet, datasetname, labelsname):
        classes = ['N','V']
        Nclass = len(classes)
        datadict, datalabel= dict(), dict()

        for feature in features:
            datadict[feature] = list()
            datalabel[feature] = list()

        def dataprocess():
          input_size = config.input_size 
          for num in tqdm(dataSet):
            from wfdb import rdrecord, rdann
            record = rdrecord('dataset/'+ num, smooth_frames= True)
            from sklearn import preprocessing
            signals0 = preprocessing.scale(np.nan_to_num(record.p_signal[:,0])).tolist()
            signals1 = preprocessing.scale(np.nan_to_num(record.p_signal[:,1])).tolist()
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(signals0, distance=150)

            feature0, feature1 = record.sig_name[0], record.sig_name[1]

            global lppened0, lappend1, dappend0, dappend1 
            lappend0 = datalabel[feature0].append
            lappend1 = datalabel[feature1].append
            dappend0 = datadict[feature0].append
            dappend1 = datadict[feature1].append
            # skip a first peak to have enough range of the sample 
            for peak in tqdm(peaks[1:-1]):
              start, end =  peak-input_size//2 , peak+input_size//2
              ann = rdann('dataset/'+ num, extension='atr', sampfrom = start, sampto = end, return_label_elements=['symbol'])
              
              def to_dict(chosenSym):
                y = [0]*Nclass
                y[classes.index(chosenSym)] = 1
                lappend0(y)
                lappend1(y)
                dappend0(signals0[start:end])
                dappend1(signals1[start:end])

              annSymbol = ann.symbol
              # remove some of "N" which breaks the balance of dataset 
              if len(annSymbol) == 1 and (annSymbol[0] in classes) and (annSymbol[0] != "N" or np.random.random()<0.15):
                to_dict(annSymbol[0])
        print("processing data...")
        dataprocess()
        noises = add_noise(config)
        for feature in ["MLII", "V1"]: 
            d = np.array(datadict[feature])
            if len(d) > 15*10**3:
                n = np.array(noises["trainset"])
            else:
                n = np.array(noises["testset"]) 
            datadict[feature]=np.concatenate((d,n))
            size, _  = n.shape 
            l = np.array(datalabel[feature])
            noise_label = [0]*Nclass
            noise_label[-1] = 1
            
            noise_label = np.array([noise_label] * size) 
            datalabel[feature] = np.concatenate((l, noise_label))

        with h5py.File(datasetname, 'w') as f:
            for key, data in datadict.items():
                f.create_dataset(key, data=data)
        with h5py.File(labelsname, 'w') as f:
            for key, data in datalabel.items():
                f.create_dataset(key, data=data)        

    if split:
        dataSaver(trainset, 'dataset/train.keras', 'dataset/trainlabel.keras')
        dataSaver(testset, 'dataset/test.keras', 'dataset/testlabel.keras')
    else:
        dataSaver(nums, 'dataset/targetdata.keras', 'dataset/labeldata.keras')

def main(config):
    def Downloadmitdb():
        ext = ['dat', 'hea', 'atr']
        nums = ['cu01','cu02','cu03','cu04','cu05','cu06','cu07','cu08','cu09','cu10',
        'cu11','cu12','cu13','cu14','cu15','cu16','cu17','cu18','cu19','cu20',
        'cu21','cu22','cu23','cu24','cu25','cu26','cu27','cu28','cu29','cu30',
        'cu31','cu32','cu33','cu34','cu35']
        for num in tqdm(nums):
            for e in ext:
                url = f"https://physionet.org/files/cudb/1.0.0/{num}/{num}.{e}"
                mkdir_recursive('dataset')
                cmd = "cd dataset && curl -O "+url
                os.system(cmd)

    if config.downloading:
        Downloadmitdb()
        #print("do not download")
    return preprocess(config.split)

if __name__=="__main__":
    config = get_config()
    main(config)

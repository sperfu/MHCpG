import sys
#  python ideeps.py --train=True --data_file=datasets/Train_all --model_dir=model
import os
import numpy
import pdb
import pickle
from keras.models import Sequential, model_from_config,Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Merge, merge
#from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU,LeakyReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras.layers import normalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import LSTM, Bidirectional 
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution2D, MaxPooling2D,Convolution1D, MaxPooling1D
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.constraints import maxnorm
from keras.models import load_model
#from seya.layers.recurrent import Bidirectional
from sklearn import svm, grid_search
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib 
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import random
import gzip
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from scipy import sparse
import pdb
from math import  sqrt
from sklearn.metrics import roc_curve, auc
import theano
import subprocess as sp
import scipy.stats as stats
#from seq_motifs import *
#import structure_motifs
from keras import backend as K
from rnashape_structure import run_rnashape
from attention import Attention
import keras.layers.core as core
import pyBigWig
import argparse

def calculate_performace(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1               
            
    acc = float(tp + tn)/test_num
    precision = float(tp)/(tp+ fp)
    sensitivity = float(tp)/ (tp+fn)
    specificity = float(tn)/(tn + fp)
    MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return acc, precision, sensitivity, specificity, MCC 

def transfer_label_from_prob(proba):
    label = [1 if val>=0.5 else 0 for val in list(proba)]
    return label

def merge_seperate_network(X_train1, X_train2, Y_train):
    left_hid = 128
    right_hid = 64
    left = get_rnn_fea(X_train1, sec_num_hidden = left_hid)
    right = get_rnn_fea(X_train2, sec_num_hidden = right_hid)
    
    model = Sequential()
    model.add(Merge([left, right], mode='concat'))
    total_hid = left_hid + right_hid
    
    model.add(Dense(total_hid, 2))
    model.add(Dropout(0.3))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd) #'rmsprop')
    
    model.fit([X_train1, X_train2], Y_train, batch_size=100, nb_epoch=100, verbose=0)
    
    return model

def get_4mer_freq(seq_file,trids):
    seq_list = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
                if len(seq):
                    mer_4_array = get_4_nucleotide_composition(trids,seq)
                    seq_list.append(mer_4_array)
                seq = ''
            else:
                seq = seq + line[:-1]
        if len(seq):
            mer_4_array = get_4_nucleotide_composition(trids,seq)
            seq_list.append(mer_4_array)

    return np.array(seq_list)

def read_seq(seq_file,inbw,inbw_histone):
    seq_list = []
    histone_list = []
    label_list = []
    seq = ''
    bw = pyBigWig.open(inbw)
    bw_histone = pyBigWig.open(inbw_histone)
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
		chrom_num,start_tmp,end_tmp,other = name.split(':')
                chrom_num = chrom_num.split('v')[-1]
                start_num = int(start_tmp)-1-500
                end_num = int(end_tmp)+500
		try:
		    bw_values_new = bw.values(chrom_num,start_num,end_num)
		    bw_values_new = np.array([item if item >= 0 else 0 for item in bw_values_new]).reshape(len(bw_values_new),1)
		    bw_histone_values_new = bw_histone.values(chrom_num,start_num,end_num)
		    bw_histone_values_new = np.array([item if item >= 0 else 0 for item in bw_histone_values_new]).reshape(len(bw_histone_values_new),1)
		    #pdb.set_trace()
		except:
		    #pdb.set_trace()
		    bw_values_new = np.zeros((end_num-start_num,1))
		    bw_values_new = bw_values_new.reshape(len(bw_values_new),1)
		    bw_histone_values_new = np.zeros((end_num-start_num,1))
		    bw_histone_values_new = bw_histone_values_new.reshape(len(bw_histone_values_new),1)
                if len(seq):
                    seq_array = get_RNA_seq_concolutional_array(seq)
		    if inbw:
		        seq_array = np.concatenate((seq_array,bw_values_old,bw_histone_values_old),axis=1)
                    seq_list.append(seq_array)                    
		    histone_list.append(bw_values_old)
		    label_list.append(int(other[-1]))
                seq = ''
            else:
                seq = seq + line[:-1]
		bw_values_old = bw_values_new
		bw_histone_values_old = bw_histone_values_new
        if len(seq):
            seq_array = get_RNA_seq_concolutional_array(seq)
	    if inbw:
	        seq_array = np.concatenate((seq_array,bw_values_old,bw_histone_values_old),axis=1)
            seq_list.append(seq_array) 
	    histone_list.append(bw_values_old)
	    label_list.append(int(other[-1]))
    #pdb.set_trace()
    #np.savetxt('histone_list.txt',np.concatenate((np.array(histone_list).reshape(len(histone_list),histone_list[0].shape[0]),np.array(label_list).reshape(len(label_list),1)),axis=1),fmt='%.1f',delimiter=',')
    return np.array(seq_list)

def read_seq_dict(seq_file, trids_3, trids_4,trids_6,nn_dict_3,nn_dict_4,nn_dict_34,nn_dict_36,nn_dict_346,inbw):
    seq_list = []
    label_list = []
    bw_value = []
    bw_value_list = []
    seq = ''
    path = 'datasets/'
    #bed_file = open(path+'sequence.bed','w')
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
                #posi_label = name.split(';')[-1]
                label = name.split(' ')[-1]
                label_list.append(int(label))
		##  add bed file
		chrom_num,start_tmp,end_tmp,_ = name.split(':')
		chrom_num = chrom_num.split('v')[-1]
		start_num = int(start_tmp)-1-500
		end_num = int(end_tmp)+500
		for inbw_file in inbw:
		    bw = pyBigWig.open(inbw_file)
		    try:
			if bw.stats(chrom_num,start_num,end_num)[0] == None:
			    #pdb.set_trace()
			    bw_value.append(0)
			else:
		            bw_value.append(bw.stats(chrom_num,start_num,end_num)[0])
		    except:
			#pdb.set_trace()
			bw_value.append(0)
		bw_value_list.append(bw_value)
		#bed_file.write('%s\t%s\t%s\t%s\n'%(chrom_num,start_num,end_num,name))
                if len(seq):
                    #seq_array = get_RNA_seq_concolutional_array(seq)
                    seq_list.append(seq)
                seq = ''
		bw_value = []
            else:
                seq = seq + line[:-1]
        if len(seq):
            seq_list.append(seq)
    bw_value_list = np.array(bw_value_list)
    #bed_file.close()
    #pdb.set_trace()
    rna_array_3 = []
    rna_array_4 = []
    rna_array_34 = []
    rna_array_36 = []
    rna_array_346 = []
    for rna_seq in seq_list:
        rna_seq = rna_seq.replace('T', 'U')
        rna_seq_pad = padding_sequence(rna_seq, max_len = 1002, repkey = 'N')
        tri_feature_3_for_3 = list(get_6_nucleotide_composition(trids_3, rna_seq_pad, nn_dict_3))
        tri_feature_3_for_34 = list(get_6_nucleotide_composition(trids_3, rna_seq_pad, nn_dict_34))
        tri_feature_3_for_36 = list(get_6_nucleotide_composition(trids_3, rna_seq_pad, nn_dict_36))
        #tri_feature_3_for_346 = list(get_6_nucleotide_composition(trids_3, rna_seq_pad, nn_dict_346))
        tri_feature_4_for_4 = list(get_6_nucleotide_composition(trids_4, rna_seq_pad, nn_dict_4))
        tri_feature_4_for_34 = list(get_6_nucleotide_composition_for_4(trids_4, rna_seq_pad, nn_dict_34,65))
        #tri_feature_4_for_346 = list(get_6_nucleotide_composition_for_4(trids_4, rna_seq_pad, nn_dict_346,65))
        tri_feature_6_for_36 = list(get_6_nucleotide_composition_for_6(trids_6, rna_seq_pad, nn_dict_36,321))
        #tri_feature_6_for_346 = list(get_6_nucleotide_composition_for_6(trids_6, rna_seq_pad, nn_dict_346,321))
        #tri_feature_346 = tri_feature_3_for_346 + tri_feature_4_for_346 + tri_feature_6_for_346
        tri_feature_34 = tri_feature_3_for_34 + tri_feature_4_for_34
        tri_feature_36 = tri_feature_3_for_36 + tri_feature_6_for_36
        tri_feature_3 = tri_feature_3_for_3
        tri_feature_4 = tri_feature_4_for_4
        #rna_array_346.append(np.asarray(tri_feature_346))
        rna_array_34.append(np.asarray(tri_feature_34))
        rna_array_36.append(np.asarray(tri_feature_36))
        rna_array_3.append(np.asarray(tri_feature_3))
        rna_array_4.append(np.asarray(tri_feature_4))

    #return np.array(rna_array_3),np.array(rna_array_4),np.array(rna_array_34),np.array(rna_array_36),np.array(rna_array_346)
    #return np.array(rna_array_3),np.array(rna_array_4),np.array(rna_array_34),np.array(rna_array_36)
    return np.array(rna_array_3),np.array(rna_array_4),np.array(rna_array_34),np.array(rna_array_36),bw_value_list

def read_struct_dict(seq_file, trids_3, trids_4,trids_6,nn_dict_3,nn_dict_4):
    seq_list = []
    label_list = []
    seq = ''
    with gzip.open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
                #posi_label = name.split(';')[-1]
                label = name.split(' ')[-1]
                label_list.append(int(label))
                if len(seq):
                    #seq_array = get_RNA_seq_concolutional_array(seq)
                    seq_list.append(seq)
                seq = ''
            else:
                seq = seq + line[:-1]
        if len(seq):
            seq_list.append(seq)
    #pdb.set_trace()
    rna_array_3 = []
    rna_array_4 = []
    rna_array_34 = []
    rna_array_36 = []
    rna_array_346 = []
    for rna_seq in seq_list:
        #rna_seq = rna_seq.replace('T', 'U')
        #rna_seq_pad = padding_sequence(rna_seq, max_len = 1002, repkey = 'N')
        rna_seq_pad = padding_sequence(rna_seq, max_len = 102, repkey = 'N')
        tri_feature_3_for_3 = list(get_6_nucleotide_composition(trids_3, rna_seq_pad, nn_dict_3))
        #tri_feature_3_for_34 = list(get_6_nucleotide_composition(trids_3, rna_seq_pad, nn_dict_34))
        #tri_feature_3_for_36 = list(get_6_nucleotide_composition(trids_3, rna_seq_pad, nn_dict_36))
        #tri_feature_3_for_346 = list(get_6_nucleotide_composition(trids_3, rna_seq_pad, nn_dict_346))
        tri_feature_4_for_4 = list(get_6_nucleotide_composition(trids_4, rna_seq_pad, nn_dict_4))
        #tri_feature_4_for_34 = list(get_6_nucleotide_composition_for_4(trids_4, rna_seq_pad, nn_dict_34,217))
        #tri_feature_4_for_346 = list(get_6_nucleotide_composition_for_4(trids_4, rna_seq_pad, nn_dict_346,217))
        #tri_feature_6_for_36 = list(get_6_nucleotide_composition_for_6(trids_6, rna_seq_pad, nn_dict_36,1513))
        #tri_feature_6_for_346 = list(get_6_nucleotide_composition_for_6(trids_6, rna_seq_pad, nn_dict_346,1513))
        #tri_feature_346 = tri_feature_3_for_346 + tri_feature_4_for_346 + tri_feature_6_for_346
        #tri_feature_34 = tri_feature_3_for_34 + tri_feature_4_for_34
        #tri_feature_36 = tri_feature_3_for_36 + tri_feature_6_for_36
        tri_feature_3 = tri_feature_3_for_3
        tri_feature_4 = tri_feature_4_for_4
        #rna_array_346.append(np.asarray(tri_feature_346))
        #rna_array_34.append(np.asarray(tri_feature_34))
        #rna_array_36.append(np.asarray(tri_feature_36))
        rna_array_3.append(np.asarray(tri_feature_3))
        rna_array_4.append(np.asarray(tri_feature_4))

    #return np.array(rna_array_3),np.array(rna_array_4),np.array(rna_array_34),np.array(rna_array_36),np.array(rna_array_346)
    return np.array(rna_array_3),np.array(rna_array_4)

def padding_sequence(seq, max_len = 1002, repkey = 'N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq

def get_6_nucleotide_composition_for_6(tris, seq, ordict,bias):
    seq_len = len(seq)
    tri_feature = []
    k = len(tris[0])
    #tmp_fea = [0] * len(tris)

    for x in range(len(seq) + 1- k):
        kmer = seq[x:x+k]
        if kmer in tris:
            ind = tris.index(kmer) + bias
            tri_feature.append(ordict[str(ind)])
        else:
            tri_feature.append(-1)
    return np.asarray(tri_feature)

def get_6_nucleotide_composition_for_4(tris, seq, ordict,bias):
    seq_len = len(seq)
    tri_feature = []
    k = len(tris[0])
    #tmp_fea = [0] * len(tris)

    for x in range(len(seq) + 1- k):
        kmer = seq[x:x+k]
        if kmer in tris:
            ind = tris.index(kmer) + bias
            tri_feature.append(ordict[str(ind)])
        else:
            tri_feature.append(-1)
    return np.asarray(tri_feature)

def get_6_nucleotide_composition(tris, seq, ordict):
    seq_len = len(seq)
    tri_feature = []
    k = len(tris[0])
    #tmp_fea = [0] * len(tris)

    for x in range(len(seq) + 1- k):
        kmer = seq[x:x+k]
        if kmer in tris:
            ind = tris.index(kmer)
            tri_feature.append(ordict[str(ind)])
        else:
            tri_feature.append(-1)
    return np.asarray(tri_feature)

def load_label_seq(seq_file):
    label_list = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
                #posi_label = name.split(';')[-1]
                #label = posi_label.split(':')[-1]
		label = name.split(' ')[-1]
                label_list.append(int(label))
    return np.array(label_list)

def read_rnashape(structure_file):
    struct_dict = {}
    with gzip.open(structure_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[:-1]
            else:
                strucure = line[:-1]
                struct_dict[name] = strucure
                
    return struct_dict

def read_structure(seq_file, path):
    seq_list = []
    structure_list = []
    struct_exist = False
    #if not os.path.exists(path + '/structure.gz'):
    #    fw = gzip.open(path + '/structure.gz', 'w')
    fw = gzip.open(path + '/structure.gz', 'w')
    #else:
    #    fw = None
    #    struct_exist = True
    #    struct_dict = read_rnashape(path + '/structure.gz')
        #pdb.set_trace()
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line
                if len(seq):
                    if struct_exist:
                        structure = struct_dict[old_name[:-1]]
                        seq_array, struct = get_RNA_structure_concolutional_array(seq[499:499+102], fw, structure = structure)
                    else:
                        fw.write(old_name)
                        seq_array, struct = get_RNA_structure_concolutional_array(seq[499:499+102], fw)
                    seq_list.append(seq_array)
                    structure_list.append(struct)
                old_name = name              
                seq = ''
            else:
                seq = seq + line[:-1]
        if len(seq): 
            if struct_exist:
                structure = struct_dict[old_name[:-1]]
                seq_array, struct = get_RNA_structure_concolutional_array(seq[499:499+102], fw, structure = structure)
            else:
                fw.write(old_name)
                seq_array, struct = get_RNA_structure_concolutional_array(seq[499:499+102], fw)
            #seq_array, struct = get_RNA_structure_concolutional_array(seq, fw)
            seq_list.append(seq_array)
            structure_list.append(struct)  
    if fw:
        fw.close()
    return np.array(seq_list), structure_list


def read_oli_feature(seq_file):
    trids4 = get_4_trids()
    seq_list = []
    seq = ''
    with gzip.open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
                if len(seq):
                    seq_array = get_4_nucleotide_composition(trids4, seq)
                    seq_list.append(seq_array)                    
                seq = ''
            else:
                seq = seq + line[:-1]
        if len(seq):
            seq_array = get_4_nucleotide_composition(trids4, seq)
            seq_list.append(seq_array) 
    
    return np.array(seq_list)    

def get_4_trids(chars_input):
    nucle_com = []
    chars = chars_input
    base=len(chars)
    end=len(chars)**4
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n/base
        ch1=chars[n%base]
        n=n/base
        ch2=chars[n%base]
        n=n/base
        ch3=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2 + ch3)
    return  nucle_com

def get_3_trids(chars_input):
    nucle_com = []
    chars = chars_input
    base=len(chars)
    end=len(chars)**3
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n/base
        ch1=chars[n%base]
        n=n/base
        ch2=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2)
    return  nucle_com

def get_6_trids(chars_input):
    nucle_com = []
    chars = chars_input
    base=len(chars)
    end=len(chars)**6
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n/base
        ch1=chars[n%base]
        n=n/base
        ch2=chars[n%base]
        n=n/base
        ch3=chars[n%base]
        n=n/base
        ch4=chars[n%base]
        n=n/base
        ch5=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2 + ch3 + ch4 + ch5)
    return  nucle_com

def get_4_nucleotide_composition(tris, seq, pythoncount = True):
    #pdb.set_trace()
    seq_len = len(seq)
    seq = seq.upper().replace('T', 'U')
    tri_feature = []
    
    if pythoncount:
        for val in tris:
            num = seq.count(val)
            #tri_feature.append(float(num)/seq_len)
            tri_feature.append(float(num))
    else:
        k = len(tris[0])
        tmp_fea = [0] * len(tris)
        for x in range(len(seq) + 1- k):
            kmer = seq[x:x+k]
            if kmer in tris:
                ind = tris.index(kmer)
                tmp_fea[ind] = tmp_fea[ind] + 1
        tri_feature = [float(val)/seq_len for val in tmp_fea]
        #pdb.set_trace()        
    return tri_feature

def process_modify_value(path,infile,inbw):
    import commands
    (status,output) = commands.getstatusoutput('wc l %s/%s'%(path,infile))
    infile_lines = int(output.split(' '))
    tmp = os.system('bigWigAverageOverBed '+path+'/'+inbw+' '+path+'/'+infile+' '+path+'/'+infile+'.tab -bedOut='+path+'/'+infile+'.cons.bed')
    (status,output) = commands.getstatusoutput('cat '+path+'/'+infile+'.cons.bed | awk \'{print $6}\'')
    modify_value = np.array([float(item) for item in output.split('\n')]).reshape(len(output.split('\n')),1)
    
    return modify_value

def load_data(path, seq = True, oli = False):
    """
        Load data matrices from the specified folder.
    """

    data = dict()
    if seq: 
        tmp = []
        tmp.append(read_seq(os.path.join(path, 'sequences.fa.gz')))
        seq_onehot, structure = read_structure(os.path.join(path, 'sequences.fa.gz'), path)
        tmp.append(seq_onehot)
        data["seq"] = tmp
        data["structure"] = structure
    
    if oli: data["oli"] = read_oli_feature(os.path.join(path, 'sequences.fa.gz'))
    
    data["Y"] = load_label_seq(os.path.join(path, 'sequences.fa.gz'))
    #np.loadtxt(gzip.open(os.path.join(path,
                #                            "matrix_Response.tab.gz")),
                #                            skiprows=1)
    #data["Y"] = data["Y"].reshape((len(data["Y"]), 1))

    return data   

def complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    complseq = [complement[base] for base in seq]
    return complseq

def reverse_complement(seq):
    seq = list(seq)
    seq.reverse()
    return ''.join(complement(seq))

def preprocess_data(X, scaler=None, stand = False):
    if not scaler:
        if stand:
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler    

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def get_RNA_seq_concolutional_array(seq, motif_len = 4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    #for seq in seqs:
    #for key, seq in seqs.iteritems():
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*4)
    
    for i in range(row-3, row):
        new_array[i] = np.array([0.25]*4)
        
    #pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        #if val == 'N' or i < motif_len or i > len(seq) - motif_len:
        #    new_array[i] = np.array([0.25]*4)
        #else:
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        #data[key] = new_array
    return new_array[3:-3]

def get_RNA_structure_concolutional_array(seq, fw, structure = None, motif_len = 6):
    if fw is None:
        struc_en = structure
    else:
        #print 'running rnashapes'
        seq = seq.replace('U', 'T')
        struc_en = run_rnashape(seq)
        fw.write(struc_en + '\n')
        
    alpha = 'FTIHMS'
    row = (len(struc_en) + 2*motif_len - 2)
    new_array = np.zeros((row, 6))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.16]*6)
    
    for i in range(row-5, row):
        new_array[i] = np.array([0.16]*6)

    for i, val in enumerate(struc_en):
        i = i + motif_len-1
        if val not in alpha:
            new_array[i] = np.array([0.16]*6)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        
    return new_array, struc_en

# def get_2d_cnn_network():
    # nb_conv = 4
    # nb_pool = 2
    # model = Sequential()
    # model.add(Convolution2D(64, nb_conv, nb_conv,
                            # border_mode='valid',
                            # input_shape=(1, 107, 4)))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(64, nb_conv, nb_conv))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    
    # return model

def fork (model, n=2):
    forks = []
    for i in range(n):
        f = Sequential()
        f.add (model)
        forks.append(f)
    return forks

'''
class MyReshape(Layer):
    def get_output(self, train):
        X = self.get_input(train)
        nshape = (1,) + X.shape 
        return theano.tensor.reshape(X, nshape)
'''

def get_embed_dim(embed_file):
    with open(embed_file) as f:
        pepEmbedding = pickle.load(f)

    embedded_dim = pepEmbedding[0].shape
    print embedded_dim
    n_aa_symbols, embedded_dim = embedded_dim
    print n_aa_symbols, embedded_dim
    # = embedded_dim[0]
    embedding_weights = np.zeros((n_aa_symbols + 1,embedded_dim))
    embedding_weights[1:,:] = pepEmbedding[0]

    return embedded_dim, embedding_weights, n_aa_symbols


def set_cnn_embed(n_aa_symbols, input_length, embedded_dim, embedding_weights, nb_filter = 16,Win_size='10'):
    #nb_filter = 64
    filter_length2 = None
    filter_length3 = None
    if Win_size == '10':
        filter_length = 10
    elif Win_size == '3':
        filter_length = 3
    elif Win_size == '5':
        filter_length = 5
    elif Win_size == '3_10':
        filter_length = 3
	filter_length2 = 10
    elif Win_size == '3_5_10':
        filter_length = 3
        filter_length2 = 10
        filter_length3 = 5
	
    dropout = 0.5
    model = Sequential()
    #pdb.set_trace()
    model.add(Embedding(input_dim=n_aa_symbols+1, output_dim = embedded_dim, weights=[embedding_weights], input_length=input_length, trainable = True))
    print 'after embed', model.output_shape
    model.add(Convolution1D(nb_filter, filter_length, border_mode='valid', init='glorot_normal'))
    model.add(Activation(LeakyReLU(.3)))
    model.add(MaxPooling1D(pool_length=3))
    model.add(Dropout(dropout))
    
    if filter_length2:
        model2 = Sequential()
        #pdb.set_trace()
        model2.add(Embedding(input_dim=n_aa_symbols+1, output_dim = embedded_dim, weights=[embedding_weights], input_length=input_length, trainable = True))
        print 'after embed', model2.output_shape
        model2.add(Convolution1D(nb_filter, filter_length2, border_mode='valid', init='glorot_normal'))
        model2.add(Activation(LeakyReLU(.3)))
        model2.add(MaxPooling1D(pool_length=3))
        model2.add(Dropout(dropout))
	model_all = Sequential()
        model_all.add(Merge([model,model2],mode='concat',concat_axis=1))
	model  = model_all
	if filter_length3:
            model3 = Sequential()
	    model3.add(Embedding(input_dim=n_aa_symbols+1, output_dim = embedded_dim, weights=[embedding_weights], input_length=input_length, trainable = True))
            print 'after embed', model3.output_shape
            model3.add(Convolution1D(nb_filter, filter_length3, border_mode='valid', init='glorot_normal'))
            model3.add(Activation(LeakyReLU(.3)))
            model3.add(MaxPooling1D(pool_length=3))
            model3.add(Dropout(dropout))
            model_all2 = Sequential()
            model_all2.add(Merge([model,model3],mode='concat',concat_axis=1))
            model  = model_all2

    return model


def set_dict_cnn_model(rna_len = 101, nb_filter = 16,rnaEmbedding_name='rnaEmbedding25.pickle',win_size='10'):
    print 'configure cnn network'
    embedded_rna_dim, embedding_rna_weights, n_nucl_symbols = get_embed_dim(rnaEmbedding_name)
    print 'symbol', n_nucl_symbols
    #pdb.set_trace()
    model = set_cnn_embed(n_nucl_symbols, rna_len, embedded_rna_dim, embedding_rna_weights, nb_filter = nb_filter,Win_size=win_size)
    return model

def set_cnn_model(input_dim, input_length,win_size='10'):
    filter_length2 = None
    filter_length3 = None
    if win_size == '10':
        filter_length = 10
    elif win_size == '3':
        filter_length = 3
    elif win_size == '5':
        filter_length = 5
    elif win_size == '3_10':
        filter_length = 3
        filter_length2 = 10
    elif win_size == '3_5_10':
        filter_length = 3
        filter_length2 = 10
        filter_length3 = 5
    nbfilter = 16
    model = Sequential()
    model.add(Convolution1D(input_dim=input_dim,input_length=input_length,
                            nb_filter=nbfilter,
                            filter_length=filter_length,
                            border_mode="valid",
                            #activation="relu",
                            subsample_length=1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=3))

    model.add(Dropout(0.5))

    if filter_length2:
        model2 = Sequential()
	model2.add(Convolution1D(input_dim=input_dim,input_length=input_length,
                            nb_filter=nbfilter,
                            filter_length=filter_length2,
                            border_mode="valid",
                            #activation="relu",
                            subsample_length=1))
        model2.add(Activation('relu'))
        model2.add(MaxPooling1D(pool_length=3))

        model2.add(Dropout(0.5))
	model_all = Sequential()
        model_all.add(Merge([model,model2],mode='concat',concat_axis=1))
        model  = model_all
	if filter_length3:
	    model3 = Sequential()
            model3.add(Convolution1D(input_dim=input_dim,input_length=input_length,
                            nb_filter=nbfilter,
                            filter_length=filter_length3,
                            border_mode="valid",
                            #activation="relu",
                            subsample_length=1))
            model3.add(Activation('relu'))
            model3.add(MaxPooling1D(pool_length=3))

            model3.add(Dropout(0.5))
	    model_all2 = Sequential()
            model_all2.add(Merge([model,model3],mode='concat',concat_axis=1))
            model  = model_all2

    return model

def set_cnn_model2(input_dim, input_length,win_size='10'):
    filter_length2 = None
    filter_length3 = None
    if win_size == '10':
        filter_length = 10
    elif win_size == '3':
        filter_length = 3
    elif win_size == '5':
        filter_length = 5
    elif win_size == '3_10':
        filter_length = 3
        filter_length2 = 10
    elif win_size == '3_5_10':
        filter_length = 3
        filter_length2 = 10
        filter_length3 = 5
    nbfilter = 16
    input = Input(shape=(input_length,input_dim))
    x_10 = Convolution1D(input_dim=input_dim,input_length=input_length,
                            nb_filter=nbfilter,
                            filter_length=filter_length,
                            border_mode="valid",
                            #activation="relu",
                            subsample_length=1)(input)
    x_10 = Activation('relu')(x_10)
    x_10 = MaxPooling1D(pool_length=3)(x_10)
    x_10 = Dropout(0.5)(x_10)
    model = Sequential()
    model.add(Convolution1D(input_dim=input_dim,input_length=input_length,
                            nb_filter=nbfilter,
                            filter_length=filter_length,
                            border_mode="valid",
                            #activation="relu",
                            subsample_length=1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=3))

    model.add(Dropout(0.5))
    model_all = Model(input,x_10)
    model = model_all

    if filter_length2:
	x_3_10 = Convolution1D(input_dim=input_dim,input_length=input_length,
                            nb_filter=nbfilter,
                            filter_length=filter_length2,
                            border_mode="valid",
                            #activation="relu",
                            subsample_length=1)(input)
	x_3_10 = Activation('relu')(x_3_10)
        x_3_10 = MaxPooling1D(pool_length=3)(x_3_10)
        x_3_10 = Dropout(0.5)(x_3_10)
        model2 = Sequential()
        model2.add(Convolution1D(input_dim=input_dim,input_length=input_length,
                            nb_filter=nbfilter,
                            filter_length=filter_length2,
                            border_mode="valid",
                            #activation="relu",
                            subsample_length=1))
        model2.add(Activation('relu'))
        model2.add(MaxPooling1D(pool_length=3))

        model2.add(Dropout(0.5))
	output = merge([x_10,x_3_10],mode='concat',concat_axis=1)
	model_all = Model(input,output)
        #model_all = Sequential()
        #model_all.add(Merge([model,model2],mode='concat',concat_axis=1))
        model  = model_all
        if filter_length3:
	    x_3_5_10 = Convolution1D(input_dim=input_dim,input_length=input_length,
                            nb_filter=nbfilter,
                            filter_length=filter_length3,
                            border_mode="valid",
                            #activation="relu",
                            subsample_length=1)(input)
	    x_3_5_10 = Activation('relu')(x_3_5_10)
            x_3_5_10 = MaxPooling1D(pool_length=3)(x_3_5_10)
            x_3_5_10 = Dropout(0.5)(x_3_5_10)
            model3 = Sequential()
            model3.add(Convolution1D(input_dim=input_dim,input_length=input_length,
                            nb_filter=nbfilter,
                            filter_length=filter_length3,
                            border_mode="valid",
                            #activation="relu",
                            subsample_length=1))
            model3.add(Activation('relu'))
            model3.add(MaxPooling1D(pool_length=3))

            model3.add(Dropout(0.5))
	    output = merge([x_10,x_3_10,x_3_5_10],mode='concat',concat_axis=1)
	    model_all = Model(input,output)
            model_all2 = Sequential()
            model_all2.add(Merge([model,model3],mode='concat',concat_axis=1))
            model  = model_all

    return model

def get_cnn_network_NR(MErge=False):
    model = Sequential()
    model.add(Convolution1D(input_dim=4,
                        input_length=1002,
                        nb_filter=128,
                        filter_length=5,
                        init='glorot_normal',
                        border_mode="same",
                        activation="relu",
                        subsample_length=1))
    print model.output_shape
    #pdb.set_trace()
    model.add(MaxPooling1D(pool_length=5, stride=3))
    print model.output_shape

#model.add(Dropout(0.2))

##add a cnn layer

    model.add(Convolution1D(input_dim=128,
                        input_length=333,
                        nb_filter=256,
                        filter_length=5,
                        init='glorot_normal',
                        border_mode="same",
                        activation="relu",
                        subsample_length=1))
    model.add(MaxPooling1D(pool_length=5, stride=3))

    model.add(Convolution1D(input_dim=256,
                        input_length=111,
                        nb_filter=512,
                        filter_length=5,
                        init='glorot_normal',
                        border_mode="same",
                        activation="relu",
                        subsample_length=1))
    model.add(MaxPooling1D(pool_length=5, stride=3))
    model.add(Dropout(0.2))

##end add layer
#model.add(brnn)

#model.add(SimpleRNN(output_dim=320,init='glorot_normal', inner_init='orthogonal',activation='tanh',input_dim=320,return_sequences=True))
#model.add(GRU(input_dim=320, output_dim=320, init='glorot_normal', inner_init='orthogonal', activation='tanh', inner_activation='hard_sigmoid',return_sequences=True))

#model.add(Dropout(0.5))
    if not MErge:
        model.add(Flatten())
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(0.5))
        print model.output_shape
    return model

def get_cnn_network(struct_onehot=True,struct_dict_num = 3,merge=False,struct_dict=False,onehot_only=False,bw_add=False,attention=False,use_two_data=False,use_LSTM=False,one_data=False,only_histone=False,win_size='10'):
    '''
     get_feature = theano.function([origin_model.layers[0].input],origin_model.layers[11].get_output(train=False),allow_input_downcast=False)
    feature = get_feature(data)
    '''
    nbfilter = 16
    rna_max_len = 1002
    rna_struct_max_len = 102
    print 'configure cnn network'
    if bw_add:
        seq_model = set_cnn_model(6,1002,win_size=win_size) ## 5
        seq_model2 = set_cnn_model2(6,1002,win_size=win_size)
        seq_model_reshape = set_cnn_model(6,1002,win_size=win_size)
	seq_model_reshape.add(core.Reshape((seq_model.output_shape[2],seq_model.output_shape[1])))
        phastCons_model = set_cnn_model(2,1002)
        dict_seq_model = set_dict_cnn_model(rna_len = rna_max_len-5 + rna_max_len-2+1, nb_filter = nbfilter, rnaEmbedding_name='rnaEmbedding25_3_6.pickle',win_size=win_size)
        dict_seq_model_reshape = set_dict_cnn_model(rna_len = rna_max_len-5 + rna_max_len-2+1, nb_filter = nbfilter, rnaEmbedding_name='rnaEmbedding25_3_6.pickle',win_size=win_size)
	#pdb.set_trace()
	dict_seq_model_reshape.add(core.Reshape((dict_seq_model.output_shape[2], dict_seq_model.output_shape[1])))
    elif not bw_add:
        seq_model = set_cnn_model(4, 1002)
        seq_model_reshape = set_cnn_model(4, 1002)
        dict_seq_model = set_dict_cnn_model(rna_len = rna_max_len-5 + rna_max_len-2, nb_filter = nbfilter, rnaEmbedding_name='rnaEmbedding25_3_6.pickle')
    #dict_seq_model = set_dict_cnn_model(rna_len = rna_max_len-2, nb_filter = nbfilter, rnaEmbedding_name='rnaEmbedding25_3.pickle')
    #dict_seq_model = set_dict_cnn_model(rna_len = rna_max_len-5 + rna_max_len-2, nb_filter = nbfilter, rnaEmbedding_name='rnaEmbedding25_3_6.pickle')
    if struct_dict_num == 3:
        dict_struct_model = set_dict_cnn_model(rna_len = rna_struct_max_len-2, nb_filter = nbfilter, rnaEmbedding_name='rnaEmbedding25_struct_3.pickle')
    elif struct_dict_num == 4:
        dict_struct_model = set_dict_cnn_model(rna_len = rna_struct_max_len-3, nb_filter = nbfilter, rnaEmbedding_name='rnaEmbedding25_struct_4.pickle')
    struct_model = set_cnn_model(6, 112)
    model = Sequential()
    if merge and not struct_dict and not onehot_only:
        model.add(Merge([seq_model, dict_seq_model], mode='concat', concat_axis=1))
    elif merge and struct_dict and not onehot_only:
	model.add(Merge([seq_model, dict_struct_model], mode='concat', concat_axis=1))
    elif not merge and not onehot_only:
        model.add(dict_seq_model)
    elif not merge and onehot_only:
        #model.add(Merge([seq_model, phastCons_model], mode='concat', concat_axis=1))
	if not bw_add:
            model.add(get_cnn_network_NR(MErge=True))
	elif bw_add:
	    if only_histone:
		model.add(phastCons_model)
	    else:
                model.add(seq_model)
	if one_data:
	    model = Sequential()
            model.add(seq_model2)
    #if struct_onehot:
    #    model.add(Merge([dict_seq_model, struct_model], mode='concat', concat_axis=1))
    #else: 
    #	model.add(Merge([dict_seq_model, dict_struct_model], mode='concat', concat_axis=1))
    model_all = Sequential()
    if attention:
	if not use_two_data:
	    if use_LSTM:
        	model.add(Bidirectional(LSTM(2*nbfilter,return_sequences=True)))
            model.add(Attention())
	elif use_two_data:
	    if use_LSTM:
		dict_seq_model.add(Bidirectional(LSTM(2*nbfilter,return_sequences=True)))
		dict_seq_model_reshape.add(Bidirectional(LSTM(2*nbfilter,return_sequences=True)))
		seq_model_reshape.add(Bidirectional(LSTM(2*nbfilter,return_sequences=True)))
	    dict_seq_model.add(Attention())
	    dict_seq_model_reshape.add(Attention())
	    seq_model.add(Attention())
	    seq_model_reshape.add(Attention())
	    if not merge and onehot_only and bw_add:
		model_all.add(Merge([seq_model,seq_model_reshape],mode='concat'))
	    elif not merge and not onehot_only:
	        model_all.add(Merge([dict_seq_model,dict_seq_model_reshape],mode='concat'))
	    model = model_all
    elif not attention:
	if not use_LSTM:
	    model.add(Bidirectional(LSTM(2*nbfilter)))
	elif use_LSTM:
	    model.add(Flatten())

    
    model.add(Dropout(0.10))
    
    model.add(Dense(nbfilter*2, activation='relu'))
    print model.output_shape
    
    return model

def get_cnn_network_old():
    '''
     get_feature = theano.function([origin_model.layers[0].input],origin_model.layers[11].get_output(train=False),allow_input_downcast=False)
    feature = get_feature(data)
    '''
    print 'configure cnn network'
    nbfilter = 16


    model = Sequential()
    model.add(Convolution1D(input_dim=4,input_length=107,
                            nb_filter=nbfilter,
                            filter_length=10,
                            border_mode="valid",
                            #activation="relu",
                            subsample_length=1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=3))
    
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    
    model.add(Dense(nbfilter, activation='relu'))

    model.add(Dropout(0.25))

    
    return model

def get_struct_network():
    '''
     get_feature = theano.function([origin_model.layers[0].input],origin_model.layers[11].get_output(train=False),allow_input_downcast=False)
    feature = get_feature(data)
    '''
    print 'configure cnn network'
    nbfilter = 16

    model = Sequential()
    model.add(Convolution1D(input_dim=6,input_length=107,
                            nb_filter=nbfilter,
                            filter_length=10,
                            border_mode="valid",
                            #activation="relu",
                            subsample_length=1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=3))
    
    model.add(Dropout(0.5))

    
    model.add(Flatten())
    
    model.add(Dense(nbfilter, activation='relu'))

    model.add(Dropout(0.25))
    
    return model


def get_rnn_fea(train, sec_num_hidden = 128, num_hidden = 128):
    print 'configure network for', train.shape
    model = Sequential()

    model.add(Dense(num_hidden, input_shape=(train.shape[1],), activation='relu'))
    model.add(PReLU())
    model.add(BatchNormalization(mode=2))
    model.add(Dropout(0.5))
    model.add(Dense(num_hidden, input_dim=num_hidden, activation='relu'))
    #model.add(Dense(num_hidden, input_shape=(num_hidden,), activation='relu'))
    model.add(PReLU())
    model.add(BatchNormalization(mode=2))
    #model.add(Activation('relu'))
    model.add(Dropout(0.5))

    return model

def get_structure_motif_fig(filter_weights, filter_outs, out_dir, protein, seq_targets, sample_i = 0, structure = None):
    print 'plot motif fig', out_dir
    #seqs, seq_targets = get_seq_targets(protein)
    seqs = structure
    if sample_i:
        print 'sampling'
        seqs = []
        for ind, val in enumerate(seqs):
            if ind in sample_i:
                seqs.append(val)
            
        
        seq_targets = seq_targets[sample_i]
        filter_outs = filter_outs[sample_i]
    
    num_filters = filter_weights.shape[0]
    filter_size = 7 #filter_weights.shape[2]

    
    #################################################################
    # individual filter plots
    #################################################################
    # also save information contents
    filters_ic = []
    meme_out = structure_motifs.meme_intro('%s/filters_meme.txt'%out_dir, seqs)

    for f in range(num_filters):
        print 'Filter %d' % f

        # plot filter parameters as a heatmap
        structure_motifs.plot_filter_heat(filter_weights[f,:,:], '%s/filter%d_heat.pdf' % (out_dir,f))

        # write possum motif file
        structure_motifs.filter_possum(filter_weights[f,:,:], 'filter%d'%f, '%s/filter%d_possum.txt'%(out_dir,f), False)
        
        structure_motifs.plot_filter_logo(filter_outs[:,:, f], filter_size, seqs, '%s/filter%d_logo'%(out_dir,f), maxpct_t=0.5)
        
        filter_pwm, nsites = structure_motifs.make_filter_pwm('%s/filter%d_logo.fa'%(out_dir,f))
        if nsites < 10:
            # no information
            filters_ic.append(0)
        else:
            # compute and save information content
            filters_ic.append(info_content(filter_pwm))

            # add to the meme motif file
            structure_motifs.meme_add(meme_out, f, filter_pwm, nsites, False)

    meme_out.close()
    
            
def get_motif_fig(filter_weights, filter_outs, out_dir, protein, sample_i = 0):
    print 'plot motif fig', out_dir
    seqs, seq_targets = get_seq_targets(protein)
    if sample_i:
        print 'sampling'
        seqs = []
        for ind, val in enumerate(seqs):
            if ind in sample_i:
                seqs.append(val)
            
        
        seq_targets = seq_targets[sample_i]
        filter_outs = filter_outs[sample_i]
    
    num_filters = filter_weights.shape[0]
    filter_size = 7#filter_weights.shape[2]

    #pdb.set_trace()
    #################################################################
    # individual filter plots
    #################################################################
    # also save information contents
    filters_ic = []
    meme_out = meme_intro('%s/filters_meme.txt'%out_dir, seqs)

    for f in range(num_filters):
        print 'Filter %d' % f

        # plot filter parameters as a heatmap
        plot_filter_heat(filter_weights[f,:,:], '%s/filter%d_heat.pdf' % (out_dir,f))

        # write possum motif file
        filter_possum(filter_weights[f,:,:], 'filter%d'%f, '%s/filter%d_possum.txt'%(out_dir,f), False)

        # plot weblogo of high scoring outputs
        plot_filter_logo(filter_outs[:,:, f], filter_size, seqs, '%s/filter%d_logo'%(out_dir,f), maxpct_t=0.5)

        # make a PWM for the filter
        filter_pwm, nsites = make_filter_pwm('%s/filter%d_logo.fa'%(out_dir,f))

        if nsites < 10:
            # no information
            filters_ic.append(0)
        else:
            # compute and save information content
            filters_ic.append(info_content(filter_pwm))

            # add to the meme motif file
            meme_add(meme_out, f, filter_pwm, nsites, False)

    meme_out.close()


    #################################################################
    # annotate filters
    #################################################################
    # run tomtom #-evalue 0.01 
    subprocess.call('tomtom -dist pearson -thresh 0.05 -eps -oc %s/tomtom %s/filters_meme.txt %s' % (out_dir, out_dir, 'Ray2013_rbp_RNA.meme'), shell=True)

    # read in annotations
    filter_names = name_filters(num_filters, '%s/tomtom/tomtom.txt'%out_dir, 'Ray2013_rbp_RNA.meme')


    #################################################################
    # print a table of information
    #################################################################
    table_out = open('%s/table.txt'%out_dir, 'w')

    # print header for later panda reading
    header_cols = ('', 'consensus', 'annotation', 'ic', 'mean', 'std')
    print >> table_out, '%3s  %19s  %10s  %5s  %6s  %6s' % header_cols

    for f in range(num_filters):
        # collapse to a consensus motif
        consensus = filter_motif(filter_weights[f,:,:])

        # grab annotation
        annotation = '.'
        name_pieces = filter_names[f].split('_')
        if len(name_pieces) > 1:
            annotation = name_pieces[1]

        # plot density of filter output scores
        fmean, fstd = plot_score_density(np.ravel(filter_outs[:,:, f]), '%s/filter%d_dens.pdf' % (out_dir,f))

        row_cols = (f, consensus, annotation, filters_ic[f], fmean, fstd)
        print >> table_out, '%-3d  %19s  %10s  %5.2f  %6.4f  %6.4f' % row_cols

    table_out.close()


    #################################################################
    # global filter plots
    #################################################################
    if True:
        new_outs = []
        for val in filter_outs:
            new_outs.append(val.T)
        filter_outs = np.array(new_outs)
        print filter_outs.shape
        # plot filter-sequence heatmap
        plot_filter_seq_heat(filter_outs, '%s/filter_seqs.pdf'%out_dir)
    
def get_seq_targets(protein):
    path = "./datasets/clip/%s/30000/test_sample_0" % protein
    data = load_data(path)
    seq_targets = np.array(data['Y'])
    
    seqs = []
    seq = ''
    fp = gzip.open(path +'/sequences.fa.gz')
    for line in fp:
        if line[0] == '>':
            name = line[1:-1]
            if len(seq):
                seqs.append(seq)                    
            seq = ''
        else:
            seq = seq + line[:-1].replace('T', 'U')
    if len(seq):
        seqs.append(seq) 
    fp.close()
    
    return seqs, seq_targets

def get_features():
    all_weights = []
    for layer in model.layers:
       w = layer.get_weights()
       all_weights.append(w)
       
    return all_weights

def convout1_f(X):
    # The [0] is to disable the training phase flag
    return _convout1_f([0] + [X])

def get_feature(model, X_batch, index):
    inputs = [K.learning_phase()] + [model.inputs[index]]
    _convout1_f = K.function(inputs, model.layers[0].layers[index].layers[1].output)
    activations =  _convout1_f([0] + [X_batch[index]])
    
    return activations

def get_motif(model, testing, protein, y, index = 0, dir1 = 'seq_cnn/', structure  = None):
    sfilter = model.layers[0].layers[index].layers[0].get_weights()
    filter_weights_old = np.transpose(sfilter[0][:,0,:,:], (2, 1, 0)) #sfilter[0][:,0,:,:]
    print filter_weights_old.shape
    #pdb.set_trace()
    filter_weights = []
    for x in filter_weights_old:
        #normalized, scale = preprocess_data(x)
        #normalized = normalized.T
        #normalized = normalized/normalized.sum(axis=1)[:,None]
        x = x - x.mean(axis = 0)
        filter_weights.append(x)
        
    filter_weights = np.array(filter_weights)
    #pdb.set_trace()
    filter_outs = get_feature(model, testing, index)
    #pdb.set_trace()
    
    #sample_i = np.array(random.sample(xrange(testing.shape[0]), 500))
    sample_i =0

    out_dir = dir1 + protein
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    if index == 0:    
        get_motif_fig(filter_weights, filter_outs, out_dir, protein, sample_i)
    else:
        get_structure_motif_fig(filter_weights, filter_outs, out_dir, protein, y, sample_i, structure)
    
def run_network(model, total_hid, training, testing, y, validation, val_y, protein=None, structure = None):
    model.add(Dense(2, input_shape=(total_hid,)))
    model.add(Activation('softmax'))
    
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    #pdb.set_trace()
    print 'model training'
    #checkpointer = ModelCheckpoint(filepath="models/" + protein + "_bestmodel.hdf5", verbose=0, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

    model.fit(training, y, batch_size=100, nb_epoch=15, verbose=0, validation_data=(validation, val_y), callbacks=[earlystopper])
    
    #pdb.set_trace()
    #get_motif(model, testing, protein, y, index = 0, dir1 = 'seq_cnn1/')
    #get_motif(model, testing, protein, y, index = 1, dir1 = 'structure_cnn1/', structure = structure)

    predictions = model.predict_proba(testing)[:,1]
    return predictions, model

def run_randomforest_classifier(data, labels, test):
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(data, labels)
    #pdb.set_trace()
    pred_prob = clf.predict_proba(test)[:,1]
    return pred_prob, clf  

def run_svm_classifier(data, labels, test):
    #C_range = 10.0 ** np.arange(-1, 2)
    #param_grid = dict(C=C_range.tolist())
    clf = svm.SVC(probability =True, kernel = 'linear')
    #grid = GridSearchCV(svr, param_grid)
    clf.fit(data, labels)
    
    #clf = grid.best_estimator_
    pred_prob = clf.predict_proba(test)[:,1]
    return pred_prob, clf
    
def calculate_auc(net, hid, train, test, true_y, train_y, rf = False, validation = None, val_y = None, protein = None, structure = None):
    #print 'running network' 
    if rf:
        print 'running oli'
        #pdb.set_trace()
        predict, model = run_svm_classifier(train, train_y, test)
    else:
        predict, model = run_network(net, hid, train, test, train_y, validation, val_y, protein = protein, structure = structure)

    
    auc = roc_auc_score(true_y, predict)
    
    print "Test AUC: ", auc
    return auc, predict



def run_seq_struct_cnn_network(protein, seq = True, fw = None, oli = False, min_len = 301):
    training_data = load_data("./datasets/clip/%s/30000/training_sample_0" % protein, seq = seq, oli = oli)
    
    seq_hid = 16
    struct_hid = 16
    #pdb.set_trace()
    train_Y = training_data["Y"]
    print len(train_Y)
    #pdb.set_trace()
    training_indice, training_label, validation_indice, validation_label = split_training_validation(train_Y)
    #pdb.set_trace()
    if seq:
        cnn_train  = []
        cnn_validation = []
        seq_data = training_data["seq"][0]
        #pdb.set_trace()
        seq_train = seq_data[training_indice]
        seq_validation = seq_data[validation_indice] 
        struct_data = training_data["seq"][1]
        struct_train = struct_data[training_indice]
        struct_validation = struct_data[validation_indice] 
        cnn_train.append(seq_train)
        cnn_train.append(struct_train)
        cnn_validation.append(seq_validation)
        cnn_validation.append(struct_validation)        
        seq_net =  get_cnn_network()
        seq_data = []
            
    y, encoder = preprocess_labels(training_label)
    val_y, encoder = preprocess_labels(validation_label, encoder = encoder) 
    
    training_data.clear()
    
    rf = False

    test_data = load_data("./datasets/clip/%s/30000/test_sample_0" % protein, seq = seq, oli = oli)
    print len(test_data)
    true_y = test_data["Y"].copy()
    
    print 'predicting'    
    if seq:
        testing = test_data["seq"]
        #structure = test_data["structure"]
        seq_auc, seq_predict = calculate_auc(seq_net, seq_hid + struct_hid, cnn_train, testing, true_y, y, validation = cnn_validation,
                                              val_y = val_y, protein = protein,  rf= rf, structure = structure)
        seq_train = []
        seq_test = []
         
        
        
    print str(seq_auc)
    fw.write( str(seq_auc) +'\n')

    mylabel = "\t".join(map(str, true_y))
    myprob = "\t".join(map(str, seq_predict))  
    fw.write(mylabel + '\n')
    fw.write(myprob + '\n')

def split_training_validation(classes, validation_size = 0.1, shuffle = False):
    """split sampels based on balnace classes"""
    num_samples=len(classes)
    classes=np.array(classes)
    classes_unique=np.unique(classes)
    num_classes=len(classes_unique)
    indices=np.arange(num_samples)
    #indices_folds=np.zeros([num_samples],dtype=int)
    training_indice = []
    training_label = []
    validation_indice = []
    validation_label = []
    for cl in classes_unique:
        indices_cl=indices[classes==cl]
        num_samples_cl=len(indices_cl)

        # split this class into k parts
        if shuffle:
            random.shuffle(indices_cl) # in-place shuffle
        
        # module and residual
        num_samples_each_split=int(num_samples_cl*validation_size)
        res=num_samples_cl - num_samples_each_split
        
        training_indice = training_indice + [val for val in indices_cl[num_samples_each_split:]]
        training_label = training_label + [cl] * res
        
        validation_indice = validation_indice + [val for val in indices_cl[:num_samples_each_split]]
        validation_label = validation_label + [cl]*num_samples_each_split

    training_index = np.arange(len(training_label))
    random.shuffle(training_index)
    training_indice = np.array(training_indice)[training_index]
    training_label = np.array(training_label)[training_index]
    
    validation_index = np.arange(len(validation_label))
    random.shuffle(validation_index)
    validation_indice = np.array(validation_indice)[validation_index]
    validation_label = np.array(validation_label)[validation_index]    
    
            
    return training_indice, training_label, validation_indice, validation_label        
        

def plot_roc_curve(labels, probality, legend_text, auc_tag = True):
    #fpr2, tpr2, thresholds = roc_curve(labels, pred_y)
    fpr, tpr, thresholds = roc_curve(labels, probality) #probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    if auc_tag:
        rects1 = plt.plot(fpr, tpr, label=legend_text +' (AUC=%6.3f) ' %roc_auc)
    else:
        rects1 = plt.plot(fpr, tpr, label=legend_text )

def read_protein_name(filename='proteinnames'):
    protein_dict = {}
    with open(filename, 'r') as fp:
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            key_name = values[0][1:-1]
            protein_dict[key_name] = values[1]
    return protein_dict
    
def read_result_file(filename = 'result_file_seq_wohle_new'):
    results = {}
    with open(filename, 'r') as fp:
        index = 0
        #protein = '28'
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            if index % 3 == 0:
                protein = values[0].split('_')[0]
            if index % 3 != 0:
                results.setdefault(protein, []).append(values)
                
                
            index = index + 1
    
    return results

def read_individual_auc(filename = 'result_file_all_new'):
    results = {}
    with open(filename, 'r') as fp:
        index = 0
        #protein = '28'
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            pro = values[0].split('_')[0]
            results[int(pro)] = values[1:-1]    
    
    return results

def read_ideep_auc(filename = 'result_mix_auc_new'):
    results = {}
    with open(filename, 'r') as fp:
        index = 0
        #protein = '28'
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            #pdb.set_trace()
            pro = values[0].split('_')[0]
            results[int(pro)] = values[1]  
    
    return results

def plot_ideep_indi_comp():
    proteins = read_protein_name()
    ideep_resut = read_ideep_auc(filename='result_mix_auc_new')
    #pdb.set_trace()
    indi_result = read_individual_auc()
    keys = indi_result.keys()
    keys.sort()
    
    new_results = []
    names = []
    for key in keys:
        str_key = str(key)
        names.append(proteins[str_key])
        tmp = []
        for val in indi_result[key]:
            tmp.append(float(val))
        #for val in ideep_resut[key]:
        tmp.append(float(ideep_resut[key]))
        #tmp = indi_result[key] + ideep_resut[key]
        new_results.append(tmp)
    pdb.set_trace()
    new_results = map(list, zip(*new_results))
    #plot_confusion_matrix(new_results)
    plot_parameter_bar(new_results, names)
            
def plot_figure():
    protein_dict = read_protein_name()
    results = read_result_file()
    
    Figure = plt.figure(figsize=(12, 15))
    
    for key, values in results.iteritems():
        protein = protein_dict[key]
        #pdb.set_trace()
        labels = [int(float(val)) for val in values[0]]
        probability = [float(val) for val in values[1]]
        plot_roc_curve(labels, probability, protein)
    #plot_roc_curve(labels[1], probability[1], '')
    #plot_roc_curve(labels[2], probability[2], '')
    
    #title_type = 'stem cell circRNAs vs other circRNAs'
    title_type = 'ROC'
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title(title_type)
    plt.legend(loc="lower right")
    plt.savefig('roc1.eps', format='eps') 
    #plt.show() 
  
def read_fasta_file(fasta_file):
    seq_dict = {}    
    fp = gzip.open(fasta_file, 'r')
    name = ''
    name_list = []
    for line in fp:
        line = line.rstrip()
        #distinguish header from sequence
        if line[0]=='>': #or line.startswith('>')
            #it is the header
            name = line[2:] #discarding the initial >
            name_list.append(name)
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line.upper().replace('U', 'T')
    fp.close()
    
    return seq_dict, name_list

def read_rna_dict(rna_dict_name):
    odr_dict = {}
    with open(rna_dict_name, 'r') as fp:
        for line in fp:
            values = line.rstrip().split(',')
            for ind, val in enumerate(values):
                val = val.strip()
                odr_dict[val] = ind

    return odr_dict

def run_predict():
    data_dir = './datasets/clip'
    fw = open('result_file_struct_auc', 'w')
    for protein in os.listdir(data_dir):
        print protein

        fw.write(protein + '\t')

        run_seq_struct_cnn_network(protein, seq = True, fw= fw)

    fw.close()
    
def load_data_file(inputfile, seq = True, onlytest = False):
    """
        Load data matrices from the specified folder.
    """
    path = os.path.dirname(inputfile)
    data = dict()
    seq_chars = ['A', 'C', 'G', 'U']
    struct_chars = ['F', 'T', 'I', 'H', 'M', 'S']
    trids_6 = get_6_trids(seq_chars)
    trids_3 = get_3_trids(seq_chars)
    #trids = get_7_trids()
    trids_4 = get_4_trids(seq_chars)
    trids_struct_6 = get_6_trids(struct_chars)
    trids_struct_3 = get_3_trids(struct_chars)
    #trids_struct_7 = get_7_trids(struct_chars)
    trids_struct_4 = get_4_trids(struct_chars)
    nn_dict_3 = read_rna_dict('rna_dict_3_backup')
    nn_dict_4 = read_rna_dict('rna_dict_4_backup')
    nn_dict_34 = read_rna_dict('rna_dict_3_4_backup')
    nn_dict_36 = read_rna_dict('rna_dict_3_6_backup')
    nn_dict_346 = read_rna_dict('rna_dict_3_4_6.backup')

    nn_struct_dict_3 = read_rna_dict('rna_struct_dict_3_backup')
    nn_struct_dict_4 = read_rna_dict('rna_struct_dict_4_backup')
    inbw = 'hg19.100way.phastCons.bw'
    inbw = 'histone/wgEncodeBroadHistoneGm12878H3k27acStdSig.bigWig'
    inbw = 'histone/wgEncodeBroadHistoneH1hescH3k27acStdSig.bigWig'
    inbw = 'GSM1368907_TW246_Gm12878_MeDIP.bigWig' #'GSM1368906_TW245_K562_MeDIP.bigWig'
    inbw_histone = 'histone/wgEncodeBroadHistoneGm12878H3k27acStdSig.bigWig' #'histone/wgEncodeBroadHistoneK562H3k27acStdSig.bigWig'
    
    #nn_struct_dict_34 = read_rna_dict('rna_struct_dict_3_4_backup')
    #nn_struct_dict_36 = read_rna_dict('rna_struct_dict_3_6_backup')
    #nn_struct_dict_346 = read_rna_dict('rna_struct_dict_3_4_6.backup')
    if seq: 
        tmp = []
	one_hot_data = read_seq(inputfile,path+'/'+inbw,path+'/'+inbw_histone)
        tmp.append(one_hot_data[:,:,:-2])
	## add word2vec dict_seq
	#pdb.set_trace()
	print 'read seq dict'
	#data_3,data_4,data_34,data_36,bw_value_list = read_seq_dict(inputfile, trids_3, trids_4,trids_6,nn_dict_3,nn_dict_4,nn_dict_34,nn_dict_36,nn_dict_346,[path+'/'+inbw])
	#modify_value = process_modify_value(path,'sequence.bed',inbw)
        #pdb.set_trace()
	#data_3_add_modify = np.concatenate((np.array(data_3),bw_value_list),axis=1)
    	#data_4_add_modify = np.concatenate((np.array(data_4),bw_value_list),axis=1)
        #data_34_add_modify = np.concatenate((np.array(data_34),bw_value_list),axis=1)
    	#data_36_add_modify = np.concatenate((np.array(data_36),bw_value_list),axis=1)
	#struct_data_3,struct_data_4,struct_data_34,struct_data_36,struct_data_346 = read_struct_dict(path+ '/structure.gz', trids_struct_3, trids_struct_4,trids_struct_6,nn_struct_dict_3,nn_struct_dict_4,nn_struct_dict_34,nn_struct_dict_36,nn_struct_dict_346)
	data_4_mer_freq = get_4mer_freq(inputfile,trids_4)
	## end add word2vec dict_seq
	#pdb.set_trace()
	##print 'read struct one hot'
        ##seq_onehot, structure = read_structure(inputfile, path)
	##print 'read struct dict'
	##struct_data_3,struct_data_4 = read_struct_dict(path+ '/structure.gz', trids_struct_3, trids_struct_4,trids_struct_6,nn_struct_dict_3,nn_struct_dict_4)
        #tmp.append(seq_onehot)
        #tmp.append(data_3)
        #tmp.append(data_36)
	#tmp.append(data_4_mer_freq)
	tmp.append(data_4_mer_freq)
	tmp.append(data_4_mer_freq)
	#tmp.append(data_4_mer_freq)
        tmp.append(one_hot_data)
	tmp.append(data_4_mer_freq)
        #tmp.append(data_36_add_modify)
        #tmp.append(struct_data_3)
        #tmp.append(struct_data_4)
        #tmp.append(data_36)
        #tmp.append(one_hot_data)
        #tmp.append(data_36_add_modify)
	tmp.append(data_4_mer_freq)
	tmp.append(one_hot_data[:,:,-2:]) #.reshape(one_hot_data.shape[0],one_hot_data.shape[1],1))
        data["seq"] = tmp
        #data["structure"] = structure
    if onlytest:
        data["Y"] = []
    else:
        data["Y"] = load_label_seq(inputfile)
        
    return data

def run_network_new(model, total_hid, training, y, validation, val_y, batch_size=100, nb_epoch=15):
    model.add(Dense(2, input_shape=(total_hid,)))
    print model.output_shape
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')#, class_mode="binary")
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    #pdb.set_trace()
    print 'model training'

    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

    model.fit(training, y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(validation, val_y), callbacks=[earlystopper])

    return model    
    
def train_ideeps(data_file, model_dir, batch_size= 100, nb_epoch = 15,dataset_num=1):
    training_data = load_data_file(data_file)
    
    seq_hid = 16
    struct_hid = 16
    train_Y = training_data["Y"]
    print len(train_Y)
    #result_file_out = open('result_file.txt','a')
    result_file_out = open('GM12878_result_file_Only_histone_MeDIP_both.txt','a')

    #result_file_out = open('result_file_merge_onehot_dict.txt','a')

    result_file_out.write('Dataset %d :\n'%(dataset_num))

    training_indice, training_label, validation_indice, validation_label = split_training_validation(train_Y)
    #pdb.set_trace()
    '''
    cnn_train  = []
    cnn_validation = []
    seq_data = training_data["seq"][0]
    #pdb.set_trace()
    #seq_data = training_data["seq"][2]
    
    seq_train = seq_data[training_indice]
    seq_validation = seq_data[validation_indice] 
    #struct_data = training_data["seq"][1]
    #struct_data = training_data["seq"][3]
    #struct_train = struct_data[training_indice]
    #struct_validation = struct_data[validation_indice] 
    cnn_train.append(seq_train)
    #cnn_train.append(struct_train)
    cnn_validation.append(seq_validation)
    #cnn_validation.append(struct_validation)        
    #seq_net =  get_cnn_network(struct_onehot=False,struct_dict_num = 5)
    seq_net = get_cnn_network_NR()
    #seq_net_attention =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=True,bw_add=False,attention=True)
    #seq_data = []
    #pdb.set_trace()
    '''       
    y, encoder = preprocess_labels(training_label,categorical=True)
    val_y, encoder = preprocess_labels(validation_label, encoder = encoder,categorical=True)
    total_hid = 64 #seq_hid + struct_hid
    total_hid_attention = seq_hid + struct_hid
    #model = run_network_new(seq_net, total_hid, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    #model_attention = run_network_new(seq_net_attention, total_hid_attention, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    
    test_data = load_data_file('dataset_differ_mer/Test_all')
    true_y = test_data["Y"].copy()
    #testing = []
    #pdb.set_trace()
    #testing.append(test_data["seq"][0])
    #testing.append(test_data["seq"][1])
    #testing.append(test_data["seq"][3])
    #pred = model.predict_proba(testing)
    #pred_attention = model_attention.predict_proba(testing)
    #auc = roc_auc_score(true_y, pred[:,1])
    #auc_attention = roc_auc_score(true_y, pred_attention[:,1])
    #print 'Seq Dict&Struct Dict 3mer Test AUC: %.4f'%(auc)
    #print 'Seq Dict&Struct Dict 3mer Attention Test AUC: %.4f'%(auc_attention)
    #pdb.set_trace()
    #result_file_out.write('Seq One-Hot Test AUC: %.4f\n'%(auc))
    #result_file_out.write('Seq One-Hot Attention Test AUC: %.4f\n'%(auc_attention))
    #model.save(os.path.join(model_dir,'model.pkl'))
    '''
    ###  struct_dict 4 mer
    cnn_train = []
    cnn_validation = []
    struct_data = training_data["seq"][1]
    struct_train = struct_data[training_indice]
    struct_validation = struct_data[validation_indice]
    cnn_train.append(seq_train)
    cnn_train.append(struct_train)
    cnn_validation.append(seq_validation)
    cnn_validation.append(struct_validation)
    seq_net =  get_cnn_network(struct_onehot=False,struct_dict_num = 4,merge = True)
    seq_net_attention =  get_cnn_network(struct_onehot=False,struct_dict_num = 4,merge = True,attention=True)
    total_hid = seq_hid + struct_hid
    model = run_network_new(seq_net, total_hid, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_attention = run_network_new(seq_net_attention, total_hid, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    testing = []
    testing.append(test_data["seq"][0])
    #testing.append(test_data["seq"][1])
    testing.append(test_data["seq"][1])
    pred = model.predict_proba(testing)
    pred_attention = model_attention.predict_proba(testing)
    auc = roc_auc_score(true_y, pred[:,1])
    auc_attention = roc_auc_score(true_y, pred_attention[:,1])
    print 'Merge One hot Dict Test AUC: %.4f'%(auc)
    print 'Merge One hot Dict Attention Test AUC: %.4f'%(auc_attention)
    result_file_out.write('Merge One hot Dict Test AUC: %.4f\n'%(auc))
    result_file_out.write('Merge One hot Dict Attention Test AUC: %.4f\n'%(auc_attention))

    ###  struct_onehot
    cnn_train = []
    cnn_validation = []
    struct_data = training_data["seq"][1]
    struct_train = struct_data[training_indice]
    struct_validation = struct_data[validation_indice]
    #cnn_train.append(seq_train)
    cnn_train.append(struct_train)
    #cnn_validation.append(seq_validation)
    cnn_validation.append(struct_validation)
    seq_net =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False)
    seq_net_attention =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,attention=True)
    model = run_network_new(seq_net, total_hid, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_attention = run_network_new(seq_net_attention, total_hid, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    testing = []
    #testing.append(test_data["seq"][2])
    testing.append(test_data["seq"][1])
    #testing.append(test_data["seq"][4])
    pred = model.predict_proba(testing)
    pred_attention = model_attention.predict_proba(testing)
    auc = roc_auc_score(true_y, pred[:,1])
    auc_attention = roc_auc_score(true_y, pred_attention[:,1])
    print 'Only DictCNN Test AUC: %.4f'%(auc)
    print 'Only DictCNN Attention Test AUC: %.4f'%(auc_attention)
    result_file_out.write('Only DictCNN Test AUC: %.4f\n'%(auc))
    result_file_out.write('Only DictCNN Attention Test AUC: %.4f\n'%(auc_attention))
    
    ###  struct_onehot
    cnn_train = []
    cnn_validation = []
    struct_data = training_data["seq"][2]
    struct_train = struct_data[training_indice]
    struct_validation = struct_data[validation_indice]
    cnn_train.append(seq_train)
    cnn_train.append(struct_train)
    cnn_validation.append(seq_validation)
    cnn_validation.append(struct_validation)
    seq_net =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = True,struct_dict=True)
    seq_net_attention =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = True,struct_dict=True,attention=True)
    model = run_network_new(seq_net, total_hid, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_attention = run_network_new(seq_net_attention, total_hid, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    testing = []
    testing.append(test_data["seq"][0])
    testing.append(test_data["seq"][2])
    #testing.append(test_data["seq"][4])
    pred = model.predict_proba(testing)
    pred_attention = model_attention.predict_proba(testing)
    auc = roc_auc_score(true_y, pred[:,1])
    auc_attention = roc_auc_score(true_y, pred_attention[:,1])
    print 'Only Onhot Struct Test AUC: %.4f'%(auc)
    print 'Only Onhot Struct Attention Test AUC: %.4f'%(auc_attention)
    result_file_out.write('Only OneHot Struct Test AUC: %.4f\n'%(auc))
    result_file_out.write('Only OneHot Struct Attention Test AUC: %.4f\n'%(auc_attention))
    '''
    '''
    ###  addmodify_cnndict
    print 'addmodify_cnndict...'
    cnn_train = []
    cnn_validation = []
    seq_data = training_data["seq"][4]

    seq_train = seq_data[training_indice]
    seq_validation = seq_data[validation_indice]

    #struct_data = training_data["seq"][3]
    #struct_train = struct_data[training_indice]
    #struct_validation = struct_data[validation_indice]
    cnn_train.append(seq_train)
    #cnn_train.append(struct_train)
    cnn_validation.append(seq_validation)
    #cnn_validation.append(struct_validation)
    seq_net =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=False,bw_add=True)
    seq_net_3win =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=False,bw_add=True,win_size='3')
    seq_net_5win =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=False,bw_add=True,win_size='5')
    seq_net_3_10win =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=False,bw_add=True,win_size='3_10')
    seq_net_3_5_10win =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=False,bw_add=True,win_size='3_5_10')
    seq_net_notLSTM =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=False,bw_add=True,use_LSTM=True)
    seq_net_attention =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=False,bw_add=True,attention=True)
    seq_net_attention_LSTM =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=False,bw_add=True,attention=True,use_LSTM=True)
    seq_net_attention_use_two_data =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=False,bw_add=True,attention=True,use_two_data=True)
    seq_net_attention_use_two_data_3_10win =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=False,bw_add=True,attention=True,use_two_data=True,win_size='3_10')
    seq_net_attention_use_two_data_3_5_10win =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=False,bw_add=True,attention=True,use_two_data=True,win_size='3_5_10')
    seq_net_attention_use_two_data_LSTM =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=False,bw_add=True,attention=True,use_two_data=True,use_LSTM=True)
    model = run_network_new(seq_net, total_hid, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_3win = run_network_new(seq_net_3win, total_hid, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_5win = run_network_new(seq_net_5win, total_hid, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    #model_3_10win = run_network_new(seq_net_3_10win, total_hid, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_3_10win = run_network_new(seq_net_3_10win, total_hid, [cnn_train[0],cnn_train[0]], y, validation = [cnn_validation[0],cnn_validation[0]], val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_3_5_10win = run_network_new(seq_net_3_5_10win, total_hid, [cnn_train[0],cnn_train[0],cnn_train[0]], y, validation = [cnn_validation[0],cnn_validation[0],cnn_validation[0]], val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_notLSTM = run_network_new(seq_net_notLSTM, total_hid, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_attention = run_network_new(seq_net_attention, total_hid, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_attention_LSTM = run_network_new(seq_net_attention_LSTM, total_hid, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_attention_use_two_data = run_network_new(seq_net_attention_use_two_data, total_hid, [cnn_train[0],cnn_train[0]], y, validation = [cnn_validation[0],cnn_validation[0]], val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_attention_use_two_data_3_10win = run_network_new(seq_net_attention_use_two_data_3_10win, total_hid, [cnn_train[0],cnn_train[0],cnn_train[0],cnn_train[0]], y, validation = [cnn_validation[0],cnn_validation[0],cnn_validation[0],cnn_validation[0]], val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_attention_use_two_data_3_5_10win = run_network_new(seq_net_attention_use_two_data_3_5_10win, total_hid, [cnn_train[0],cnn_train[0],cnn_train[0],cnn_train[0],cnn_train[0],cnn_train[0]], y, validation = [cnn_validation[0],cnn_validation[0],cnn_validation[0],cnn_validation[0],cnn_validation[0],cnn_validation[0]], val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_attention_use_two_data_LSTM = run_network_new(seq_net_attention_use_two_data_LSTM, total_hid, [cnn_train[0],cnn_train[0]], y, validation = [cnn_validation[0],cnn_validation[0]], val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    testing = []
    testing.append(test_data["seq"][4])
    #testing.append(test_data["seq"][2])
    #testing.append(test_data["seq"][4])
    pred = model.predict_proba(testing)
    pred_3win = model_3win.predict_proba(testing)
    pred_5win = model_5win.predict_proba(testing)
    pred_3_10win = model_3_10win.predict_proba([testing[0],testing[0]])
    pred_3_5_10win = model_3_5_10win.predict_proba([testing[0],testing[0],testing[0]])
    pred_notLSTM = model_notLSTM.predict_proba(testing)
    pred_attention = model_attention.predict_proba(testing)
    pred_attention_LSTM = model_attention_LSTM.predict_proba(testing)
    pred_attention_use_two_data = model_attention_use_two_data.predict_proba([testing[0],testing[0]])
    pred_attention_use_two_data_3_10win = model_attention_use_two_data_3_10win.predict_proba([testing[0],testing[0],testing[0],testing[0]])
    pred_attention_use_two_data_3_5_10win = model_attention_use_two_data_3_5_10win.predict_proba([testing[0],testing[0],testing[0],testing[0],testing[0],testing[0]])
    pred_attention_use_two_data_LSTM = model_attention_use_two_data_LSTM.predict_proba([testing[0],testing[0]])
    #pdb.set_trace()
    auc = roc_auc_score(true_y, pred[:,1])
    auc_3win = roc_auc_score(true_y, pred_3win[:,1])
    auc_5win = roc_auc_score(true_y, pred_5win[:,1])
    auc_3_10win = roc_auc_score(true_y, pred_3_10win[:,1])
    auc_3_5_10win = roc_auc_score(true_y, pred_3_5_10win[:,1])
    auc_notLSTM = roc_auc_score(true_y, pred_notLSTM[:,1])
    auc_attention = roc_auc_score(true_y, pred_attention[:,1])
    auc_attention_LSTM = roc_auc_score(true_y, pred_attention_LSTM[:,1])
    auc_attention_use_two_data = roc_auc_score(true_y, pred_attention_use_two_data[:,1])
    auc_attention_use_two_data_3_10win = roc_auc_score(true_y, pred_attention_use_two_data_3_10win[:,1])
    auc_attention_use_two_data_3_5_10win = roc_auc_score(true_y, pred_attention_use_two_data_3_5_10win[:,1])
    auc_attention_use_two_data_LSTM = roc_auc_score(true_y, pred_attention_use_two_data_LSTM[:,1])
    print 'Only CNNDict Add Modify Test AUC: %.4f'%(auc)
    print 'Only CNNDict Add Modify 3win Test AUC: %.4f'%(auc_3win)
    print 'Only CNNDict Add Modify 5win Test AUC: %.4f'%(auc_5win)
    print 'Only CNNDict Add Modify 3_10win Test AUC: %.4f'%(auc_3_10win)
    print 'Only CNNDict Add Modify 3_5_10win Test AUC: %.4f'%(auc_3_5_10win)
    print 'Only CNNDict Add Modify notLSTM Test AUC: %.4f'%(auc_notLSTM)
    print 'Only CNNDict Add Modify Attention Test AUC: %.4f'%(auc_attention)
    print 'Only CNNDict Add Modify Attention LSTM Test AUC: %.4f'%(auc_attention_LSTM)
    print 'Only CNNDict Add Modify Attention Use Two data Test AUC: %.4f'%(auc_attention_use_two_data)
    print 'Only CNNDict Add Modify Attention Use Two data 3_10win Test AUC: %.4f'%(auc_attention_use_two_data_3_10win)
    print 'Only CNNDict Add Modify Attention Use Two data 3_5_10win Test AUC: %.4f'%(auc_attention_use_two_data_3_5_10win)
    print 'Only CNNDict Add Modify Attention Use Two data LSTM Test AUC: %.4f'%(auc_attention_use_two_data_LSTM)
    result_file_out.write('Only CNNDict Add Modify Test AUC: %.4f\n'%(auc))
    result_file_out.write('Only CNNDict Add Modify 3win Test AUC: %.4f\n'%(auc_3win))
    result_file_out.write('Only CNNDict Add Modify 5win Test AUC: %.4f\n'%(auc_5win))
    result_file_out.write('Only CNNDict Add Modify 3_10win Test AUC: %.4f\n'%(auc_3_10win))
    result_file_out.write('Only CNNDict Add Modify 3_5_10win Test AUC: %.4f\n'%(auc_3_5_10win))
    result_file_out.write('Only CNNDict Add Modify notLSTM Test AUC: %.4f\n'%(auc_notLSTM))
    result_file_out.write('Only CNNDict Add Modify Att Test AUC: %.4f\n'%(auc_attention))
    result_file_out.write('Only CNNDict Add Modify Att LSTM Test AUC: %.4f\n'%(auc_attention_LSTM))
    result_file_out.write('Only CNNDict Add Modify Att Use Two data Test AUC: %.4f\n'%(auc_attention_use_two_data))
    result_file_out.write('Only CNNDict Add Modify Att 3_10win Use Two data Test AUC: %.4f\n'%(auc_attention_use_two_data_3_10win))
    result_file_out.write('Only CNNDict Add Modify Att 3_5_10win Use Two data Test AUC: %.4f\n'%(auc_attention_use_two_data_3_5_10win))
    result_file_out.write('Only CNNDict Add Modify Att Use Two data LSTM Test AUC: %.4f\n'%(auc_attention_use_two_data_LSTM))
    '''
    
    ### addmodify_onhot
    print 'addmodify_onhot...'
    cnn_train = []
    cnn_validation = []
    seq_data = training_data["seq"][3]

    seq_train = seq_data[training_indice]
    seq_validation = seq_data[validation_indice]

    #struct_data = training_data["seq"][3][:,:,-1].reshape(training_data["seq"][3].shape[0],training_data["seq"][3].shape[1],1)
    #struct_train = struct_data[training_indice]
    #struct_validation = struct_data[validation_indice]
    cnn_train.append(seq_train)
    #cnn_train.append(struct_train)
    cnn_validation.append(seq_validation)
    #cnn_validation.append(struct_validation)
    seq_net =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=True,bw_add=True)
    seq_net_notLSTM =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=True,bw_add=True,use_LSTM=True)
    seq_net_notLSTM_only_histone =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=True,bw_add=True,use_LSTM=True,only_histone=True)
    seq_net_notLSTM_one_data =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=True,bw_add=True,use_LSTM=True,one_data=True)
    seq_net_notLSTM_3win =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=True,bw_add=True,use_LSTM=True,win_size='3')
    seq_net_notLSTM_5win =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=True,bw_add=True,use_LSTM=True,win_size='5')
    seq_net_notLSTM_3_10win =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=True,bw_add=True,use_LSTM=True,win_size='3_10')
    seq_net_notLSTM_3_5_10win =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=True,bw_add=True,use_LSTM=True,win_size='3_5_10')
    seq_net_notLSTM_3_5_10win_one_data =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=True,bw_add=True,use_LSTM=True,one_data=True,win_size='3_5_10')
    seq_net_attention =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=True,bw_add=True,attention=True)
    seq_net_attention_LSTM =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=True,bw_add=True,attention=True,use_LSTM=True)
    seq_net_attention_use_two_data =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=True,bw_add=True,attention=True,use_two_data=True)
    seq_net_attention_use_two_data_LSTM =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=True,bw_add=True,attention=True,use_two_data=True,use_LSTM=True)
    model = run_network_new(seq_net, total_hid, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_notLSTM = run_network_new(seq_net_notLSTM, total_hid, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_notLSTM_one_data = run_network_new(seq_net_notLSTM_one_data, total_hid, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_notLSTM_3win = run_network_new(seq_net_notLSTM_3win, total_hid, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_notLSTM_5win = run_network_new(seq_net_notLSTM_5win, total_hid, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_notLSTM_3_10win = run_network_new(seq_net_notLSTM_3_10win, total_hid, [cnn_train[0],cnn_train[0]], y, validation = [cnn_validation[0],cnn_validation[0]], val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_notLSTM_3_5_10win = run_network_new(seq_net_notLSTM_3_5_10win, total_hid, [cnn_train[0],cnn_train[0],cnn_train[0]], y, validation = [cnn_validation[0],cnn_validation[0],cnn_validation[0]], val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_notLSTM_3_5_10win_one_data = run_network_new(seq_net_notLSTM_3_5_10win_one_data, total_hid, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_attention = run_network_new(seq_net_attention, total_hid, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_attention_LSTM = run_network_new(seq_net_attention_LSTM, total_hid, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_attention_use_two_data = run_network_new(seq_net_attention_use_two_data, total_hid, [cnn_train[0],cnn_train[0]], y, validation = [cnn_validation[0],cnn_validation[0]], val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    model_attention_use_two_data_LSTM = run_network_new(seq_net_attention_use_two_data_LSTM, total_hid, [cnn_train[0],cnn_train[0]], y, validation = [cnn_validation[0],cnn_validation[0]], val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    testing = []
    testing.append(test_data["seq"][3])
    #testing.append(test_data["seq"][3][:,:,-1].reshape(test_data["seq"][3].shape[0],test_data["seq"][3].shape[1],1))
    #testing.append(test_data["seq"][4])
    #pdb.set_trace()
    pred = model.predict_proba(testing)
    pred_notLSTM = model_notLSTM.predict_proba(testing)
    pred_notLSTM_one_data = model_notLSTM_one_data.predict_proba(testing)
    pred_notLSTM_3win = model_notLSTM_3win.predict_proba(testing)
    pred_notLSTM_5win = model_notLSTM_5win.predict_proba(testing)
    pred_notLSTM_3_10win = model_notLSTM_3_10win.predict_proba([testing[0],testing[0]])
    pred_notLSTM_3_5_10win = model_notLSTM_3_5_10win.predict_proba([testing[0],testing[0],testing[0]])
    pred_notLSTM_3_5_10win_one_data = model_notLSTM_3_5_10win_one_data.predict_proba(testing)
    pred_attention = model_attention.predict_proba(testing)
    pred_attention_LSTM = model_attention_LSTM.predict_proba(testing)
    pred_attention_use_two_data = model_attention_use_two_data.predict_proba([testing[0],testing[0]])
    pred_attention_use_two_data_LSTM = model_attention_use_two_data_LSTM.predict_proba([testing[0],testing[0]])
    #pdb.set_trace()
    auc = roc_auc_score(true_y, pred[:,1])
    auc_notLSTM = roc_auc_score(true_y, pred_notLSTM[:,1])
    auc_notLSTM_one_data = roc_auc_score(true_y, pred_notLSTM_one_data[:,1])
    auc_notLSTM_3win = roc_auc_score(true_y, pred_notLSTM_3win[:,1])
    auc_notLSTM_5win = roc_auc_score(true_y, pred_notLSTM_5win[:,1])
    auc_notLSTM_3_10win = roc_auc_score(true_y, pred_notLSTM_3_10win[:,1])
    auc_notLSTM_3_5_10win = roc_auc_score(true_y, pred_notLSTM_3_5_10win[:,1])
    auc_notLSTM_3_5_10win_one_data = roc_auc_score(true_y, pred_notLSTM_3_5_10win_one_data[:,1])
    auc_attention = roc_auc_score(true_y, pred_attention[:,1])
    auc_attention_LSTM = roc_auc_score(true_y, pred_attention_LSTM[:,1])
    auc_attention_use_two_data = roc_auc_score(true_y, pred_attention_use_two_data[:,1])
    auc_attention_use_two_data_LSTM = roc_auc_score(true_y, pred_attention_use_two_data_LSTM[:,1])
    print 'Only Onhot Add Modify Test AUC: %.4f'%(auc)
    print 'Only Onhot Add Modify notLSTM Test AUC: %.4f'%(auc_notLSTM)
    print 'Only Onhot Add Modify notLSTM One_data Test AUC: %.4f'%(auc_notLSTM_one_data)
    print 'Only Onhot Add Modify notLSTM 3win Test AUC: %.4f'%(auc_notLSTM_3win)
    print 'Only Onhot Add Modify notLSTM 5win Test AUC: %.4f'%(auc_notLSTM_5win)
    print 'Only Onhot Add Modify notLSTM 3_10win Test AUC: %.4f'%(auc_notLSTM_3_10win)
    print 'Only Onhot Add Modify notLSTM 3_5_10win Test AUC: %.4f'%(auc_notLSTM_3_5_10win)
    print 'Only Onhot Add Modify notLSTM 3_5_10win One_data Test AUC: %.4f'%(auc_notLSTM_3_5_10win_one_data)
    print 'Only Onhot Add Modify Attention Test AUC: %.4f'%(auc_attention)
    print 'Only Onhot Add Modify Attention LSTM Test AUC: %.4f'%(auc_attention_LSTM)
    print 'Only Onhot Add Modify Attention Use Two Data Test AUC: %.4f'%(auc_attention_use_two_data)
    print 'Only Onhot Add Modify Attention Use Two Data LSTM Test AUC: %.4f'%(auc_attention_use_two_data_LSTM)
    result_file_out.write('Only OneHot Add Modify Test AUC: %.4f\n'%(auc))
    result_file_out.write('Only OneHot Add Modify notLSTM Test AUC: %.4f\n'%(auc_notLSTM))
    result_file_out.write('Only OneHot Add Modify notLSTM One_data Test AUC: %.4f\n'%(auc_notLSTM_one_data))
    result_file_out.write('Only OneHot Add Modify notLSTM 3win Test AUC: %.4f\n'%(auc_notLSTM_3win))
    result_file_out.write('Only OneHot Add Modify notLSTM 5win Test AUC: %.4f\n'%(auc_notLSTM_5win))
    result_file_out.write('Only OneHot Add Modify notLSTM 3_10win Test AUC: %.4f\n'%(auc_notLSTM_3_10win))
    result_file_out.write('Only OneHot Add Modify notLSTM 3_5_10win Test AUC: %.4f\n'%(auc_notLSTM_3_5_10win))
    result_file_out.write('Only OneHot Add Modify notLSTM 3_5_10win One_data Test AUC: %.4f\n'%(auc_notLSTM_3_5_10win_one_data))
    result_file_out.write('Only OneHot Add Modify Attention Test AUC: %.4f\n'%(auc_attention))
    result_file_out.write('Only OneHot Add Modify Attention LSTM Test AUC: %.4f\n'%(auc_attention_LSTM))
    result_file_out.write('Only OneHot Add Modify Attention Use Two Data Test AUC: %.4f\n'%(auc_attention_use_two_data))
    result_file_out.write('Only OneHot Add Modify Attention Use Two Data LSTM Test AUC: %.4f\n'%(auc_attention_use_two_data_LSTM))
    
    print 'addmodify_onhot...'
    cnn_train = []
    cnn_validation = []
    seq_data = training_data["seq"][6]

    seq_train = seq_data[training_indice]
    seq_validation = seq_data[validation_indice]

    #struct_data = training_data["seq"][3][:,:,-1].reshape(training_data["seq"][3].shape[0],training_data["seq"][3].shape[1],1)
    #struct_train = struct_data[training_indice]
    #struct_validation = struct_data[validation_indice]
    cnn_train.append(seq_train)
    #cnn_train.append(struct_train)
    cnn_validation.append(seq_validation)

    seq_net_notLSTM_only_histone =  get_cnn_network(struct_onehot=True,struct_dict_num = 4,merge = False,struct_dict=True,onehot_only=True,bw_add=True,use_LSTM=True,only_histone=True)
    model_notLSTM_only_histone = run_network_new(seq_net_notLSTM_only_histone, total_hid, cnn_train, y, validation = cnn_validation, val_y = val_y, batch_size=batch_size, nb_epoch = nb_epoch)
    testing = []
    testing.append(test_data["seq"][6])
    pred_not_LSTM_only_histone = model_notLSTM_only_histone.predict_proba(testing)
        
    auc_notLSTM_only_histone = roc_auc_score(true_y, pred_not_LSTM_only_histone[:,1])
    only_histone_labels = transfer_label_from_prob(pred_not_LSTM_only_histone[:,1])
    print 'Only Onhot Add Modify notLSTM only histone Test AUC: %.4f'%(auc_notLSTM_only_histone)
    result_file_out.write('Only OneHot Add Modify notLSTM only histone Test AUC: %.4f\n'%(auc_notLSTM_only_histone))

    #rf_train = training_data["seq"][5]
    #rf_test = test_data["seq"][5]
    #pred_label,fitt = run_randomforest_classifier(rf_train, train_Y, rf_test)
    #auc_rf = roc_auc_score(true_y,pred_label)
    #print 'RF Test AUC: %.4f'%(auc_rf)
    #result_file_out.write('RF Test AUC: %.4f\n'%(auc_rf))

    result_file_out.close()

def test_ideeps(data_file, model_dir, outfile='prediction.txt', onlytest = False):
    test_data = load_data_file(data_file, onlytest= onlytest)
    pdb.set_trace()
    print len(test_data)
    true_y = test_data["Y"].copy()
    
    print 'predicting'    
    
    testing = test_data["seq"] #it includes one-hot encoding sequence and structure
    #structure = test_data["structure"]
    model = load_model(os.path.join(model_dir,'model.pkl')) 
    pdb.set_trace()
    pred = model.predict_proba(testing)
    #=calculate_performace(test_num, pred_y,  labels)
    auc = roc_auc_score(true_y, pred[:,1])
    print 'Test AUC: %.4f'%(auc)
    
    fw = open(outfile, 'w')
    myprob = "\n".join(map(str, predictions[:, 1]))
    #fw.write(mylabel + '\n')
    fw.write(myprob)
    fw.close()
    
def run_ideeps(parser):
    data_file = parser.data_file
    out_file = parser.out_file
    train = parser.train
    model_dir = parser.model_dir
    predict = parser.predict
    batch_size = parser.batch_size
    n_epochs = parser.n_epochs
    data_set_num = parser.dataset_num
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    if predict:
        train = False
 
    if train:
        print 'model training'
        train_ideeps(data_file, model_dir, batch_size= batch_size, nb_epoch = n_epochs,dataset_num= data_set_num)
    else:
        print 'model prediction'
        test_ideeps(data_file, model_dir, outfile = out_file)
        

def parse_arguments(parser):
    parser.add_argument('--data_file', type=str, metavar='<data_file>', help='the sequence file used for training, it contains sequences and label (0, 1) in each head of sequence.')
    parser.add_argument('--train', type=bool, default=True, help='use this option for training model')
    parser.add_argument('--model_dir', type=str, default='model', help='The directory to save the trained models for future prediction')
    parser.add_argument('--predict', type=bool, default=False,  help='Predicting the RNA-protein binding sites for your input sequences, if using train, then it will be False')
    parser.add_argument('--out_file', type=str, default='prediction.txt', help='The output file used to store the prediction probability of testing data')
    parser.add_argument('--batch_size', type=int, default=200, help='The size of a single mini-batch (default value: 50)')
    parser.add_argument('--n_epochs', type=int, default=10, help='The number of training epochs (default value: 30)')
    parser.add_argument('--dataset_num', type=int, default=0, help='The number of dataset (default value: 10)')
    args = parser.parse_args()
    return args

         
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    run_ideeps(args)


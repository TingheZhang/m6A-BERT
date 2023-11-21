import pandas as pd
import os, csv
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split
import pickle as pl
import random,argparse
from os.path import exists

def seq2kmer(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    seq=str(seq)
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers


def get_fasta(path, file):
    seq = []

    record = list(SeqIO.parse(path + file, 'fasta'))

    for gene in record:
        seq.append(gene.seq.upper())  # # lower to upper

    return seq

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--do_val", action="store_true", help="Whether to split validation data.")
    parser.add_argument("--kmer", required=True, type=int, help="how many mer should be generated")
    parser.add_argument("--extend_len", required=True, type=int, help="generate data for which extended length")
    parser.add_argument("--task", required=True, type=str, help="data should be generated for which task, choose from pretrain or finetune")
    parser.add_argument("--data_dir", required=True, type=str,help="load data from which dir")
    parser.add_argument("--save_dir", required=True, type=str,help="save data to which dir")
    parser.add_argument("--seed", type=int,default=52,help="random seed")

    args = parser.parse_args()

    val=args.do_val

    kmer = args.kmer
    length=args.extend_len
    task=args.task

    if task=='finetune':

        data_path=args.data_dir

        file_pos='pos.fa'
        file_neg='neg.fa'

        seq_pos=get_fasta(data_path,file_pos)
        seq_neg=get_fasta(data_path,file_neg)
        seq_kmer_pos=list()
        label_kmer_pos=list()
        seq_kmer_neg=list()
        label_kmer_neg=list()

        random.seed(args.seed)
        random.shuffle(seq_neg)
        for i in range(len(seq_pos)):
            seq_kmer_pos.append(seq2kmer(seq_pos[i],kmer))
            label_kmer_pos.append(1)

        for j in range(len(seq_pos)):
            seq_kmer_neg.append(seq2kmer(seq_neg[j],kmer))
            label_kmer_neg.append(0)

        rand_seed=args.seed
        seq_kmer_pos_train,seq_kmer_pos_test,label_kmer_pos_train,label_kmer_pos_test=train_test_split(seq_kmer_pos,label_kmer_pos,random_state=rand_seed,test_size=0.2)
        seq_kmer_neg_train,seq_kmer_neg_test,label_kmer_neg_train,label_kmer_neg_test=train_test_split(seq_kmer_neg,label_kmer_neg,random_state=rand_seed,test_size=0.2)
        if val==True:
            seq_kmer_pos_train,seq_kmer_pos_val,label_kmer_pos_train,label_kmer_pos_val=train_test_split(seq_kmer_pos_train, label_kmer_pos_train, random_state=rand_seed, test_size=0.1)
            seq_kmer_neg_train,seq_kmer_neg_val,label_kmer_neg_train,label_kmer_neg_val=train_test_split(seq_kmer_neg_train, label_kmer_neg_train, random_state=rand_seed, test_size=0.1)
            val_x=seq_kmer_pos_val+seq_kmer_neg_val
            val_y=label_kmer_pos_val+label_kmer_neg_val

        train_x=seq_kmer_pos_train+seq_kmer_neg_train
        train_y=label_kmer_pos_train+label_kmer_neg_train
        test_x=seq_kmer_pos_test+seq_kmer_neg_test
        test_y=label_kmer_pos_test+label_kmer_neg_test

        # ind_train_shuffle=range(len(train_x))
        # random.shuffle(ind_train_shuffle)
        np.random.seed(rand_seed)
        ind_train_shuffle = np.random.choice(range(len(train_x)), replace=False, size=len(train_x))

        train_x=np.array(train_x)
        train_y=np.array(train_y)

        output_columns = ['seq','label']
        data = zip(train_x[ind_train_shuffle],train_y[ind_train_shuffle])
        save_path = args.save_dir

        if exists(save_path)==False:
            os.mkdir(save_path)

        with open(save_path+'train.tsv', 'w', newline='') as f_output:
            tsv_output = csv.writer(f_output, delimiter='\t')
            tsv_output.writerow(output_columns)
            for seq_kmer, label_kmer in data:
                tsv_output.writerow([seq_kmer, label_kmer])



        if val==True:

            data = zip(test_x, test_y)

            with open(save_path + 'test.tsv', 'w', newline='') as f_output:
                tsv_output = csv.writer(f_output, delimiter='\t')
                tsv_output.writerow(output_columns)
                for seq_kmer, label_kmer in data:
                    tsv_output.writerow([seq_kmer, label_kmer])

            data = zip(val_x,val_y)

            with open(save_path+'dev.tsv', 'w', newline='') as f_output:
                tsv_output = csv.writer(f_output, delimiter='\t')
                tsv_output.writerow(output_columns)
                for seq_kmer, label_kmer in data:
                    tsv_output.writerow([seq_kmer, label_kmer])

        else:
            data = zip(test_x, test_y)

            with open(save_path + 'dev.tsv', 'w', newline='') as f_output:
                tsv_output = csv.writer(f_output, delimiter='\t')
                tsv_output.writerow(output_columns)
                for seq_kmer, label_kmer in data:
                    tsv_output.writerow([seq_kmer, label_kmer])

    if task=='pretrain':

        data_path = args.data_dir

        seq = get_fasta(data_path, 'pretrain.fa')

        seq_kmer=list()
        for i in range(len(seq)):
            seq_kmer.append(seq2kmer(seq[i].back_transcribe(),kmer))

        save_path = args.save_dir
        if exists(save_path + '/pre/') == False:
            os.mkdir(save_path + '/pre/')

        if val:
            seq_kmer_train, seq_kmer_val = train_test_split(
                seq_kmer, random_state=rand_seed, test_size=0.1)

            with open(save_path + '/pre/' + str(kmer) + '_train.txt', 'w') as f:
                for s in seq_kmer_train:
                    f.writelines(s)
                    f.writelines('\n')

            with open(save_path+'/pre/'+str(kmer)+'_val.txt', 'w') as f:
                for s in seq_kmer_val:
                    f.writelines(s)
                    f.writelines('\n')
        else:
            with open(save_path+'/pre/'+str(kmer)+'.txt', 'w') as f:
                for s in seq_kmer:
                    f.writelines(s)
                    f.writelines('\n')

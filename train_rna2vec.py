# python3
# -*- coding:utf-8 -*-

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file：PycharmProject-PyCharm-train_rna2vec.py
@time:2021/3/28 9:57 
"""
import argparse
import os,sys
from collections import Counter
import glob
import random
import string
import re
import resource
import logbook
from logbook.compat import redirect_logging
import configargparse
import numpy as np
import arrow
from Bio import SeqIO
import resource
from gensim.models import word2vec

sys.path.extend(['.', '..'])

class Tee(object):
    def __init__(self, fptr):
        self.file = fptr
    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self
    def __exit__(self, exception_type, exception_value, traceback):
        sys.stdout = self.stdout
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

class Fasta_Sequence_cut:
    def __init__(self,fasta_files,Min_SeqLength,Max_SeqLength,kmer_low,kmer_high):
        self.logger = logbook.Logger(self.__class__.__name__)
        self.fasta_files = fasta_files
        self.epochs = args.epochs
        self.Min_SeqLength = Min_SeqLength
        self.Max_SeqLength = Max_SeqLength
        self.kmer_low = kmer_low
        self.kmer_high = kmer_high
        self.kmer_len_counter = Counter()
        self.nb_kmers = 0
        self.random_num = np.random.RandomState(123)
        self.fasta_seq = 0
        self.cut_seq = 0
        self.cut_seq_lengthlist=[]
        self.iter_count = 0

    def SeqGenerator_fastahandle(self):
        for curr_epoch in range(self.epochs):
            for fasta_file in self.fasta_files:
                with open(fasta_file) as fasta:
                    self.logger.info('Opened file: {}'.format(fasta_file))
                    self.logger.info('Memory usage: {} MB'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1E6))
                    self.logger.info('Current epoch: {} / {}'.format(curr_epoch + 1, self.epochs))
                    yield fasta

    def SeqGenerator(self):
        for fasta_handle in self.SeqGenerator_fastahandle():
            for seq_record in SeqIO.parse(fasta_handle,'fasta'):
                self.fasta_seq +=1
                whole_seq = seq_record.seq
                self.logger.info('Whole fasta seqlen:{}'.format(len(whole_seq)))
                cut_seq_stat = 0
                while cut_seq_stat < len(whole_seq):
                    self.cut_seq +=1
                    seqlen = random.randint(self.Min_SeqLength,self.Max_SeqLength)
                    #seqlen = len(seq_record.seq)     #miRna修改
                    segment = seq_record.seq[cut_seq_stat: cut_seq_stat+seqlen]
                    cut_seq_stat += seqlen
                    self.logger.debug('Cut seq len:{}'.format(len(segment)))
                    self.cut_seq_lengthlist.append(len(segment))
                    yield segment

    def SlidingKmerFragmenter(self,seq):
        return [seq[i: i + self.random_num.randint(self.kmer_low, self.kmer_high + 1)] for i in range(len(seq) - self.kmer_high + 1)]

    def Kmer_Histogram(self,seq):
        for kmer in seq:
            self.kmer_len_counter[len(kmer)] += 1
            self.nb_kmers += 1

    def Kmer_stat(self, fptr):
        for kmer_len in sorted(self.kmer_len_counter.keys()):
            self.logger.info('Percent of {:2d}-mers: {:3.1f}% ({})'.format(
                kmer_len,
                100.0 * self.kmer_len_counter[kmer_len] / self.nb_kmers,
                self.kmer_len_counter[kmer_len]))
        total_bps = sum([l * c for l, c in self.kmer_len_counter.items()])
        self.logger.info('Number of base-pairs: {}'.format(total_bps))

    def sequence_stat(self,fptr):
        self.logger.info('Number of fasta sequence: {}'.format(self.fasta_seq))
        self.logger.info('After cut Number of sequence: {}'.format(self.cut_seq))
        np.savez(args.out_dir+'/cut_seq_length.npz',Cut_Lenght=self.cut_seq_lengthlist)

    def __iter__(self):
        self.iter_count += 1
        for seq in self.SeqGenerator():
            seq = seq.upper()
            if True and self.random_num.rand() < 0.5:
                seq =  seq.reverse_complement()

            acgt_seq_split = list(filter(bool,re.split(r'[^ACGTacgt]+', str(seq))))
            for acgt_seq in acgt_seq_split:
                kmer_seq = self.SlidingKmerFragmenter(acgt_seq)
                if len(kmer_seq) > 0:
                    if self.iter_count ==1:
                        self.Kmer_Histogram(kmer_seq)
                    yield kmer_seq


class Learner:
    def __init__(self,outputfile,context_halfsize, gensim_iters, vec_dim):
        self.logger = logbook.Logger(self.__class__.__name__)
        self.model = None
        self.outputfile = outputfile
        # 三个word2vec 参数
        self.context_halfsize =context_halfsize
        self.gensim_iters = gensim_iters
        self.vec_dim =vec_dim
        self.use_skipgram = 1
        # 日志打印信息
        self.logger.info('Context window half size: {}'.format(self.context_halfsize))
        self.logger.info('Use skipgram: {}'.format(self.use_skipgram))
        self.logger.info('gensim_iters: {}'.format(self.gensim_iters))
        self.logger.info('vec_dim: {}'.format(self.vec_dim))

    def train(self,kmer_seq_generator):
        self.model = word2vec.Word2Vec(
            sentences=kmer_seq_generator,
            size=self.vec_dim,
            window=self.context_halfsize,
            min_count=2,
            workers=4,
            sg=self.use_skipgram,
            iter=self.gensim_iters)

    def write_vec(self):
        out_filename = '{}.w2v'.format(self.outputfile)
        self.model.wv.save_word2vec_format(out_filename, binary=False)


if __name__ == '__main__':
    argp = configargparse.get_argument_parser()
    argp.add('-c', is_config_file=True, help='config file path')
    argp.add_argument('--inputs', help='FASTA files', nargs='+', required=True)
    argp.add_argument('--k-low', help='k-mer start range (inclusive)', type=int, default=5)
    argp.add_argument('--k-high', help='k-mer end range (inclusive)', type=int, default=5)

    argp.add_argument('--Min_seq', help='cut sequence min length', type=int, default=200)
    argp.add_argument('--Max_seq', help='cut sequence max length', type=int, default=300)
    argp.add_argument('--vec_dim', help='vector dimension', type=int, default=12)
    argp.add_argument('--context', help='half size of context window (the total size is 2*c+1)', type=int, default=4)
    argp.add_argument('--epochs', help='number of epochs', type=int, default=1)
    argp.add_argument('--gensim_iters', help="gensim's internal iterations", type=int, default=1)
    argp.add_argument('--out_dir', help="output directory", default='../dataset/dna2vec/results')
    args = argp.parse_args()

    '''
    step1: 
    1\输入文件，fasta文件
    参数：args.inputs
    '''
    inputs = [glob.glob(s)[0] for s in args.inputs]
    print(inputs)
    # inputs = ['/home/jlk/Project/011-DNA2Vec/dna2vec-master/example_inputs/chrUn_KI270750v1.fa'....
    fileSizeSum = sum([os.stat(f).st_size for f in inputs])

    '''
    2\定义输出日志文件：
    参数：args.k_low,args.k_high,args.vec_dim,args.context,args.kmer_fragmenter
    '''
    out_dir = args.out_dir
    out_file_suffix_word = ''.join(random.SystemRandom().choice(string.ascii_lowercase +
                                string.ascii_uppercase + string.digits) for _ in range(3))
    out_file_suffix = 'k{}to{}-{}d-{}c-{}Mbp-{}-{}'.format(args.k_low,args.k_high,args.vec_dim,args.context,
                                            fileSizeSum * args.epochs, 'sliding',out_file_suffix_word)
    out_file = '{}/{}-{}-{}'.format(out_dir,'testRNA2vec',arrow.utcnow().format('YYYYMMDD-HHmm'),out_file_suffix,)
    out_txt_filename = '{}.txt'.format(out_file)
    print(out_txt_filename) # ./outdir//testDNA2vec-20210328-1317-k3to5-12d-4c-630543Mbp-sliding-jQo

    '''
    3\写到日志文件中
      work2vec 开始训练
    '''
    with open(out_txt_filename, 'w') as summary_fptr:
        with Tee(summary_fptr):
            logbook.StreamHandler(sys.stdout, level='DEBUG').push_application()
            redirect_logging()
            SeqGenerator = Fasta_Sequence_cut(fasta_files=inputs,
                                              Min_SeqLength=args.Min_seq,
                                              Max_SeqLength=args.Max_seq,
                                              kmer_low=args.k_low,
                                              kmer_high=args.k_high)
            learner = Learner(outputfile=out_file,
                              context_halfsize=args.context,
                              gensim_iters=args.gensim_iters,
                              vec_dim=args.vec_dim)

            learner.train(kmer_seq_generator=SeqGenerator)
            learner.write_vec()
            SeqGenerator.Kmer_stat(sys.stdout)
            SeqGenerator.sequence_stat(sys.stdout)

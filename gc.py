#!/usr/bin/python3
# encoding: utf-8
"""
@author: zhuhz
@contact: 875078992@qq.com
@file: gc.py
@time: 2019/8/3 20:07
"""


import argparse


def gc_cal(input):
    gc = {}
    gc_cnt = 0
    seq_len = 0
    with open(input, 'r') as f:
        for line in f:
            if line[0] == '>':
                if seq_len:
                    gc[header] = gc_cnt / seq_len
                    gc_cnt = seq_len = 0
                header = line.strip()[1:]
            else:
                seq = line.strip().upper()
                gc_cnt += seq.count('G') + seq.count('C')
                seq_len += len(seq)
        if seq_len:
            gc[header] = gc_cnt / seq_len
    return gc


if __name__ == '__main__':
    print('This short script calculates gc content for each sequence in the fasta file.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, help='path of the input fasta file')
    parser.add_argument('--output', '-o', type=str, help='path of the output file')
    args = parser.parse_args()

    gc_content = gc_cal(args.input)

    with open(args.output, 'w') as f:
        for k, v in gc_content.items():
            f.write('{:s}\t{:.2%}\n'.format(k, v))

    print('Done!')

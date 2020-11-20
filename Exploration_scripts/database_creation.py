import os
import sys
import json
import sqlite3

# open file, we have ref @ lines & id position
def dictionary_ref(data_load_ref ='./DNA_project/Data/BLAST/ref/id20.refs.txt', name_database = './DNA_project/Data/BLAST/ref/id20.refs.db'):

    conn = sqlite3.connect(name_database)
    c = conn.cursor()
    try:
        c.execute("DROP TABLE ref_mapping")
    except Exception as e:
        print (e)
    c.execute("CREATE TABLE ref_mapping (sseqid,rseq);")
    ref = open(data_load_ref,'r')
    strand_processing_counter = 0
    for i in ref:
        strand_processing_counter +=1
        split_input = i.strip().split('-')
        c.execute(" INSERT INTO ref_mapping (sseqid,rseq) VALUES ( ? , ? );",(str(strand_processing_counter),split_input[1]))
        test = False
        if test == True:
            if strand_processing_counter == 100:
                conn.commit()
                conn.close()
                break
    conn.commit()
    conn.close()

def dictionary_ref_noisy_strands(data_load_ref ='./DNA_project/Data/BLAST/noisy/NoisyStrands_noN.txt', name_database = './DNA_project/Data/BLAST/noisy/NoisyStrands_noN.db'):
    conn = sqlite3.connect(name_database)
    c = conn.cursor()
    try:
        c.execute("DROP TABLE noisy_mapping")
    except Exception as e:
        print (e)
    c.execute("CREATE TABLE noisy_mapping (nseqid,nseq);")
    ref = open(data_load_ref,'r')
    strand_processing_counter = 0
    for i in ref:
        strand_processing_counter +=1
        split_input = i.strip()
        c.execute(" INSERT INTO noisy_mapping (nseqid,nseq) VALUES ( ? , ? );",(str(strand_processing_counter),split_input))
        test = False
        if test == True:
            if strand_processing_counter == 100:
                conn.commit()
                conn.close()
                break
    conn.commit()
    conn.close()


def dictionary_ref_check(name_database = './DNA_project/Data/BLAST/ref/id20.refs.db',query=1):
    conn = sqlite3.connect(name_database)
    c = conn.cursor()
    c.execute('SELECT * FROM ref_mapping WHERE sseqid=?',  (str(query),) )
    return (c.fetchall()) # or c.fetchone() for a single value

def dictionary_ref_noisy_check(name_database = './DNA_project/Data/BLAST/ref/id20.refs.db',query=1):
    conn = sqlite3.connect(name_database)
    c = conn.cursor()
    c.execute('SELECT * FROM noisy_mapping WHERE nseqid=?',  (str(query),) )
    return (c.fetchall()) # or c.fetchone() for a single value

#%%
dictionary_ref()
#%%
dictionary_ref_check(query='90000')
#%%
dictionary_ref_noisy_strands()
#%%
dictionary_ref_noisy_check(name_database = './DNA_project/Data/BLAST/noisy/NoisyStrands_noN.db', query='60000')


#
# f = open('./DNA_project/Data/result/dev_dataset.txt','r')
# # num_lines = sum(1 for line in open('dev_dataset.txt.txt'))
# ref = open('./DNA_project/Data/BLAST/id20.refs.txt','r')
#
# sys.exit()
#
# strand_processing_counter = 0
# for i in f:
#     strand_processing_counter +=1
#     split_input = i.strip().split(',')
#     print (split_input)
#     ref_strand = split_input[-1]
#     out_strand = split_input[-2]
#     print (ref_strand)
#     print (out_strand)
#     if strand_processing_counter == 100:
#         sys.exit()

# 42
#   q:AGTGCAACAAGTCAATCCGTGTCGACTCGTGTGCGACGCTGTGCACACACATCTGCGTCGAGTCTCTCTGATGTCTCACTAGTCTGTGTGCTCGCGCTACACGACACTGAGACACTGTCTCGCGCAGAGCAATTGAATGCTTGCTTG
#   s:AGTGCCACAAGTCAATCCGTGTCGACTCGTGTGCGAAGCTGTGCAACCACATCTGCGTCGAGTCTATCTGATGTCTCACTAGTCTGTGTGCTCGCGCTTCACGACACTGAGACACTGTCTCGCGCAGAGCAATTGAATGCTTGCTTG
# ref:AGTGCAACAAGTCAATCCGTGTCGACTCGTGTGCGACGCTGTGCACACACATCTGCGTCGAGTCTCTCTGATGTCTCACTAGTCTGTGTGCTCGCGCTACACGACACTGAGACACTGTCTCGCGCAGAGCAATTGAATGCTTGCTTGCCG


#17 position 139828 query start :'1', query end:'145', subject start: '150', subject end:'1'
#query   :CGGCAAGCAAGCATTCAATTCTATATCAGTAGTACTCACAGACAGCTACTCGCATGTGCAGACTCTACTGACAGTGTATCTACAGTATACTCGTCGCGATGCTACTAGCAGCAGTGTGCTAGATGCTGTAACGGATTGACTTGTTGCACT
# subject:CGGCAAGCAAGCATTCAAT-----ATCAGTAGTACTCACAGACAGCTACTCGCATGTGCAGACTCTACTGACAGTGTATCTACAGTATACTCGTCGCGATGCTACTAGCAGCAGTGTGCTAGATGCTGTAACGGATTGACTTGTTGCACT
# in ref :AGTGCAACAAGTCAATCCGTTACAGCATCTAGCACACTGCTGCTAGTAGCATCGCGACGAGTATACTGTAGATACACTGTCAGTAGAGTCTGCACATGCGAGTAGCTGTCTGTGAGTACTACTGATATAGAATTGAATGCTTGCTTGCCG
# rev    :CGGCAAGCAAGCATTCAA

# 16 position 178752 startq '5', endq  '154', start s '1', end s '150'
# q:AGTGCAACAAGTCAATCCGTGTCGCGCGCTGATGATCAGTCTAGTAGCGCACGCGCTACACATGTGCGTGCGCGCTCTCACTATACATACTCTACGCTCTGTCGTATCGTGTGAGCGCGCAGTCGAGAGCAATTGAATGCTTGCTTGCCG
# s:AGTGCAACAAGTCAATCCGTGTCGCGCGCTGATGATCAGTCTAGTAGCGCACGCGCTACACATGTGCGTGCGCGCTCTCACTATACATACTCTACGCTCTGTCGTATCGTGTGAGCGCGCAGTCGAGAGCAATTGAATGCTTGCTTGCCG
# r:AGTGCAACAAGTCAATCCGTGTCGCGCGCTGATGATCAGTCTAGTAGCGCACGCGCTACACATGTGCGTGCGCGCTCTCACTATACATACTCTACGCTCTGTCGTATCGTGTGAGCGCGCAGTCGAGAGCAATTGAATGCTTGCTTGCCG

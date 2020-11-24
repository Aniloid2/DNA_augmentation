import os
import sys
import json
import sqlite3

import json
import matplotlib.pyplot as plt
import seaborn as sns
# open file, we have ref @ lines & id position

def ram_dictionary(data_load ='./DNA_project/Data/BLAST/ref/id20.refs.txt',create=True ):
    if create:
        ref = open(data_load,'r')
        mapping = {}
        strand_processing_counter = 0
        for i in ref:
            strand_processing_counter +=1
            split_input = i.strip().split('-')[1]
            mapping[str(strand_processing_counter)] = split_input
        with open('./DNA_project/Data/BLAST/ref/id20.refs.json', 'w') as fp:
            json.dump(mapping, fp)
        return mapping
    else:
        with open('./DNA_project/Data/BLAST/ref/id20.refs.json','r') as json_file:
            mapping= json.load(json_file)
        return mapping

#%%
def ram_dictionary_noisy(data_load ='./DNA_project/Data/BLAST/noisy/NoisyStrands_noN.txt',create=True ):
    if create:
        ref = open(data_load,'r')
        mapping = {}
        strand_processing_counter = 0
        for i in ref:
            strand_processing_counter +=1
            split_input = i.strip()
            mapping[str(strand_processing_counter)] = split_input
        with open('./DNA_project/Data/BLAST/noisy/NoisyStrands_noN.json', 'w') as fp:
            json.dump(mapping, fp)
        return mapping
    else:
        with open('./DNA_project/Data/BLAST/noisy/NoisyStrands_noN.json','r') as json_file:
            mapping= json.load(json_file)
        return mapping


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


def analytics_candidates(data_load = './DNA_project/Data/result/dev_dataset.txt', plot=1 ):
    f = open(data_load,'r')
    map = ram_dictionary(create=False)


    strand_processing_counter = 0
    dictionary = True
    names = ['inverted' ,'non_inverted' ,'inverted_150_length' ,'inverted_not_150_length' ,'non_inverted_150_length' ,'non_inverted_not_150_length' ]
    analytics = {'inverted':0,'non_inverted':0,'inverted_150_length':0,'inverted_not_150_length':0,'non_inverted_150_length':0,'non_inverted_not_150_length':0}
    error_profile = {'inverted':[],'non_inverted':[],'inverted_150_length':[],'inverted_not_150_length':[],'non_inverted_150_length':[],'non_inverted_not_150_length':[]}
    analytics_length_distribution = {'inverted':[],'non_inverted':[]}
    for i in f:
        strand_processing_counter +=1
        split_input = i.strip().split(',')
        sseqid = split_input[1]
        equality = split_input[2]
        sstart = split_input[8]
        send = split_input[9]
        # strand count analytics and % difference
        if int(sstart) > int(send):
            # it's inverted so skip
            analytics['inverted'] +=1
            error_profile['inverted'].append(float(equality))
            if int(sstart) == 150 and int(send)==1:
                analytics['inverted_150_length'] +=1
                error_profile['inverted_150_length'].append(float(equality))
            else:
                analytics['inverted_not_150_length'] +=1
                error_profile['inverted_not_150_length'].append(float(equality))
        else:
            analytics['non_inverted'] +=1
            error_profile['non_inverted'].append(float(equality))
            if int(sstart) == 1 and int(send)==150:
                analytics['non_inverted_150_length'] +=1
                error_profile['non_inverted_150_length'].append(float(equality))
            else:
                analytics['non_inverted_not_150_length'] +=1
                error_profile['non_inverted_not_150_length'].append(float(equality))

        subject_strand = split_input[-1]
        query_strand = split_input[-2]
        if dictionary:
            ref_strand = map[str(sseqid)]
        else:
            ref_strand = dictionary_ref_check(query=sseqid)[0][1]

        # strand length analytics
        if int(sstart) > int(send): # inverted
            diff = int(sstart) - int(send)
            analytics_length_distribution['inverted'].append(diff)

        else: #non inverted int(sstart) < int(send)
            diff = int(send) - int(sstart)
            analytics_length_distribution['non_inverted'].append(diff)





    if plot==0:
        f, axs = plt.subplots(6,1,figsize=(7,18))
        for i in range(len(names)):
            axs[i].hist(error_profile[names[i]], bins=180)
            axs[i].set_title(names[i])
            axs[i].set_xlabel('Percentage of identical matches')
            axs[i].set_ylabel('Number of strands')
        plt.tight_layout()
        plt.savefig('./DNA_project/Outputs/Analytics/distribution_dev_dataset.png')
    elif plot ==1:
        f, axs = plt.subplots(2,1,figsize=(7,13))
        for i in range(len(names[:2])):
            axs[i].hist(analytics_length_distribution[names[i]], bins=180)
            axs[i].set_title(names[i])
            axs[i].set_xlabel('Length strands')
            axs[i].set_ylabel('')
        plt.tight_layout()
        plt.savefig('./DNA_project/Outputs/Analytics/distribution_length_dataset.png')
    else:
        pass

    return {'analytics':analytics,'error_profile':error_profile,'analytics_length_distribution':analytics_length_distribution}


#%%
# analitics on non inverted not 150 lenght
# - their size distribution
# - wrap it up in a function
import pprint
a = analytics_candidates(plot = 1)
print ([i for i in a['analytics_length_distribution']['inverted'] if i > 150])
#%%
o = open('./DNA_project/Data/BLAST/NoisyStrands.txt','r')

#%%
map_ref = ram_dictionary(create=False)
map_noisy = ram_dictionary_noisy(create=False)
#%%
f = open('./DNA_project/Data/result/dev_dataset.txt','r')
# f = open('./DNA_project/Data/result/id20.small.result.txt','r')
#%%
save_directory = './DNA_project/Data/result/clean_noisy_dataset_dev.txt'
save =  open(save_directory,'w')
strand_processing_counter =0
save.write('qseqid,sseqid,equality,qstart,qend,sstart,send,ref_strand,noisy_strand \n',)
for i in f:
    strand_processing_counter +=1
    split_input = i.strip().split(',')
    firstid = split_input[0]
    sseqid = split_input[1]
    equality = split_input[2]
    qstart = split_input[6]
    qend = split_input[7]
    sstart = split_input[8]
    send = split_input[9]
    # print ('firstid',split_input[0],'secondid',split_input[1],'qstart',qstart,'qend',qend,'sstart',sstart,'send',send)
    # print ('q',split_input[-1])
    # print ('s',split_input[-2])
    # print ('r',map_ref[sseqid])
    # print ('n',map_noisy[firstid])
    save.write('{},{},{},{},{},{},{},{},{} \n'.format(firstid,sseqid,equality,qstart,qend,sstart,send,map_ref[sseqid],map_noisy[firstid]))
    # if strand_processing_counter == 4:
    #     save.close()
    #     sys.exit()

save.close()

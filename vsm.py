doc_num=143
voc_num=3298

word_dic=[]
word_doc=[]

import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
matrix=np.zeros([doc_num,voc_num])
path='new_input.txt'
count_doc=0


def tf_count(tf):
    tf=math.log(tf+1)
    return tf

with open(path,encoding='utf-8') as f:
    for line in f.readlines():
        words={}
        line=line.split('\n')[0].split(' ')

        for word in line:
            if word not in word_dic:
                word_dic.append(word)
                word_doc.append(0)
            if word in words:
                words[word]+=1
            else:
                words[word]=1


        norm_f=0
        for i in words:
            norm_f+=words[i]*words[i]
        norm_f=math.sqrt(norm_f)

        for word in words:
            matrix[count_doc][word_dic.index(word)]=tf_count(words[word]/float(norm_f))
            word_doc[word_dic.index(word)]+=1
        count_doc+=1
        # print(words)


for num,i in enumerate(word_doc):
    word_doc[num]=math.log(doc_num/float(i))
word_doc=np.array(word_doc)
for num,i in enumerate(matrix):
    matrix[num]=i*word_doc

np.save('matrix',matrix)

def _s(A,B,key):

    # print(A.shape)
    if key=='dot':
        return np.dot(A,B)
    elif key=='cos':
        A = np.array(A).reshape((1, voc_num))
        B = np.array(B).reshape((1, voc_num))
        return max(max(cosine_similarity(A,B)))

    elif key=='dice':
        A=np.array(A)
        B=np.array(B)
        num=A.dot(B)
        denom=A.dot(A)+B.dot(B)
        return 2*num/float(denom)
    elif key=='jaccard':
        A = np.array(A)
        B = np.array(B)
        num=A.dot(B)
        denom=A.dot(A)+B.dot(B)-num
        return num/float(denom)


def sd(key):
    sim_list={}
    for i in range(len(matrix)):
        for j in range(i,len(matrix)):
            if i==j:continue
            # sim_list.append(_s(matrix[i],matrix[j]))
            if key not in ['dot','dice','jaccard','cos']:
                raise ValueError('keyword Error')
            else:
                sim_list[str(i)+'+'+str(j)]=_s(matrix[i],matrix[j],key)
    sim_list=(sorted(sim_list.items(),key=lambda e:e[1],reverse=True))
    return sim_list

sim_list=sd('cos')
for i in range(20):
    print(sim_list[i])


# print(matrix)



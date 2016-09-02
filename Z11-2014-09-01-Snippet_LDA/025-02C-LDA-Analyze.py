import gensim
import pickle
import re
import nltk
import string
import numpy
import math
import os
import time
#################################################

# Get global tagger/chunker/regexp 
execfile("025-02Z-common.py")

#######################
#regexp
tA01_reg01_fn2activityid=re.compile(r"""a([-]?\d+).txt$""")


#################################################
# GLOBAL VAR: MAY NEED TO MODIFY
#################################################

tA02_fnamelist=pickle.load(open("025-02B-gensim-fnamelist","r"))
tA02_cpbow2=gensim.corpora.mmcorpus.MmCorpus("025-02B-gensim-cpbow2")
tA02_dict1=gensim.corpora.dictionary.Dictionary.load("025-02B-gensim-dict1")
tA02_lda=gensim.models.ldamodel.LdaModel.load("025-02B-gensim-lda")



#######################
# the RELATIVE FOLDER PATH TO the raw data and preprocessed data
tA02_dir1_pp = "002-textinput-plain"
tA02_dir2_raw = "002-textinput-PP"


tA01_regc02_ppdir =re.compile("^"+tA02_dir1_pp)




#################################################
#######################

def tY01_regexp01(str01):
  if (str01==""):
    return ""
  
  str02=tY01_regc02A_linebreak.sub(". ",str01)
  str02=tY01_regc01A_num.sub("\g<pf> ",str02)
  str02=tY01_regc02B_rmpunct.sub(" ",str02)
  str02=tY01_regc02E_rmspace1.sub(" ",str02)
  str02=tY01_regc02C_punctaddls.sub("\g<pf> \g<pu>",str02)
  str02=tY01_regc02D_punctaddrs.sub("\g<pu> \g<sf>",str02)
  str02=tY01_regc02F_rmspace2.sub("",str02)
  
  if (str02==""):
    return ""
  
  #setence tokenizer
  str03=nltk.sent_tokenize(str02)
  
  str09_P1JN=[]
  str09_P2N=[]
  
  
  for str04 in str03:
    
    #str04=str03[7]
    #str04="Enhancing optical communications with brand new fibers"
    
    if tY01_regc00D_uppercase.match(str04):
      str04=str04.lower()
    
    str05=tY10_raubt_tagger.tag(nltk.word_tokenize(str04))
    #replace none type by Z0
    str05b=[]
    str05t_prev="Y0"
    str05t_prev2="Y0"
    str05w_prev="Y0"
    for i in range(len(str05)):
      str05w=str05[i][0]
      str05t=str05[i][1]
      if (str05t==None):
        if tY01_regc09A_abbreviation.match(str05w):
          str05t="NN-ABR"
        else:
          str05t="Z0"
      elif (str05t[0:2]=="VB"):
        #use the following line and comment out the rest of if to include VBG in LDA
        str05t="NN-VB"
        #if ( (str05t_prev[0:2]=="JJ") or 
        #  (((tY01_regc00C_startW.match(str05t_prev)) or 
        #    (str05t_prev[0:2]=="CC") ) and (str05t_prev2[0]=="N")) or
        #  ( (i+1<len(str05)) and 
        #    ( (str05[i+1][1]==None) or (tY01_regc00C_startW.match(str05[i+1][1])) ) ) ):
        #  str05t="NN"
      elif (len(str05w)<=3):
        str05t="Z3"
      str05t_prev2=str05t_prev
      str05t_prev=str05t
      str05b.append( (str05w, str05t) )
    
    #parse phrases
    str06=tY11_NPChunk.parse(str05b)
    
    #can use str06.draw() to visualize the tree
    #or print(str06) for simple text
    
    #add phrases
    for str07 in str06:
      if (not isinstance(str07,nltk.Tree)):
        continue
      str08a=[tY12_lemmatize(w.lower()) for w,t in str07]
      if any([(w in tY13_stopwords_lem) for w in str08a]):
        continue
      #str08a=[w.lower() for w in str08a]
      str08b=" ".join(str08a)
      #old nltk uses tree.node. new nltk uses tree.label()
      if str07.label()=="P1JN":
        str09_P1JN.append(tY01_regc07A_posquote.sub("\g<sf>",str08b))
      elif str07.label()=="P2N":
        str09_P2N.append(tY01_regc07A_posquote.sub("\g<sf>",str08b))
  
  return " ".join(str09_P1JN+str09_P2N)


#####
# convert a string to lda corpus
def tA01_str2cpbow(str0):
  t3_str1=filter(lambda x: x in string.printable,str0)
  t3_str2=tY01_regexp01(t3_str1)
  return tA02_dict1.doc2bow(t3_str2.split())



#####
def tA01_printldadoc_index(index):
  
  t3_cpbow=tA02_cpbow2[index]
  t3_cpbow2=[(tA02_dict1.id2token[id], n) for id, n in t3_cpbow]
  t3_fname=tA02_fnamelist[index]
  t3_fnameb=tA01_regc02_ppdir.sub(tA02_dir2_raw,t3_fname)
  
  t3_file=open(t3_fnameb,"r")
  t3_str=t3_file.read()
  t3_file.close()
  
  t3_doctopic=tA02_lda[t3_cpbow]
  t3_doctopic.sort(key=lambda x: x[1],reverse=True)
  t3_doctopic2=[(tid,round(p,2)) for tid,p in t3_doctopic]
  
  if len(t3_doctopic2)<=0:
    return
  
  t3_topic0=[(w,round(p,3)) for w,p in tA02_lda.show_topic(topicid=t3_doctopic2[0][0],topn=20)]
  if len(t3_doctopic2)>1:
    t3_topic1=[(w,round(p,3)) for w,p in tA02_lda.show_topic(topicid=t3_doctopic2[1][0],topn=20)]
  else:
    t3_topic1=[]
  
  if len(t3_doctopic2)>2:
    t3_topic2=[(w,round(p,3)) for w,p in tA02_lda.show_topic(topicid=t3_doctopic2[2][0],topn=20)]
  else: 
    t3_topic2=[]
  
  print("########## ORIGINAL DOC ##########")
  print(t3_str)
  print("########## PREPROCESSED WORDS ##########")
  print(t3_cpbow2)
  print("########## TOPICS OF DOCUMENTS ##########")
  print(t3_doctopic)
  print("########## WORDS OF TOPICS ##########")
  print("TOPIC A")
  print(t3_topic0)
  print("TOPIC B")
  print(t3_topic1)
  print("TOPIC C")
  print(t3_topic2)

#####
def tA01_printldadoc_str(str1):
  
  t3_cpbow=tA01_str2cpbow(str1)
  t3_cpbow2=[(tA02_dict1.id2token[id], n) for id, n in t3_cpbow]
  
  t3_doctopic=tA02_lda[t3_cpbow]
  t3_doctopic.sort(key=lambda x: x[1],reverse=True)
  t3_doctopic2=[(tid,round(p,2)) for tid,p in t3_doctopic]
  
  if len(t3_doctopic2)<=0:
    return
  
  t3_topic0=[(w,round(p,3)) for w,p in tA02_lda.show_topic(topicid=t3_doctopic2[0][0],topn=20)]
  if len(t3_doctopic2)>1:
    t3_topic1=[(w,round(p,3)) for w,p in tA02_lda.show_topic(topicid=t3_doctopic2[1][0],topn=20)]
  else:
    t3_topic1=[]
  
  if len(t3_doctopic2)>2:
    t3_topic2=[(w,round(p,3)) for w,p in tA02_lda.show_topic(topicid=t3_doctopic2[2][0],topn=20)]
  else: 
    t3_topic2=[]
  
  print("########## ORIGINAL DOC ##########")
  print(str1)
  print("########## PREPROCESSED WORDS ##########")
  print(t3_cpbow2)
  print("########## TOPICS OF DOCUMENTS ##########")
  print(t3_doctopic)
  print("########## WORDS OF TOPICS ##########")
  print("TOPIC A")
  print(t3_topic0)
  print("TOPIC B")
  print(t3_topic1)
  print("TOPIC C")
  print(t3_topic2)



#####
def tA01_fname2index(fname):
  
  t1_regc=re.compile("/"+fname+"$")
  
  for index in range(len(tA02_fnamelist)):
    if t1_regc.search(tA02_fnamelist[index]):
      return index
  return None





#################################################
#6849324276002972933.txt


#convert a filename to index
doc_ind=tA01_fname2index("Concept-HMM")

#print information of a document with index (0)
tA01_printldadoc_index(doc_ind)

#show words of topic with topicid
tA02_lda.show_topic(topicid=0,topn=20)

# tA02_lda does not return full topic prob (ignore prob<0.01)
# tA02_lda.__getitem__(doc,eps=0) to have all topic probs




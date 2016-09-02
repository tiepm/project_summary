# -*- coding: utf-8 -*-
#################################################
import sys
import os
import re
import nltk
import time
import string
import operator

#################################################
# 
print("USAGE: python 025-02A-LDA-PP.py input-folder output-folder")
print("EXAMPLE: python 025-02A-LDA-PP.py 002-textinput-plain 002-textinput-PP")

#################################################
time1=time.time()
#################################################


# Get global tagger/chunker/regexp 
execfile("025-02Z-common.py")

#################################################
tY01_langlist=nltk.corpus.stopwords.fileids()
tY01_swlist={}
for lang in tY01_langlist:
  tY01_swlist[lang]=set(nltk.corpus.stopwords.words(lang))

def tY01_getlang(t1_str1):
  
  t1_str2=[w.lower() for w in nltk.wordpunct_tokenize(t1_str1)]
  t1_str3=set(t1_str2)
  
  swcount=dict.fromkeys(tY01_langlist,0)
  for lang in tY01_langlist:
    t1_str4=t1_str3.intersection(tY01_swlist[lang])
    swcount[lang]=len(t1_str4)
  return max(swcount.iteritems(),key=operator.itemgetter(1))[0]

#################################################

def tY01_regexp01(str01):
  if (str01==""):
    return ""
  
  str02=tY01_regc02A_linebreak.sub(". ",str01)
  str02=tY01_regc01A_num.sub("\g<pf> ",str02)
  str02=tY01_regc02B_rmpunct.sub(" ",str02)
  str02=tY01_regc02C_punctaddls.sub("\g<pf> \g<pu>",str02)
  str02=tY01_regc02D_punctaddrs.sub("\g<pu> \g<sf>",str02)
  
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



#################################################

tA01_dir1 = sys.argv[1]
tA01_dir2 = sys.argv[2]

#tA01_dir1 = "data20-engage"
#tA01_dir2 = "data20-LDA-PP"

tA01_filesize=1000

print("input folder: " + tA01_dir1 + " | ouput folder: "+ tA01_dir2 +
  " | min_size: " + str(tA01_filesize))

t1_count1=0
t1_count2=0

for t1_root, t1_sfolders, t1_fns in os.walk(tA01_dir1,followlinks=True):
  t1_dir4=os.path.relpath(t1_root,tA01_dir1)
  t1_root2=os.path.join(tA01_dir2,t1_dir4)
  if not os.path.exists(t1_root2):
    os.makedirs(t1_root2)
  for t1_fn0 in t1_fns:
    t1_fn1=os.path.join(t1_root,t1_fn0)
    if os.stat(t1_fn1).st_size<tA01_filesize:
      t1_count1+=1
      continue
    t1_fn2=os.path.join(t1_root2,t1_fn0)
    t1_file1=open(t1_fn1,"r")
    t1_str1=filter(lambda x: x in string.printable,t1_file1.read())
    t1_file1.close()
    
    if (tY01_getlang(t1_str1)!="english"):
      t1_count2 += 1
      continue
    
    t1_str2=tY01_regexp01(t1_str1)
    
    t1_file2=open(t1_fn2,"w")
    t1_file2.write(t1_str2)
    t1_file2.close()


print("skip: " + str(t1_count1) + " small files and "+ str(t1_count2) + " non-english files")

time2=time.time()
print("Processing time: ",time2-time1)









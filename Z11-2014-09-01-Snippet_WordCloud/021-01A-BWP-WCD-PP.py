# -*- coding: utf-8 -*-
#################################################
import sys
import os
import re
import nltk
import time
import string
#import numpy

#import numpy #needed for nltk named entity recognition
#importing error: modify the file yourself
#  File "/usr/local/lib/python2.7/dist-packages/nltk/chunk/named_entity.py", line 16, in <module>
#    from nltk.classify import MaxentClassifier
#  to 
#    from nltk.classify.maxent import MaxentClassifier

#################################################
# 

print("USAGE: python 021-01A-BWP-WCD-PP.py intpub-folder bio-folder output-folder")
print("EXAMPLE: python 021-01A-BWP-WCD-PP.py data10-intpub data10-bio data10-WCD-PP")

#################################################
time1=time.time()
#################################################

#################################################



# use 'government', 'hobbies', 'humor', 'learned', 'news', 'reviews',
tY10_tsen_train=nltk.corpus.brown.tagged_sents(categories=[
  'government', 'hobbies', 'humor', 'learned', 'news', 'reviews'])

###
def tY10_backoff_tagger(tagged_sents, tagger_classes, backoff=None):
  if not backoff:
    backoff = tagger_classes[0](tagged_sents)
    del tagger_classes[0]
  
  for cls in tagger_classes:
    tagger = cls(tagged_sents, backoff=backoff)
    backoff = tagger
  
  return backoff


tY10_word_patterns = [
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
    (r'.*ould$', 'MD'),
    #(r'.*ing$', 'VBG'),
    (r'.*ed$', 'VBD'),
    (r'.*ness$', 'NN'),
    (r'.*ment$', 'NN'),
    (r'.*ful$', 'JJ'),
    (r'.*ious$', 'JJ'),
    (r'.*ble$', 'JJ'),
    (r'.*ic$', 'JJ'),
    (r'.*ive$', 'JJ'),
    (r'.*ic$', 'JJ'),
    (r'.*est$', 'JJ'),
    (r'^a$', 'PREP'),
]

tY10_raubt_tagger = tY10_backoff_tagger(tY10_tsen_train, 
  [nltk.tag.AffixTagger, nltk.tag.UnigramTagger, 
    nltk.tag.BigramTagger, nltk.tag.TrigramTagger],
  backoff=nltk.tag.RegexpTagger(tY10_word_patterns))

#tY10_raubt_tagger.tag(nltk.word_tokenize('where   is   the   location     of'))

#################################################

tY11_NPChunk=nltk.RegexpParser(r"""
  P1JN: { <JJ.*>+(<N.*>+<Z1>?<N.*>+|<N.*>+) }
  P2N: { <N.*>+<Z1>?<N.*>+|<N.*><N.*>+ }
  P3NS: { <N.*> }""")

#Lemmatizer
tY12_Lem=nltk.stem.wordnet.WordNetLemmatizer()


#################################################

tY01_regc00A_intpubsplit=re.compile(r"(####|&&&&|@@@@)")
#tY01_regc00A_intpubsplit.split(r"""####
#Optical communications
#Optical networking
#&&&&
#investigating ar.
#@@@@
#Optical phase-shift-keyed transmission""")

#tY01_regc00B_startw =re.compile(r"^\w")
#tY01_regc00B_startw.match(r"'")
tY01_regc00C_startW =re.compile(r"^\W")
#tY01_regc00C_startW.match(r"'")

tY01_regc00D_uppercase =re.compile(r"^[A-Z0-9\s\W]+$")
#tY01_regc00D_uppercase.match("AYF ASHGD  £$ ASD  $ 67")

tY01_regc01A_num=re.compile(r"(?P<pf>(^|\W))[-+ ]*\d[\d.',]*(\d[eE]?[-+ ]*\d[\d.',]+)?")
#tY01_regc01_num.sub("\g<pf> ","-+- 123.23.'123e -123.123 daf 76% 76 76.7 fgas")

tY01_regc02A_linebreak=re.compile(r"(\r?\n)+")
#tY01_regc02_linebreak.sub(".","&&&&\r\nHe is presently")

#remove punctuations; keep ";'.!?,:|-"
#tY01_regc02B_rmpunct=re.compile(r'[\"`£$%^&\*@#~<>/\\\_+=]+')
#tY01_regc02B_rmpunct.sub(" ","i won't 123£ `! $%^&* -_+= dfas:df |{ asfs dfassd fa}. sdsaf; dsf: a \\ / , 123")

tY01_regc02C_nonwsplit=re.compile(r"\W")
#tY01_regc02C_nonwsplit.split("high-capacity network")

#add space between characters and punctuations (except ')
#tY01_regc05_punct2=re.compile(r"(?P<pf>\w)(?P<pu>[;.!?(){},:|]+)")
#tY01_regc05_punct2.sub("\g<pf> \g<pu>","afdf. sdfas sdf.sdfas sdf( asd{}fasd P(W|w)")

#tY01_regc05_punct3=re.compile(r"(?P<pu>[;.!?(){},:|]+)(?P<sf>\w)")
#tY01_regc05_punct3.sub("\g<pu> \g<sf>","afdf. sdfas sdf.sdfas")

#only keep noun and unknown
#tY01_regc05A_pos1=re.compile(r"^N.*",re.IGNORECASE)

#remove left spaces of posession quotes
tY01_regc07A_posquote =re.compile(r"[ ]+(?P<sf>['])")

#
tY01_regc09A_abbreviation =re.compile(r"^([A-Z0-9]{3,}[A-Z0-9_-]*|[A-Z0-9_-]*[A-Z0-9]{3,})$")
#tY01_regc09A_abbreviation.match("DQPSK")

#
tY01_regc11A_bio_wordmark =re.compile(r"""\s(bell lab|join|staff|author|co[-]?author|journal|conference|paper|patent|fellow|chair|member|institute|IEEE|academy|editor|committee|award|ph\.?d|msc|m\.s\.|bsc|b\.s\.|degree|born|child|grew|editor|chief|board|medal|director|world|international|inc\.|startup|cto |founder)""",re.VERBOSE|re.IGNORECASE)
#tY01_regc11A_bio_wordmark.search("he b.s. sdfa")

#################################################

def tY01_regexp01(str01):
  if (str01==""):
    return [[],[]]
  
  str02=tY01_regc02A_linebreak.sub(". ",str01)
  #str02=tY01_regc01A_num.sub("\g<pf> ",str02)
  #str02=tY01_regc02B_rmpunct.sub(" ",str02)
  
  #setence tokenizer
  str03=nltk.sent_tokenize(str02)
  
  str09_P1JN=[]
  str09_P2N=[]
  str09_P3NS=[]
  
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
        elif (str05w=="'") or (str05w=="'s"):
          str05t="Z1"
        else:
          str05t="Z0"
      elif (str05t[0:3]=="VBG"):
        if ( (str05t_prev[0:2]=="JJ") or 
          (((tY01_regc00C_startW.match(str05t_prev)) or 
            (str05t_prev[0:2]=="CC") ) and (str05t_prev2[0]=="N")) or
          ( (i+1<len(str05)) and 
            ( (str05[i+1][1]==None) or (tY01_regc00C_startW.match(str05[i+1][1])) ) ) ):
          str05t="NN"
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
      str08a=[w.lower() for w,t in str07]
      str08a[-1]=tY12_Lem.lemmatize(str08a[-1])
      #str08a=[w.lower() for w in str08a]
      str08b=" ".join(str08a)
      if str07.label()=="P1JN":
        str09_P1JN.append(tY01_regc07A_posquote.sub("\g<sf>",str08b))
      elif str07.label()=="P2N":
        str09_P2N.append(tY01_regc07A_posquote.sub("\g<sf>",str08b))
      elif str07.label()=="P3NS":
        str09_P3NS.append(str08b)
  
  return [str09_P1JN+str09_P2N,str09_P3NS]

###

def tY01_flatten(ngramlist):
  wordlist=[]
  for lw in ngramlist[0]:
    wordlist=wordlist+(tY01_regc02C_nonwsplit.split(lw))
  
  for lw in ngramlist[1]:
    wordlist=wordlist+(tY01_regc02C_nonwsplit.split(lw))
  
  wordlist2=[w for w in wordlist if ((not tY01_regc01A_num.match(w)) and (len(w)>2))]
  return wordlist2

###

def tY01_simplify(ngramlist,wordset):
  #ngramlist=t1_ngram_bio
  #wordset=t1_set1_intpub
  
  ngramlist0=[]
  for lw in ngramlist[0]:
    for w in tY01_regc02C_nonwsplit.split(lw):
      if (len(w)>2) and (w in wordset):
        ngramlist0.append(lw)
        break
  
  ngramlist1=[]
  for lw in ngramlist[1]:
    for w in tY01_regc02C_nonwsplit.split(lw):
      if (len(w)>2) and (w in wordset):
        ngramlist1.append(lw)
        break
  
  return [ngramlist0,ngramlist1]

###

def tY01_count(lword,locr):
  lcount=[0]*len(lword)
  
  for i in range(len(lword)):
    w1=lword[i]
    for w2 in tY01_regc02C_nonwsplit.split(w1):
      if (len(w2)>2):
        lcount[i]=lcount[i]+locr.count(w2)
  
  return lcount

###

def tY01_bio_shortening(str01):
  str02=nltk.sent_tokenize(str01)
  str03=[s for s in str02 if (not tY01_regc11A_bio_wordmark.search(s))]
  return " ".join(str03)



#################################################

tA01_dir1 = sys.argv[1]
tA01_dir2 = sys.argv[2]
tA01_dir3 = sys.argv[3]

#tA01_dir1 = "data10-belllabs-intpub"
#tA01_dir2 = "data10-belllabs-bio"
#tA01_dir3 = "data10-belllabs-WCD-PP"

print("intpub folder: ",tA01_dir1,"| bio folder: ", tA01_dir2,"| output folder: ", tA01_dir3)

for t1_root, t1_sfolders, t1_fns in os.walk(tA01_dir1,followlinks=True):
  t1_dir4=os.path.relpath(t1_root,tA01_dir1)
  t1_root2=os.path.join(tA01_dir2,t1_dir4)
  t1_root3=os.path.join(tA01_dir3,t1_dir4)
  if not os.path.exists(t1_root3):
    os.makedirs(t1_root3)
  for t1_fn0 in t1_fns: 
    t1_fn0b=t1_fn0[:-4]   #assume that t1_fn0 has format file.txt
    t1_fn1=os.path.join(t1_root,t1_fn0)
    t1_fn2=os.path.join(t1_root2,t1_fn0b)+"_bio.txt"
    t1_fn3=os.path.join(t1_root3,t1_fn0)
    t1_file1=open(t1_fn1,"r")
    t1_file3=open(t1_fn3,"w")
    
    t1_str1=filter(lambda x: x in string.printable,t1_file1.read())
    t1_str1B=tY01_regc00A_intpubsplit.split(t1_str1)
    if (len(t1_str1B)<=1):
      t1_file1.close()
      t1_file2.close()
      continue
    t1_str2_int=""
    t1_str2_pub=""
    
    for i in range(1,len(t1_str1B)):
      if t1_str1B[i-1][0:4]=="####":
        t1_str2_int=t1_str1B[i]
      elif t1_str1B[i-1][0:4]=="@@@@":
        t1_str2_pub=t1_str1B[i]
    if os.path.isfile(t1_fn2):
      t1_file2=open(t1_fn2,"r")
      t1_str2_bio=filter(lambda x: x in string.printable,t1_file2.read())
    else:
      t1_file2=None
      t1_str2_bio=""
    
    t1_ngram_int=tY01_regexp01(t1_str2_int)
    t1_ngram_pub=tY01_regexp01(t1_str2_pub)
    #if len(t1_ngram_int[0]) + len(t1_ngram_int[1]) + \
    #  len(t1_ngram_pub[0]) + len(t1_ngram_pub[1])>0:
    #  t1_ngram_bio=tY01_regexp01(t1_str2_bio)
    #else:
    t1_ngram_bio=tY01_regexp01(tY01_bio_shortening(t1_str2_bio))
    
    t1_str3_int=tY01_flatten(t1_ngram_int)
    t1_str3_pub=tY01_flatten(t1_ngram_pub)
    
    t1_set1_intpub=set(t1_str3_int+t1_str3_pub)
    
    if len(t1_set1_intpub)>4:
      t1_ngram_bio2=tY01_simplify(t1_ngram_bio,t1_set1_intpub)
    else:
      t1_ngram_bio2=t1_ngram_bio
    t1_str3_bio2=tY01_flatten(t1_ngram_bio2)
    
    t1_ngram_pub2=tY01_simplify(t1_ngram_pub,t1_set1_intpub)
    t1_str3_pub2=tY01_flatten(t1_ngram_pub2)
    
    t1_list_all=list(set(t1_ngram_int[0]+t1_ngram_int[1]+
      t1_ngram_pub2[0]+t1_ngram_pub2[1]+
      t1_ngram_bio2[0]+t1_ngram_bio2[1]))
    
    t1_count_int=tY01_count(t1_list_all,t1_str3_int)
    t1_count_pub=tY01_count(t1_list_all,t1_str3_pub2)
    t1_count_bio=tY01_count(t1_list_all,t1_str3_bio2)
    
    t1_list_len=[len([w2 for w2 in tY01_regc02C_nonwsplit.split(w1) if len(w2)>2]) \
      for w1 in t1_list_all]
    
    for i in range(len(t1_list_all)):
      t1_str6=t1_list_all[i] + "|" + str(t1_count_int[i]) + \
        "|" + str(t1_count_pub[i]) + \
        "|" + str(t1_count_bio[i]) + "|" + str(t1_list_len[i]) + "\n"
      t1_file3.write(t1_str6)
    
    print(t1_fn1)
    #print(t1_str6)
    
    
    t1_file1.close()
    if (t1_file2!=None):
      t1_file2.close()
    t1_file3.close()
    
    #if (t1_fn0=="273.txt"):
    #  stop_here


time2=time.time()
print("Processing time: ",time2-time1)







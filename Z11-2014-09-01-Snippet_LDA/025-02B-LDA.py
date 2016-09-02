# -*- coding: utf-8 -*-
#################################################
import sys
import os
import logging
import gensim
import time
import pickle


#################################################
# 
print("USAGE: python 025-02B-LDA.py input-folder ")
print("EXAMPLE: python 025-02B-LDA.py 002-textinput-PP")

#################################################
time1=time.time()
#################################################


#################################################

tA01_dir1 = sys.argv[1]


### create output hierarchy dir
print("source folder: ",tA01_dir1)

#for t1_root, t1_sfolders, t1_files in os.walk(tA01_dir1,followlinks=True):
#  t1_dir4=os.path.join(tA01_dir2,os.path.relpath(t1_root,tA01_dir1))
#  print(t1_dir4)
#  if not os.path.exists(t1_dir4):
#    os.makedirs(t1_dir4)

### create a courpus class for iterating through subdirs
class MyCorpus(object):
  def __init__(self,dir):
    self.dir=dir
    self.reset()
  
  def reset(self):
    self.osgen=os.walk(self.dir,followlinks=True)
    res=self.osgen.next()
    self.root=res[0]
    self.dnames=res[1]
    self.fnames=res[2]
    self.fnamesgen=iter(self.fnames)
    self.fnamelist=[]
  
  def __iter__(self):
    return self
  
  
  def next(self):
    while True:
      try:
        fn0=self.fnamesgen.next()
        fn1=os.path.join(self.root,fn0)
        self.fnamelist.append(fn1)
        file1=open(fn1,"r")
        str1=file1.read().split()
        file1.close()
        return str1
      except StopIteration:
        res=self.osgen.next()
        self.root=res[0]
        self.dnames=res[1]
        self.fnames=res[2]
        self.fnamesgen=iter(self.fnames)
    
  


time1=time.time()

# corpus iterator on text corpus
t2_cptext1=MyCorpus(tA01_dir1)

# build a dict
t2_dict1=gensim.corpora.Dictionary(t2_cptext1)
t2_dict1.filter_extremes(no_below=1, no_above=0.99, keep_n=None)
#t2_dict1.filter_extremes(no_below=50, no_above=0.3, keep_n=None)
t2_fnamelist=t2_cptext1.fnamelist


#t2_dict1.items()

# use the dict to build a bag-of-word corpus
t2_cptext1.reset()
t2_cpbow2=[t2_dict1.doc2bow(text) for text in t2_cptext1]

# use the bag-of-word corpus to build lda
t2_lda = gensim.models.ldamodel.LdaModel(corpus=t2_cpbow2, 
  id2word=t2_dict1, num_topics=100, update_every=1, 
  chunksize=10000, passes=1)

time2=time.time()
print("Processing time: ",time2-time1)

# misc transformation
# tfidf transform
# in gensim, tfidf, lda, lsi are transformations
#   and you can call corpus2=t1[corpus1]
#   /However the above function doesn't actually transform the corpus due to 
#     and memory and computation constraint
#   /You can only call 
#     for doc in corpus 2:
#   /Let's call cptext as the original text 
#     (gensim dont see cptext as the real corpus)
#     dictionary can transform cptext to cpbow (bag of word)
#     lda: cpbow => cplda
#     tfidf: cpbow => cptfidf
#     lsi: cptfidf => cplsi



# saving
# fnamelist contains the RELATIVE PATHS to preprocessed filenames
#   'data20-engage-LDA-PP/poll/a20018403.txt'
pickle.dump(t2_fnamelist,open("025-02B-gensim-fnamelist","w"))
gensim.corpora.MmCorpus.serialize("025-02B-gensim-cpbow2",t2_cpbow2)
t2_dict1.save("025-02B-gensim-dict1")
t2_lda.save("025-02B-gensim-lda")


















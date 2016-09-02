# -*- coding: utf-8 -*-
#################################################
import sys
import os
import time



#################################################
print("USAGE     : python 004-04D-KEA2-KExtractL.py ddir ddir-pp ldir model")
print("  ddir    : folder of raw documents (RELATIVE)")
print("  ddir-pp : folder of preprocessed documents (RELATIVE)")
print("  ldir    : folder of output labels (RELATIVE)")
print("  model   : input file of trained model (RELATIVE)")
print("WARNING   : COMMAND SYNTAX IS NOT CHECKED")
print("EXAMPLE   : python 004-04D-KEA2-KExtractL.py data02/ data02-KEA2-PP/ data02-KEA2-PP-output-label/ 004-04B-KEA2-model")
print("NOTE      : This code uses KEA2-KExtract")


print("ASSUMPTION")
print("  output KEA results are in the following format")
print("    the first line contains the keyphrases (phrase1|||phrase2)")
print("    the second line is the keysentence")
print("  FOR ENRON-NEO4j data: filename has the format YYYY-MM-DD-ID01")
print("    where <ID01> is the email ID in neo4j database")

#################################################
#sys.argv=["004-04D-KEA2-KExtractL.py","data-3months/data02B","data-3months/data02B-KEA2-PP","data-3months/data02B-KEA2","004-04B-KEA2-model"]

tA01_dir1 = sys.argv[1]
tA01_dir2 = sys.argv[2]
tA01_dir3 = sys.argv[3]
tA01_modelfile=sys.argv[4]

time1=time.time()
for t1_root, t1_sfolders, t1_fns in os.walk(tA01_dir1,followlinks=True):
  if len(t1_fns)<=0:
    continue
  t1_dirrel=os.path.relpath(t1_root,tA01_dir1)
  t1_dir4=os.path.join(tA01_dir2,t1_dirrel)
  t1_dir5=os.path.join(tA01_dir3,t1_dirrel)
  
  t1_cmdstr="Rscript 004-04C-KEA2-KExtract.R "+\
    t1_root + " " + t1_dir4 + " " + t1_dir5 + " " + tA01_modelfile
  
  os.system(t1_cmdstr)


time2=time.time()
print("Total running time: ",time2-time1)








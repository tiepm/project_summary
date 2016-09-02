options(echo=FALSE)
args<-commandArgs(trailingOnly = TRUE)
print(args)

library(tm)
library(rJava)
library(RKEA)

print("REQUIRE: R packages filehash, tm, rJava, RKEA. rJava may require liblzma-dev")
print("USAGE: Rscript 004-04D-KEA2-KExtractL.R ddir ddir-pp ldir model")
print("  ddir    : folder of raw documents (RELATIVE)")
print("  ddir-pp : folder of preprocessed documents (RELATIVE)")
print("  ldir   : folder of output labels (RELATIVE)")
print("  model  : input file of trained model (RELATIVE)")
print("WARNING  : COMMAND SYNTAX IS NOT CHECKED")
print("EXAMPLE  : Rscript 004-04D-KEA2-KExtractL.R data02 data02-KEA2-PP data02-KEA2 004-04B-KEA2-model")
print("NOTE     : This code uses KEA2-KExtract")

print("<ASSUMPTION>")
print("  output KEA results have the format")
print("    the first line contains the keyphrases (phrase1|||phrase2)")
print("    the second line is the keysentence")
print("  FOR ENRON-NEO4j: filename has the format YYYY-MM-DD-ID01")
print("    where <ID01> is the email ID in neo4j database")
print("</ASSUMPTION>")



############################################################################################

#args<-c("data02","data02-KEA2-PP/","data02-KEA2","004-04B-KEA2-model")

if (length(args)<4) {
  print("Not enough arguments")
  stop()
}

############################################################################################

###

tD01_dir01<-gsub("/$","",args[1])         #RELATIVE
tD01_dir02<-gsub("/$","",args[2])         #RELATIVE
tD01_dir03<-gsub("/$","",args[3])         #RELATIVE
tD02_modelfile01<-args[4]                 #RELATIVE

print(sprintf("ddir:%s",tD01_dir01))
print(sprintf("ldir:%s",tD01_dir03))
print(sprintf("model:%s",tD02_modelfile01))

### browse subfolders 
t1_yeardirs<-dir(tD01_dir02)
t1_dirs01<-paste(tD01_dir02,t1_yeardirs,sep="/")
t1_flag01<-file.info(t1_dirs01)$isdir
t1_yeardirs<-t1_yeardirs[t1_flag01]


time1<-proc.time()

for (t1_yeardir in t1_yeardirs) {
  t1_dir01<-paste(tD01_dir02,t1_yeardir,sep="/")
  t1_monthdirs<-dir(t1_dir01)
  t1_dirs02<-paste(t1_dir01,t1_monthdirs,sep="/")
  t1_flag02<-file.info(t1_dirs02)$isdir
  t1_monthdirs<-t1_monthdirs[t1_flag02]
  
  
  for (t1_monthdir in t1_monthdirs) {
    #t1_dir03<-gsub("^[[:graph:]]+/([[:digit:]]{4}/[[:digit:]]{1,2})$","\\1",t1_dir02)
    #t1_dir04<-paste(tD01_dir03,t1_dir03,sep="/")
    
    t1_dir03_raw<-sprintf("%s/%s/%s",tD01_dir01,t1_yeardir,t1_monthdir,sep="/")
    t1_dir03_pp<-sprintf("%s/%s/%s",tD01_dir02,t1_yeardir,t1_monthdir,sep="/")
    
    
    t1_cstr01<-sprintf("Rscript 004-04C-KEA2-KExtract.R %s %s %s %s",
      t1_dir03_raw,
      t1_dir03_pp,
      tD01_dir03,
      tD02_modelfile01)
    
    system(t1_cstr01)
  }
}

time2<-proc.time()
time21<-time2-time1
print(sprintf("Total running time: self: %0.2f|system: %0.2f|elapsed: %0.2f",
  time21["user.self"],
  time21["sys.self"],
  time21["elapsed"]))
remove(time1,time2,time21)

















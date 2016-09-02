options(echo=FALSE)
args<-commandArgs(trailingOnly = TRUE)
print(args)


# DONT USE THIS SCRIPT DIRECTLY

library(tm)
library(rJava)
library(RKEA)


############################################################################################

#args<-c("data-3months/data02/2000/1",
#  "data-3months/data02-KEA2-PP/2000/1",
#  "data-3months/data02-KEA2",
#  "004-04B-KEA2-model")



#args<-c("data-3months/data02B/AB/2",
#  "data-3months/data02B-KEA2-PP/AB/2",
#  "data-3months/data02B-KEA2/AB/2",
#  "004-04B-KEA2-model")



#args<-c("data-3months/data02B/AB",
#  "data-3months/data02B-KEA2-PP/AB",
#  "data-3months/data02B-KEA2/AB",
#  "004-04B-KEA2-model")



if (length(args)<4) {
  print("Not enough arguments")
  stop()
}




############################################################################################


### get the key sentence from the document and keyphrase
#tZ01_getKS<-function(doc,key,
#    rm_origMsg=TRUE,
#    rm_forwMsg=TRUE,
#    rm_citation=TRUE) {
tZ01_KEA_getKS<-function(doc,key,split=NULL) {
  #
  #assign("env_tZ01_getKS",value=environment(),envir=.GlobalEnv)
  
  #
  if (length(key)<=0) {
    return("")
  }
  
  docstr<-as.character(doc$content)
  docstr<-gsub("([[:alnum:]]+)([[:punct:]]+)","\\1 \\2",docstr)
  docstr<-gsub("([[:punct:]]+)([[:alnum:]]+)","\\1 \\2",docstr)
  if (is.null(split)) {
    docstr<-unlist(strsplit(docstr,"([[:space:][:punct:]]+)[.!;:?]([[:space:]]+)"))
  } else {
    docstr<-unlist(strsplit(docstr,split=split,fixed=TRUE))
  }
  #
  key2<-gsub("([[:punct:]]+)","",key)
  #
  kdm<-matrix(FALSE,nrow=length(key2),ncol=length(docstr))
  for (i in 1:length(key2)) {
    kdm[i,]<-grepl(key2[i],docstr)
  }
  #
  ind02<-which.max(colSums(kdm))
  docstr[ind02]
}



###
#due to new version of tm
tZ01_getID<-function(x) {
  x$meta$id
}


###
tZ01_printtime<-function(t1,str) {
  t2<-proc.time()
  t21<-t2-t1
  print(sprintf("%s: self: %0.2f|system: %0.2f|elapsed: %0.2f",
    str,t21["user.self"],t21["sys.self"],t21["elapsed"]))
  t2
}

############################################################################################
t1_hashcode<-paste(trunc(runif(16)*10),collapse="")
###

#
###
#dir01: raw data folder
tC01_dir01<-paste(getwd(),gsub("/$","",args[1]),sep="/")
tC01_filedb01<-sprintf("004-04C-dbcontrol01-%s.db",t1_hashcode) #relative
tC01_filedb01B<-paste(getwd(),tC01_filedb01,sep="/")            #absolute

#only use absolute path with PCorpus
#dir02: raw data folder
tC01_dir02<-paste(getwd(),gsub("/$","",args[2]),sep="/")
tC01_filedb02<-sprintf("004-04C-dbcontrol02-%s.db",t1_hashcode) #relative
tC01_filedb02B<-paste(getwd(),tC01_filedb02,sep="/")            #absolute

#dir03: output folder
tC01_dir03<-paste(getwd(),gsub("/$","",args[3]),sep="/")

#
tC02_modelfile01<-paste(getwd(),args[4],sep="/")

#
#t1_filedb<-paste(getwd(),
#  sprintf("004-04C-dbcontrol-t1-%s.db",t1_hashcode),sep="/")
### 

print(sprintf("ddir:%s",tC01_dir01))
print(sprintf("ddir-dbfile:%s",tC01_filedb01B))

print(sprintf("ddir-pp:%s",tC01_dir02))
print(sprintf("ddir-pp-dbfile:%s",tC01_filedb02B))

print(sprintf("ldir:%s",tC01_dir03))

print(sprintf("model:%s",tC02_modelfile01))



###
#options(echo=TRUE)

time_prev<-proc.time()

#file.remove(tC01_filedb01B)

tC01_data01 <- PCorpus(DirSource(tC01_dir01,recursive=FALSE),
    dbControl = list(dbName = tC01_filedb01B, dbType = "DB1"))

###
#file.remove(tC01_filedb02B)

tC01_data02 <- PCorpus(DirSource(tC01_dir02,recursive=FALSE),
    dbControl = list(dbName = tC01_filedb02B, dbType = "DB1"))

time_prev<-tZ01_printtime(time_prev,"Corpus loading time")

###
#matching the index between tC01_data02 and tC01_data01
#also remove unmatched files
t1_fn01<-unlist(lapply(tC01_data01,FUN=tZ01_getID))
t1_fn02<-unlist(lapply(tC01_data02,FUN=tZ01_getID))

t1_fnmatch<-match(t1_fn02,t1_fn01)
t1_id02<-which(is.finite(t1_fnmatch))
t1_id01<-t1_fnmatch[is.finite(t1_fnmatch)]

tC01_data01<-tC01_data01[t1_id01]
tC01_data02<-tC01_data02[t1_id02]

print(sprintf("Matching: %d files",length(t1_id01)))
time_prev<-tZ01_printtime(time_prev,"Matching time")

###
tC03_KEA<-extractKeywords(tC01_data02,model=tC02_modelfile01)

time_prev<-tZ01_printtime(time_prev,"Keyword extraction time")

###
# check the results for each document
i<-length(tC03_KEA)
tC01_data01[[i]]
tC01_data02[[i]]
tC03_KEA[[i]]
tZ01_KEA_getKS(tC01_data01[[i]],tC03_KEA[[i]])

# for data02B/AB/2. try i<-3 and
#tZ01_KEA_getKS(tC01_data01[[i]],tC03_KEA[[i]],split="|||")




###
dir.create(tC01_dir03,recursive=TRUE)
for (i in 1:length(tC03_KEA)) {
  
  t2_id<-tZ01_getID(tC01_data01[[i]])
  t2_str<-paste(
    paste(tC03_KEA[[i]],collapse="|||"),"\n",
    tZ01_KEA_getKS(tC01_data01[[i]],tC03_KEA[[i]]),sep="")
  t2_cfile<-paste(tC01_dir03,t2_id,sep="/")
  write(t2_str,file=t2_cfile)
  
}
time_prev<-tZ01_printtime(time_prev,"File export time")


file.remove(tC01_filedb01B)
file.remove(tC01_filedb02B)




options(echo=FALSE)
args<-commandArgs(trailingOnly = TRUE)
print(args)

library(tm)
library(rJava)
library(RKEA)

print("REQUIRE: R packages filehash, tm, rJava, RKEA. rJava may require liblzma-dev")
print("USAGE: Rscript 004-04B-KEA2-training.R ddir ldir model")
print("  ddir   : folder of training documents (RELATIVE)")
print("  ldir   : folder of training labels (RELATIVE)")
print("  model  : output file of trained model (RELATIVE)")
print("WARNING  : COMMAND SYNTAX IS NOT CHECKED")
print("EXAMPLE  :Rscript 004-04B-KEA2-training.R data02-KEA2-PP-train/ data02-KEA2-PP-train-label/ 004-04B-KEA2-model")


print("<ASSUMPTION> ")
print("  Two folders: training emails and training labels")
print("    the label file has the same name as the corresponding document file")
print("    it is allowed that the number of label files is smaller ")
print("        than the number of document files")
print("        In that case. only the labelled document files are used")
print("  Label format in each label file: (phrase1|||phrase2)")
print("  FOR ENRON-NEO4j data: filename has the format YYYY-MM-DD-ID01")
print("    where <ID01> is the email ID in neo4j database")
print("</ASSUMPTION>")
############################################################################################

#args<-c("data-3months/data02-KEA2-PP-train",
#  "data-3months/data02-KEA2-PP-train-label",
#  "004-04B-KEA2-model")


if (length(args)<3) {
  print("Not enough arguments")
  stop()
}




############################################################################################

###
tZ01_KEA_sw<-c(stopwords("english"),
  "tel","po","cc","bcc",
  "am","pm","subject","thanks","tkx","thank",
  "regards","best","good")


###
tZ01_KEA_map01<-function(x,strFlag=FALSE) {
  #
  #assign("env_KEA_map01",value=environment(),envir=.GlobalEnv)
  if (strFlag) {
    xstr<-x
  } else {
    xstr<-x$content
  }
  
  #remove email address
  xstr<-gsub("[[:alpha:]]([[:graph:]])*@([[:graph:]])+ "," ",tolower(xstr))
  #links
  xstr<-gsub("([[:alpha:]])+://([[:graph:]])+"," ",xstr)
  #remove attachments doc/exe/zip: later
  
  #stop words
  xstr<-removeWords(xstr,tZ01_KEA_sw)
  #punctuation
  xstr<-gsub("([\x21\x23-\x26\x28-\x40\x5b-\x60\x7b-\x80])+"," ",xstr)
  #strip white spaces
  gsub("([[:blank:]])+"," ",xstr)
  
  if (strFlag) {
    x<-xstr
  } else {
    x$content<-xstr
  }
  x
}

###
tZ01_splitKW<-function(x) {
  strsplit(paste(as.character(x),collapse=""),
    split="|||",
    fixed=TRUE)[[1]]
}


### get the key sentence from the document and keyphrase
#tZ01_getKS<-function(doc,key,
#    rm_origMsg=TRUE,
#    rm_forwMsg=TRUE,
#    rm_citation=TRUE) {
tZ01_KEA_getKS<-function(doc,key) {
  #
  #assign("env_tZ01_getKS",value=environment(),envir=.GlobalEnv)
  
  #
  if (length(key)<=0) {
    return("")
  }
  
  docstr<-as.character(doc$content)
  # remove original message
  #if (rm_origMsg) {
  #  ind01<-grep("([[:space:]]*)(-+)([[:space:]]*)Original Message([[:space:]]*)(-+)",docstr)[1]
  #  if (is.finite(ind01)) {
  #    if (ind01>1) 
  #      docstr<-docstr[1:(ind01-1)]
  #    else docstr<-character(0)
  #  }
  #}
  # remove forwarded message 
  #if (rm_forwMsg) {
  #  ind01<-grep("([[:space:]]*)(-+)([[:space:]]*)Forwarded by",docstr)[1]
  #  if (is.finite(ind01)) {
  #    if (ind01>1) 
  #      docstr<-docstr[1:(ind01-1)]
  #    else docstr<-character(0)
  #  }
  #}
  # remove citation
  #if (rm_citation) {
  #  ind01<-grep("^(<+)([[:space:]]+)",docstr)
  #  if (length(ind01)) {
  #    docstr<-docstr[-ind01]
  #  }
  #}
  # break to sentences (use regexp instead of Maxent's sentence detector)
  docstr<-paste(docstr,collapse=" ")
  docstr<-gsub("([[:alnum:]]+)([[:punct:]]+)","\\1 \\2",docstr)
  docstr<-gsub("([[:punct:]]+)([[:alnum:]]+)","\\1 \\2",docstr)
  docstr<-unlist(strsplit(docstr,"([[:space:][:punct:]]+)[.!;:?]([[:space:]]+)"))
  #docstr<-gsub("([[:space:]]+)"," ",docstr)
  #
  #docstr2<-removeWords(tolower(docstr),stopwords("english"))
  docstr2<-tZ01_KEA_map01(docstr,strFlag=TRUE)
  key2<-gsub("([[:punct:]]+)","",tolower(key))
  #
  kdm<-matrix(FALSE,nrow=length(key2),ncol=length(docstr2))
  for (i in 1:length(key2)) {
    kdm[i,]<-grepl(key2[i],docstr2)
  }
  #
  ind02<-which.max(colSums(kdm))
  docstr[ind02]
}


#due to new version of tm
tZ01_getID<-function(x) {
  x$meta$id
}

############################################################################################

###
t1_hashcode<-paste(trunc(runif(16)*10),collapse="")
###
tB01_dir01<-paste(getwd(),gsub("/$","",args[1]),sep="/")
tB01_filedb01<-sprintf("004-04B-dbcontrol01-%s.db",t1_hashcode) #relative
tB01_filedb01B<-paste(getwd(),tB01_filedb01,sep="/")            #absolute

tB01_dir03<-paste(getwd(),gsub("/$","",args[2]),sep="/")
tB01_filedb03<-sprintf("004-04B-dbcontrol03-%s.db",t1_hashcode) #relative
tB01_filedb03B<-paste(getwd(),tB01_filedb03,sep="/")            #absolute

tB02_modelfile01<-paste(getwd(),args[3],sep="/")

t1_filedb<-paste(getwd(),
  sprintf("004-04B-dbcontrol-t1-%s.db",t1_hashcode),sep="/")


### 
print(sprintf("ddir:%s",tB01_dir01))
print(sprintf("ddir-dbfile:%s",tB01_filedb01B))
print(sprintf("ddir-dbfile-temp:%s",t1_filedb))

print(sprintf("ldir:%s",tB01_dir03))
print(sprintf("ldir-dbfile:%s",tB01_filedb03B))
print(sprintf("model:%s",tB02_modelfile01))


###
time1<-proc.time()
### data
#file.remove(tB01_filedb01B)

tB01_data01 <- PCorpus(DirSource(tB01_dir01,recursive=TRUE),
    dbControl = list(dbName = tB01_filedb01B, dbType = "DB1"))

tB01_data01_id<-unlist(lapply(tB01_data01,FUN=tZ01_getID))
### label data
#file.remove(tB01_filedb03B)

tB01_data03 <- PCorpus(DirSource(tB01_dir03,recursive=TRUE),
    dbControl = list(dbName = tB01_filedb03B, dbType = "DB1"))

tB01_data03_KW<-lapply(tB01_data03,FUN=tZ01_splitKW)
tB01_data03_id<-unlist(lapply(tB01_data03,FUN=tZ01_getID))

tB01_data01_trainid<-match(tB01_data03_id,tB01_data01_id)
tB01_data03_KW<-tB01_data03_KW[is.finite(tB01_data01_trainid)]
tB01_data03_id<-tB01_data03_id[is.finite(tB01_data01_trainid)]
tB01_data01_trainid<-tB01_data01_trainid[is.finite(tB01_data01_trainid)]


file.copy(tB01_filedb01B,t1_filedb,overwrite=TRUE)
t1_data<-tB01_data01
t1_data$dbcontrol<-list(dbName=t1_filedb,dbtype="DB1")
###
t1_data<-tm_map(t1_data, tZ01_KEA_map01)
###
file.remove(tB02_modelfile01)
createModel(t1_data[tB01_data01_trainid],
  keywords=tB01_data03_KW,
  model=tB02_modelfile01)

tB03_KEA<-extractKeywords(t1_data,model=tB02_modelfile01)

i<-1
print("AN EXAMPLE. Check if this works")
print(tB01_data01[[i]])
print(tB03_KEA[[i]])
print(tZ01_KEA_getKS(tB01_data01[[i]],tB03_KEA[[i]]))


###
time2<-proc.time()
time21<-time2-time1
print(sprintf("Total running time: self: %0.2f|system: %0.2f|elapsed: %0.2f",
  time21["user.self"],
  time21["sys.self"],
  time21["elapsed"]))
remove(time1,time2,time21)



file.remove(tB01_filedb01B)
file.remove(tB01_filedb03B)
file.remove(t1_filedb)


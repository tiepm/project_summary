library(wordcloud)
########################################################################################################################

#############################################################################
options(echo=FALSE)
args<-commandArgs(trailingOnly = TRUE)
print(args)

print("USAGE: Rscript 021-01B-BWP-WCD.R inputfolder-PP outputfolder")
print("EXAMPLE : Rscript 021-01B-BWP-WCD.R data10-WCD-PP data10-WCD")

print(args)

if (length(args)<2) {
  print("Not enough arguments")
  stop()
}


#############################################################################

#tB01_dir01<-"data10-belllabs-WCD-PP"
tB01_dir01<-gsub("/$","",args[1])          #RELATIVE

#tB01_dir03<-"data10-belllabs-WCD"
tB01_dir03<-gsub("/$","",args[2])          #RELATIVE

dir.create(tB01_dir03,recursive=TRUE)

tB01_palette <- brewer.pal(8,"Dark2")

t2_fns01<-dir(tB01_dir01)
t2_fns02<-paste(tB01_dir01,t2_fns01,sep="/")
t2_fns01_ind<-which(!file.info(t2_fns02)$isdir)
t2_fns01<-t2_fns01[t2_fns01_ind]
t2_fns02<-t2_fns02[t2_fns01_ind]

t2_fns01B<-t2_fns01
for (i in seq_along(t2_fns01)){
  t2_fns01B[i]<-paste(substr(t2_fns01B[i],1,nchar(t2_fns01B[i])-4),".jpg",sep="")
}
t2_fns03<-paste(tB01_dir03,t2_fns01B,sep="/")

time1<-proc.time()

t2_defscale_pdf<-c(2.5,0.15)
t2_defscale_jpeg<-c(2.5,0.15)*1.4

set.seed(0)
for (i in seq_along(t2_fns01)) {
  
  t2_df01=tryCatch(read.table(t2_fns02[i],sep="|",stringsAsFactors=FALSE),
    error=function(e) {print(e); return(NULL)} )
  if (is.null(t2_df01)) {
    next
  }
  
  dimnames(t2_df01)<-list(dimnames(t2_df01)[[1]],
    c("word","cint","cpub","cbio","len"))
  t2_df01$count<-(8*t2_df01$cint+2*t2_df01$cpub+t2_df01$cbio)/t2_df01$len
  t2_df01<-t2_df01[which(is.finite(t2_df01$count)),]
  t2_df01$count<-t2_df01$count/sum(t2_df01$count)
  
  
  for (j in seq(1,0.8,by=-0.05)) {
    print(j)
    #pdf(file=t2_fns03[i],height=5,width=5)
    jpeg(file=t2_fns03[i],height=720,width=720)
    #png("wordcloud.png", width=1280,height=1280)
    t2_failList<-(wordcloud(t2_df01$word,t2_df01$count, 
        scale=t2_defscale_jpeg*j,min.freq=1e-3,
        max.words=100, random.order=FALSE, rot.per=.15, colors=tB01_palette))
    dev.off()
    
    if ((is.null(t2_failList))||(t2_failList>20))
      break
  }
  
}


time2<-proc.time()
time21<-time2-time1
print(sprintf("Total running time: self: %0.2f|system: %0.2f|elapsed: %0.2f",
  time21["user.self"],
  time21["sys.self"],
  time21["elapsed"]))
remove(time1,time2,time21)










################################################################################
# Interactive script:
#     Testing several methods on PKDD upselling challenge
# The script is divided into blocks (of different methods)
################################################################################


library(ggplot2)

library(randomForest)
library(doMC)
library(mgcv)
library(dbscan)


###
rm(list=setdiff(ls(),c(ls(pattern="^d0[1-2]_"),ls(pattern="^prj_"),"Sysconf",ls(pattern="^res0"))) )
rm(d01_user,d01_train,d01_train2,d01_train3)

###
# rm(list=ls(pattern="^res0[2-3]"))

###
# save(list=c(ls(pattern="^d0[1-2]")),file="tempB01.RData")

# save(list=ls(pattern="^res0[2-3]"),file="tempB01_tree.RData")

# save(list=c(ls(pattern="^res1[0-9]"),ls(pattern="^d1[0-2]")),file="tempB01_bamtree.RData")

# save(list=c(ls(pattern="^res2[0-9]"),ls(pattern="^d2[0-2]")),file="tempB01_localbam_tree.RData")

###
load(file="tempB01.RData")

# load(file="tempB01_tree.RData")

# load(file="tempB01_bamtree.RData")

# load(file="tempB01_localbam_tree.RData")

###
source("./script/001-helper.R")



# save(list=ls(),file="tempB02.RData")

################################################################################

##########
d01_bank<-read.table(Sysconf$FN_DATA01_BANK,header=TRUE,sep=",")
d01_bank<-prj_d01_preprocessing(d01_bank,"bank")

d01_user14<-read.table(Sysconf$FN_DATA01_USER2014,header=TRUE,sep=",")
d01_user14<-prj_d01_preprocessing(d01_user14,"user")

d01_user15<-read.table(Sysconf$FN_DATA01_USER2015,header=TRUE,sep=",")
d01_user15<-prj_d01_preprocessing(d01_user15,"user")

d01_train14<-read.table(Sysconf$FN_DATA01_TRAIN2014,header=TRUE,sep=",")
d01_train14<-prj_d01_preprocessing(d01_train14,"train")

d01_train15<-read.table(Sysconf$FN_DATA01_TRAIN2015,header=TRUE,sep=",")
d01_train15<-prj_d01_preprocessing(d01_train15,"train")

# global function: match d01_train14$geo_x, geo_y of bank actvities to bank$geo_x, geo_y
prj_match_bank_poi_id()



########## NOTE
# a/two user sets 14/15 are completely disjoint
# b/there are users in user14 but not in train14 (users with no activity)
#   (same with user15,train15)
# c/There are users in user14 dataset but apply for credit card in 2015
#     (e.g. user14: id=111628). 
#   There are users who already had a CC, then stoped that CC and re-apply a new one
#     (e.g. user14: id=111628). 
#   There are users who already had a CC, then stopped that CC and do not re-apply 
#     for a new CC (e.g. 160574)
#   There are users who already had a CC but still apply for another one? (e.g. 62940)
# d/Task output
#   Task 1: Not sure if the desired output is 5-branch for the second half of 2015 
#     or for the whole 2015
#   Task 2: Count the users applying in the first half of 2015?
#     Count the users applying in 2016?
# e/Activity:
#   If channel=="n" (webshop), there is no info of time/loc/mc/card/amt/geo
#   If channel=="b" (branch), there is no info of time/mc/card/amt/geo
#     Also, in that case, train$poi_id will match with a bank id bank$poi_id


##########

t01_poiid_potential<-prj_get_potential_poiid(d01_user14,d01_train14,2)

#
d01_userB14a<-prj_get_userB(d01_user14,d01_train14,
  prj_POSIX2date("2014-01-01"),prj_POSIX2date("2014-06-30"),
  poiid=t01_poiid_potential$ind_all)

#
d01_userB14b<-prj_get_userB(d01_user14,d01_train14,
  prj_POSIX2date("2014-07-01"),prj_POSIX2date("2014-12-31"),
  poiid=t01_poiid_potential$ind_all)

#
d01_userB15a<-prj_get_userB(d01_user15,d01_train15,
  prj_POSIX2date("2015-01-01"),prj_POSIX2date("2015-06-30"),
  poiid=t01_poiid_potential$ind_all)


#
norm_userB<-TRUE
if (norm_userB==TRUE) {
  cnames<-setdiff(colnames(d01_userB14a),"user_id")
  d01_userB14a[,cnames]<-d01_userB14a[,cnames]-rep(colMeans(d01_userB14a)[cnames],each=nrow(d01_userB14a))
  
  cnames<-setdiff(colnames(d01_userB14b),"user_id")
  d01_userB14b[,cnames]<-d01_userB14b[,cnames]-rep(colMeans(d01_userB14b)[cnames],each=nrow(d01_userB14b))
  
  cnames<-setdiff(colnames(d01_userB15a),"user_id")
  d01_userB15a[,cnames]<-d01_userB15a[,cnames]-rep(colMeans(d01_userB15a)[cnames],each=nrow(d01_userB15a))
  
}


#
rm(t01_poiid_potential,norm_userB)

##########
d01_userC14a<-prj_get_userstat_per_partition(d01_user14,d01_userB14a,
  c("c0106","w0106"),
  partition=list(tt2_14a=which(d01_user14$tt2_14a),
  tt2_14b=which(d01_user14$tt2_14b),
  tt2_15a=which(d01_user14$tt2_15a),
  tt2_none=which(d01_user14$tt2_none)) )

d01_userC14b<-prj_get_userstat_per_partition(d01_user14,d01_userB14b,
  c("c0712","w0712"),
  partition=list(tt2_14a=which(d01_user14$tt2_14a),
  tt2_14b=which(d01_user14$tt2_14b),
  tt2_15a=which(d01_user14$tt2_15a),
  tt2_none=which(d01_user14$tt2_none)) )


##########
tempres<-prj_get_userD(c("tt2_14b","tt2_15a"),samratio_14a=1.0,samratio_14b=1.0)
d02_train<-tempres$dtrain
d02_val<-tempres$dval
d02_test<-tempres$dtest
rm(tempres)


################################################################################
# RF: prediction of upselling in the next 6 months
################################################################################

##########
registerDoMC(cores = 6)

drun<-d02_train

drun_rm<-c("user_id",
  colnames(drun)[grep("poi_id",colnames(drun))]
  # colnames(drun)[setdiff(grep("poi_id",colnames(drun)),grep("poi_id_9119",colnames(drun)))]
  )

for (cname in drun_rm) {
  drun[[cname]]<-NULL
}
rm(drun_rm,cname)


temptab<-rev(table(drun$label))
temptab=c(1,200)
classwt<-(temptab)/sum(temptab)
classwt

# parallel
res02a <- foreach(ntree=rep(1000, 6), .packages='randomForest') %dopar% {
  
  # [Memory issue] Use at most 1-2 threads on full data with classwt
  # randomForest(label ~ ., data=drun,ntree=ntree,do.trace=10,classwt=classwt)
  
  # Downsampling can support more threads
  randomForest(label ~ ., data=drun,ntree=ntree,do.trace=10,
    sampsize=c(2700,1800))
}
rm(drun,temptab,classwt)
res02b<-do.call(combine,res02a)
rm(res02a)

# non-parallel
res02a<-list()
for (i in 1:1) {
  res02a[[i]]<-randomForest(label ~ ., data=drun,ntree=3000,do.trace=10,
      sampsize=c(1500,1800))
}

# res02b<-do.call(combine,res02a)
res02b<-res02a[[1]]



#
res02b$lab1<-d02_train$label
res02b$predp1<-predict(res02b,newdata=d02_train,type="prob")
res02b$predr1<-predict(res02b,newdata=d02_train,type="response")
res02b$lab2<-d02_val$label
res02b$predp2<-predict(res02b,newdata=d02_val,type="prob")
res02b$predr2<-predict(res02b,newdata=d02_val,type="response")
res02b$lab3<-NULL
res02b$predp3<-predict(res02b,newdata=d02_test,type="prob")
res02b$predr3<-predict(res02b,newdata=d02_test,type="response")

#
res02b$auc1<-prj_get_auc(as.numeric(as.character(res02b$lab1)=="1"),
  res02b$predp1[,"1"])
res02b$confmat1<-prj_get_confmat(res02b$lab1,res02b$predr1)


res02b$auc2<-prj_get_auc(as.numeric(as.character(res02b$lab2)=="1"),
  res02b$predp2[,"1"])
res02b$confmat2<-prj_get_confmat(res02b$lab2,res02b$predr2)


#
res02b$confmat1/rowSums(res02b$confmat1)
c(sum(res02b$confmat1[c(1,4)])/sum(res02b$confmat1), sum(res02b$confmat1[c(2,3)])/sum(res02b$confmat1))
res02b$auc1


res02b$confmat2/rowSums(res02b$confmat2)
c(sum(res02b$confmat2[c(1,4)])/sum(res02b$confmat2), sum(res02b$confmat2[c(2,3)])/sum(res02b$confmat2))
res02b$auc2

##########
temptab<-cbind(d02_test$user_id,res02b$predp3[,"1"])
colnames(temptab)<-c("#USER_ID","SCORE")
write.table(temptab,file="task2_output_15a_res02b_sr100_ss27001800.txt",quote=FALSE,sep=",",row.names=FALSE)
rm(temptab)


################################################################################
# RF: prediction of upselling in the current 6 months
################################################################################

tempres<-prj_get_userD(c("tt2_14a","tt2_14b"),samratio_14a=1.0,samratio_14b=0.95)
d03_train<-tempres$dtrain
d03_val<-tempres$dval
d03_test<-tempres$dtest
rm(tempres)

##########
registerDoMC(cores = 6)

drun<-d03_train
drun$user_id<-NULL

res03a <- foreach(ntree=rep(500, 6), .packages='randomForest') %dopar% {
  
  # Downsampling can support 6 threads
  randomForest(label ~ ., data=drun,ntree=ntree,do.trace=10,
    sampsize=c(1500,1800))
}
rm(drun)
res03b<-do.call(combine,res03a)
rm(res03a)

#
res03b$lab1<-d03_train$label
res03b$predp1<-predict(res03b,newdata=d03_train,type="prob")
res03b$predr1<-predict(res03b,newdata=d03_train,type="response")
res03b$lab2<-d03_val$label
res03b$predp2<-predict(res03b,newdata=d03_val,type="prob")
res03b$predr2<-predict(res03b,newdata=d03_val,type="response")
res03b$lab3<-NULL
res03b$predp3<-predict(res03b,newdata=d03_test,type="prob")
res03b$predr3<-predict(res03b,newdata=d03_test,type="response")

#
res03b$auc1<-prj_get_auc(as.numeric(as.character(res03b$lab1)=="1"),
  res03b$predp1[,"1"])
res03b$confmat1<-prj_get_confmat(res03b$lab1,res03b$predr1)


res03b$auc2<-prj_get_auc(as.numeric(as.character(res03b$lab2)=="1"),
  res03b$predp2[,"1"])
res03b$confmat2<-prj_get_confmat(res03b$lab2,res03b$predr2)


#
res03b$confmat1/rowSums(res03b$confmat1)
c(sum(res03b$confmat1[c(1,4)])/sum(res03b$confmat1), sum(res03b$confmat1[c(2,3)])/sum(res03b$confmat1))
res03b$auc1


res03b$confmat2/rowSums(res03b$confmat2)
c(sum(res03b$confmat2[c(1,4)])/sum(res03b$confmat2), sum(res03b$confmat2[c(2,3)])/sum(res03b$confmat2))
res03b$auc2

##########
temptab<-cbind(d02_test$user_id,res02b$predp3[,"1"]*res03b$predp3[,"0"])
colnames(temptab)<-c("#USER_ID","SCORE")
write.table(temptab,file="task2_output_15a_res02b_sr100_ss15001800_res03b_sr095_ss15001800.txt",quote=FALSE,sep=",",row.names=FALSE)
rm(temptab)

##########
temptab<-cbind(d02_test$user_id,1-res02b$predp3[,"0"]*res03b$predp3[,"0"])
colnames(temptab)<-c("#USER_ID","SCORE")
write.table(temptab,file="task2_output_15a_res02b_sr100_ss15001800_res03b_sr095_ss15001800_union.txt",quote=FALSE,sep=",",row.names=FALSE)
rm(temptab)


################################################################################
# RF: MISC
################################################################################

##########
ind01<-which((as.character(res02b$predr2)=="0")&(as.character(res02b$lab2)=="1"))[5]
d02_val[ind01:(ind01+3),]
ind02<-d02_val[ind01,"user_id"]
d01_user14[d01_user14$user_id==ind02,]
d01_train14[d01_train14$user_id==ind02,]
rm(ind01,ind02)

# user with credit card activity but still submit for another cc
which( (d02_val$acrd_cnt_c>0) & (as.character(d02_val$label)=="1") )

which( (d02_val$acrd_cnt_c>0) & (as.character(res02b$predr2)=="1") )

# this user has a cc activity but cc is not recorded in c2014XX
#   Alos, this user applies for another CC in 2015
d02_val[1561,]

d01_user14[d01_user14$user_id==2148,]
d01_train14[d01_train14$user_id==2148,]



# user with no activity but still apply for a cc
which( (d02_val$avt_cnt==0) & (as.character(d02_val$label)=="1") )

which( (d02_val$avt_cnt==0) & (as.character(res02b$predr2)=="1") )



##########
x1a<-which(as.character(res02b$lab2)=="1")
x1b<-which(as.character(res02b$predr2)=="1")
x2a<-which(as.character(res03b$lab2)=="1")
x2b<-which(as.character(res03b$predr2)=="1")

intersect(setdiff(x1b,x2b),x1a)

rm(x1a,x1b,x2a,x2b)




##########
prj_get_auc(as.numeric(as.character(res02b$lab2)=="1"),
  res02b$predp2[,"1"])

tempp<-res02b$predp2[,"1"]*res03b$predp2[,"0"]
prj_get_auc(as.numeric(as.character(res02b$lab2)=="1"),
  tempp)

#
prj_get_auc(as.numeric(as.character(res02b$lab1)=="1"),
  res02b$predp1[,"1"])

tempp<-res02b$predp1[,"1"]*res03b$predp1[,"0"]
prj_get_auc(as.numeric(as.character(res02b$lab1)=="1"),
  tempp)



##########
x1<-lapply(d02_train,FUN=function(x) if (is.factor(x)) {table(x)} else {quantile(x,prob=0:10/10)} )
x1$label<-NULL
x2<-lapply(d02_test,FUN=function(x) if (is.factor(x)) {table(x)} else {quantile(x,prob=0:10/10)} )

cbind(unlist(x1),unlist(x2))

################################################################################
# BAM + RF
################################################################################

##########
drun<-d02_train
drun_rm<-c("user_id",
  colnames(drun)[grep("poi_id",colnames(drun))]
  # colnames(drun)[setdiff(grep("poi_id",colnames(drun)),grep("poi_id_9119",colnames(drun)))]
  )
for (cname in drun_rm) {
  drun[[cname]]<-NULL
}
rm(cname,drun_rm)

#
sampling_size<-c(20000,4000)
bam_weights<-c(rep(1,sampling_size[1]),rep(5,sampling_size[2]))
negsam_rmth<-0.5

#
str_formulaA<-"label ~"
for (cname in setdiff(colnames(drun),"label")) {
  if (is.numeric(drun[[cname]])) {
    str_formulaA<-paste0(str_formulaA," + s(",cname,",bs='cr',k=4)")
    # str_formulaA<-paste0(str_formulaA," + ",cname)
  } else {
    str_formulaA<-paste0(str_formulaA," + s(",cname,",bs='re')")
  }
}
rm(cname)
str_formulaA<-sub("~ [+]","~",str_formulaA)

set.seed(0)
drun2<-drun

bamres_list<-list()
bam_control<-gam.control()
bam_control$maxit<-100
bam_control$trace<-TRUE
for (i in 1:50) {
  #
  ind01a_neg<-sample(which(as.character(drun2$label)=="0"),sampling_size[1])
  ind01b_pos<-sample(which(as.character(drun2$label)=="1"),sampling_size[2])
  
  drun3<-drun2[c(ind01a_neg,ind01b_pos),]
  
  # make sure that there is one observation for any factor level
  for (cname in setdiff(colnames(drun3),"label")) {
    if (is.factor(drun3[[cname]])) {
      temptab<-table(drun3[[cname]])
      ind03<-which(temptab<=0)
      if (length(ind03)>0) {
        for (j in 1:length(ind03)) {
          ind03b<-sample.int(nrow(drun3),1)
          drun3[ind03b,cname]<-names(ind03)[j]
        }
      }
    }
  }
  
  #
  bamres<-bam(formula(str_formulaA),data=drun3,family="binomial",
    weights=bam_weights,control=bam_control)
  drun2_pred<-predict(bamres,drun2,type="response")
  print(sprintf("Iteration %d:",i))
  print(sprintf("   Neg: (%9d / %9d) . Quant(0,..,100): %s",
    sum(drun2_pred[as.character(drun2$label)=="0"]>0.5),sum(as.character(drun2$label)=="0"),
    toString(round(quantile(drun2_pred[as.character(drun2$label)=="0"],probs=0:10/10),2) ) ) )
  print(sprintf("   Pos: (%9d / %9d) cases. Quant(0,..,100): %s",
    sum(drun2_pred[as.character(drun2$label)=="1"]<=0.5),sum(as.character(drun2$label)=="1"),
    toString(round(quantile(drun2_pred[as.character(drun2$label)=="1"],probs=0:10/10),2) ) ) )
  
  # remove some negative samples
  possam_min_th<-min(drun2_pred[as.character(drun2$label)=="1"])
  ind02<-(as.character(drun2$label)=="0")&
    (drun2_pred<max(possam_min_th,min(negsam_rmth,quantile(drun2_pred[as.character(drun2$label)=="0"],probs=0.05))) )
  
  drun2<-drun2[!ind02,]
  
  #
  bamres_list[[i]]<-bamres
}

res10_bamlist<-bamres_list
rm(drun,drun2,drun3,possam_min_th,bamres,drun2_pred,ind01a_neg,ind01b_pos,ind02,bamres_list,ind03,ind03b)
rm(sampling_size,bam_weights,negsam_rmth,str_formulaA)
rm(i,j,temptab,cname,cnames,bam_control)

##########

d12_train<-d02_train[,]
d12_val<-d02_val[,]
d12_test<-d02_test[,,drop=FALSE]

for (i in 1:length(res10_bamlist)) {
  namei<-paste0("bam",i)
  d12_train[[namei]]<-predict(res10_bamlist[[i]],d02_train,type="response")
  d12_val[[namei]]<-predict(res10_bamlist[[i]],d02_val,type="response")
  d12_test[[namei]]<-predict(res10_bamlist[[i]],d02_test,type="response")
}
rm(i,namei)

for (cname in colnames(d12_train)) {
  dimnames(d12_train[[cname]])<-NULL
}
for (cname in colnames(d12_val)) {
  dimnames(d12_val[[cname]])<-NULL
}
for (cname in colnames(d12_test)) {
  dimnames(d12_test[[cname]])<-NULL
}
rm(cname)



##########
registerDoMC(cores = 3)

drun<-d12_train
drun$user_id<-NULL

# parallel
res12a <- foreach(ntree=rep(1000, 6), .packages='randomForest') %dopar% {
  
  # Downsampling can support more threads
  randomForest(label ~ ., data=drun,ntree=ntree,do.trace=10,
    sampsize=c(3200,4000))
}
rm(drun,temptab,classwt)
res12b<-do.call(combine,res12a)
rm(res12a)



#
res12b$lab1<-d12_train$label
res12b$predp1<-predict(res12b,newdata=d12_train,type="prob")
res12b$predr1<-predict(res12b,newdata=d12_train,type="response")
res12b$lab2<-d12_val$label
res12b$predp2<-predict(res12b,newdata=d12_val,type="prob")
res12b$predr2<-predict(res12b,newdata=d12_val,type="response")
res12b$lab3<-NULL
res12b$predp3<-predict(res12b,newdata=d12_test,type="prob")
res12b$predr3<-predict(res12b,newdata=d12_test,type="response")

#
res12b$auc1<-prj_get_auc(as.numeric(as.character(res12b$lab1)=="1"),
  res12b$predp1[,"1"])
res12b$confmat1<-prj_get_confmat(res12b$lab1,res12b$predr1)


res12b$auc2<-prj_get_auc(as.numeric(as.character(res12b$lab2)=="1"),
  res12b$predp2[,"1"])
res12b$confmat2<-prj_get_confmat(res12b$lab2,res12b$predr2)


#
res12b$confmat1/rowSums(res12b$confmat1)
c(sum(res12b$confmat1[c(1,4)])/sum(res12b$confmat1), sum(res12b$confmat1[c(2,3)])/sum(res12b$confmat1))
res12b$auc1


res12b$confmat2/rowSums(res12b$confmat2)
c(sum(res12b$confmat2[c(1,4)])/sum(res12b$confmat2), sum(res12b$confmat2[c(2,3)])/sum(res12b$confmat2))
res12b$auc2

##########
temptab<-cbind(d02_test$user_id,res12b$predp3[,"1"])
colnames(temptab)<-c("#USER_ID","SCORE")
write.table(temptab,file="task2_output_15a_res12b_sr100_ss32004000.txt",quote=FALSE,sep=",",row.names=FALSE)
rm(temptab)





################################################################################
# local BAM + RF
################################################################################

##########
drun<-d02_train
drun_rm<-c(
  colnames(drun)[grep("poi_id",colnames(drun))]
  # colnames(drun)[setdiff(grep("poi_id",colnames(drun)),grep("poi_id_9119",colnames(drun)))]
  )
for (cname in drun_rm) {
  drun[[cname]]<-NULL
}
rm(cname,drun_rm)

drun_tnorm<-drun
drun_tnorm$user_id<-NULL
for (cname in colnames(drun_tnorm)) {
  if (is.factor(drun_tnorm[[cname]])) {
    drun_tnorm[[cname]]<-NULL
  }
}
rm(cname)
drun_tnorm<-as.matrix(drun_tnorm)
drun_tnorm<-(t(drun_tnorm)-colMeans(drun_tnorm))/apply(drun_tnorm,MARGIN=2,FUN=sd)

# use all positive samples
sampling_size<-c(8000,-1)
ind01b_posall<-which(as.character(drun2$label)=="1")
bam_weights<-c(rep(1,sampling_size[1]),rep(2,length(ind01b_posall)))

# find boundary nodes of positive sets
drun_tnorm2<-drun_tnorm[,ind01b_posall]
drun_tnorm2c<-rowMeans(drun_tnorm2)
ind04_posset_boundary<-rep(FALSE,length(ind01b_posall))
for (i in 1:length(ind01b_posall)) {
  if (all(colSums((drun_tnorm2-drun_tnorm2[,i])*(drun_tnorm2c-drun_tnorm2[,i]))>=0)) {
    ind04_posset_boundary[i]<-TRUE
  }
  
}
ind04_posset_boundary<-ind01b_posall[ind04_posset_boundary]
rm(drun_tnorm2,drun_tnorm2c,i)

boundary_n<-min(length(ind04_posset_boundary),24)
if (boundary_n<length(ind04_posset_boundary)) {
  ind04_posset_boundary<-sort(sample(ind04_posset_boundary,boundary_n))
}
ind04_posset_bdr_userid<-drun$user_id[ind04_posset_boundary]
rm(ind04_posset_boundary)


#
str_formulaA<-"label ~"
for (cname in setdiff(colnames(drun),c("label","user_id"))) {
  if (is.numeric(drun[[cname]])) {
    str_formulaA<-paste0(str_formulaA," + s(",cname,",bs='cr',k=4)")
    # str_formulaA<-paste0(str_formulaA," + ",cname)
  } else {
    str_formulaA<-paste0(str_formulaA," + s(",cname,",bs='re')")
  }
}
rm(cname)
str_formulaA<-sub("~ [+]","~",str_formulaA)

set.seed(0)
drun2<-drun
drun2_tnorm<-drun_tnorm

bamres_list<-list()
bam_control<-gam.control()
bam_control$maxit<-50
bam_control$trace<-TRUE
for (i in 1:boundary_n) {
  #
  ind05_userid<-ind04_posset_bdr_userid[i]
  ind05c_neg<-which(as.character(drun2$label)=="0")
  ind05d_pos<-which(as.character(drun2$label)=="1")
  ind05b<-which((drun2$user_id==ind05_userid)&(as.character(drun2$label)=="1"))
  
  tempdist01<-colSums((drun2_tnorm[,ind05c_neg]-drun2_tnorm[,ind05b])^2)
  ind05e<-ind05c_neg[order(tempdist01)][1:sampling_size[1]]
  
  
  drun3<-drun2[c(ind05e,ind05d_pos),]
  
  # make sure that there is one observation for any factor level
  for (cname in setdiff(colnames(drun3),"label")) {
    if (is.factor(drun3[[cname]])) {
      temptab<-table(drun3[[cname]])
      ind06<-which(temptab<=0)
      if (length(ind06)>0) {
        for (j in 1:length(ind06)) {
          ind06b<-sample.int(nrow(drun3),1)
          drun3[ind06b,cname]<-names(ind06)[j]
        }
      }
    }
  
  }
  
  #
  bamres<-gam(formula(str_formulaA),data=drun3,family="binomial",
    weights=bam_weights,control=bam_control)
  drun3_pred<-predict(bamres,drun3,type="response")
  print(sprintf("Iteration %d:",i))
  print(sprintf("   Neg: (%9d / %9d) . Quant(0,..,100): %s",
    sum(drun3_pred[as.character(drun3$label)=="0"]>0.5),sum(as.character(drun3$label)=="0"),
    toString(round(quantile(drun3_pred[as.character(drun3$label)=="0"],probs=0:10/10),2) ) ) )
  print(sprintf("   Pos: (%9d / %9d) cases. Quant(0,..,100): %s",
    sum(drun3_pred[as.character(drun3$label)=="1"]<=0.5),sum(as.character(drun3$label)=="1"),
    toString(round(quantile(drun3_pred[as.character(drun3$label)=="1"],probs=0:10/10),2) ) ) )
  
  # remove some negative samples
  ind05f<-rep(TRUE,nrow(drun2))
  ind05f[ind05e]<-FALSE
  drun2<-drun2[ind05f,]
  drun2_tnorm<-drun2_tnorm[,ind05f]
  
  #
  bamres_list[[i]]<-bamres
}

res20_bamlist<-bamres_list
#
rm(drun,drun2,drun3,drun3_pred,bamres,bamres_list,drun_tnorm,drun2_tnorm)
rm(sampling_size,bam_weights,str_formulaA)
rm(i,j,temptab,cname,bam_control,boundary_n,tempdist01)
rm(list=ls(pattern="^ind0[1-6]"))



##########
d22_train<-d02_train[,]
d22_val<-d02_val[,]
d22_test<-d02_test[,,drop=FALSE]

for (i in 1:length(res20_bamlist)) {
  namei<-paste0("bam",i)
  d22_train[[namei]]<-predict(res20_bamlist[[i]],d02_train,type="response")
  d22_val[[namei]]<-predict(res20_bamlist[[i]],d02_val,type="response")
  d22_test[[namei]]<-predict(res20_bamlist[[i]],d02_test,type="response")
}
rm(i,namei)

for (cname in colnames(d22_train)) {
  dimnames(d22_train[[cname]])<-NULL
}
for (cname in colnames(d22_val)) {
  dimnames(d22_val[[cname]])<-NULL
}
for (cname in colnames(d22_test)) {
  dimnames(d22_test[[cname]])<-NULL
}
rm(cname)



##########
registerDoMC(cores = 3)

drun<-d22_train
drun$user_id<-NULL

# parallel
res22a <- foreach(ntree=rep(1000, 6), .packages='randomForest') %dopar% {
  
  # Downsampling can support more threads
  randomForest(label ~ ., data=drun,ntree=ntree,do.trace=10,
    sampsize=c(1700,1800))
}
rm(drun,temptab,classwt)
res22b<-do.call(combine,res22a)
rm(res22a)



#
res22b$lab1<-d22_train$label
res22b$predp1<-predict(res22b,newdata=d22_train,type="prob")
res22b$predr1<-predict(res22b,newdata=d22_train,type="response")
res22b$lab2<-d22_val$label
res22b$predp2<-predict(res22b,newdata=d22_val,type="prob")
res22b$predr2<-predict(res22b,newdata=d22_val,type="response")
res22b$lab3<-NULL
res22b$predp3<-predict(res22b,newdata=d22_test,type="prob")
res22b$predr3<-predict(res22b,newdata=d22_test,type="response")

#
res22b$auc1<-prj_get_auc(as.numeric(as.character(res22b$lab1)=="1"),
  res22b$predp1[,"1"])
res22b$confmat1<-prj_get_confmat(res22b$lab1,res22b$predr1)


res22b$auc2<-prj_get_auc(as.numeric(as.character(res22b$lab2)=="1"),
  res22b$predp2[,"1"])
res22b$confmat2<-prj_get_confmat(res22b$lab2,res22b$predr2)


#
res22b$confmat1/rowSums(res22b$confmat1)
c(sum(res22b$confmat1[c(1,4)])/sum(res22b$confmat1), sum(res22b$confmat1[c(2,3)])/sum(res22b$confmat1))
res22b$auc1


res22b$confmat2/rowSums(res22b$confmat2)
c(sum(res22b$confmat2[c(1,4)])/sum(res22b$confmat2), sum(res22b$confmat2[c(2,3)])/sum(res22b$confmat2))
res22b$auc2

##########
temptab<-cbind(d02_test$user_id,res22b$predp3[,"1"])
colnames(temptab)<-c("#USER_ID","SCORE")
write.table(temptab,file="task2_output_15a_res22b_sr100_ss17001800.txt",quote=FALSE,sep=",",row.names=FALSE)
rm(temptab)




################################################################################
# BAM + local BAM + RF
################################################################################
# load("tempB01_bamtree.RData")
# load("tempB01_localbam_tree.RData")

##########
d32_train<-d02_train[,]
d32_val<-d02_val[,]
d32_test<-d02_test[,,drop=FALSE]

for (i in 1:length(res10_bamlist)) {
  namei<-paste0("bam1_",i)
  d32_train[[namei]]<-predict(res10_bamlist[[i]],d02_train,type="response")
  d32_val[[namei]]<-predict(res10_bamlist[[i]],d02_val,type="response")
  d32_test[[namei]]<-predict(res10_bamlist[[i]],d02_test,type="response")
}
for (i in 1:length(res20_bamlist)) {
  namei<-paste0("bam2_",i)
  d32_train[[namei]]<-predict(res20_bamlist[[i]],d02_train,type="response")
  d32_val[[namei]]<-predict(res20_bamlist[[i]],d02_val,type="response")
  d32_test[[namei]]<-predict(res20_bamlist[[i]],d02_test,type="response")
}
rm(i,namei)

for (cname in colnames(d32_train)) {
  dimnames(d32_train[[cname]])<-NULL
}
for (cname in colnames(d32_val)) {
  dimnames(d32_val[[cname]])<-NULL
}
for (cname in colnames(d32_test)) {
  dimnames(d32_test[[cname]])<-NULL
}
rm(cname)

##########
# do this to free memory
# rm(list=c(ls(pattern="^res[0-2]"),ls(pattern="^d[1-2][0-9]_")))
# save(list=ls(),file="tempB02.RData")
# q()
# load("tempB02.RData")

##########
registerDoMC(cores = 3)

drun<-d32_train
drun$user_id<-NULL

# parallel
res32a <- foreach(ntree=rep(1000, 6), .packages='randomForest') %dopar% {
  
  # Downsampling can support more threads
  randomForest(label ~ ., data=drun,ntree=ntree,do.trace=10,
    sampsize=c(1800,1800))
}
rm(drun,temptab,classwt)
res32b<-do.call(combine,res32a)
rm(res32a)



#
res32b$lab1<-d32_train$label
res32b$predp1<-predict(res32b,newdata=d32_train,type="prob")
res32b$predr1<-predict(res32b,newdata=d32_train,type="response")
res32b$lab2<-d32_val$label
res32b$predp2<-predict(res32b,newdata=d32_val,type="prob")
res32b$predr2<-predict(res32b,newdata=d32_val,type="response")
res32b$lab3<-NULL
res32b$predp3<-predict(res32b,newdata=d32_test,type="prob")
res32b$predr3<-predict(res32b,newdata=d32_test,type="response")

#
res32b$auc1<-prj_get_auc(as.numeric(as.character(res32b$lab1)=="1"),
  res32b$predp1[,"1"])
res32b$confmat1<-prj_get_confmat(res32b$lab1,res32b$predr1)


res32b$auc2<-prj_get_auc(as.numeric(as.character(res32b$lab2)=="1"),
  res32b$predp2[,"1"])
res32b$confmat2<-prj_get_confmat(res32b$lab2,res32b$predr2)


#
res32b$confmat1/rowSums(res32b$confmat1)
c(sum(res32b$confmat1[c(1,4)])/sum(res32b$confmat1), sum(res32b$confmat1[c(2,3)])/sum(res32b$confmat1))
res32b$auc1


res32b$confmat2/rowSums(res32b$confmat2)
c(sum(res32b$confmat2[c(1,4)])/sum(res32b$confmat2), sum(res32b$confmat2[c(2,3)])/sum(res32b$confmat2))
res32b$auc2

##########
temptab<-cbind(d02_test$user_id,res32b$predp3[,"1"])
colnames(temptab)<-c("#USER_ID","SCORE")
write.table(temptab,file="task2_output_15a_res32b_sr100_ss18001800.txt",quote=FALSE,sep=",",row.names=FALSE)
rm(temptab)



################################################################################
# NN + local GAM
################################################################################


##########

res50<-prj_NNLG_get_numdata(d02_train,d02_val,d02_test)


##########
# dbscan: clustering results are not clear
if (FALSE) {
  drun<-res50$drun_train
  drun_tnorm<-res50$drun_tnorm_train
  ind01a_negall<-which(as.character(drun$label)=="0")
  ind01b_posall<-which(as.character(drun$label)=="1")
  drun_tnorm2a<-drun_tnorm[,ind01a_negall]
  drun_tnorm2b<-drun_tnorm[,ind01b_posall]
  
  #
  set.seed(0)
  drun_tnorm3b<-(drun_tnorm2b-rowMeans(drun_tnorm2b))/apply(drun_tnorm2b,MARGIN=1,FUN=sd)
  dist01<-quantile(colMeans(
      ( drun_tnorm3b[,sample.int(ncol(drun_tnorm3b),50000,replace=TRUE)] -
        drun_tnorm3b[,sample.int(ncol(drun_tnorm3b),50000,replace=TRUE)])^2 ),
    probs=0.5)
  
  dbscan_fit<-dbscan(t(drun_tnorm3b),eps=dist01,minPts=3)
  
  ind02a<-which(dbscan_fit$cluster==0)[1]
  quantile(colSums((drun_tnorm3b-drun_tnorm3b[,ind02a])^2),probs=0:10/100)
  
  rm(dist01,dbscan_fit,ind02a,drun_tnorm3b,drun,drun_tnorm,ind01a_negall,
    ind01b_posall,drun_tnorm2a,drun_tnorm2b)
}

##########
# mds on positive set: not much to interpret
if (FALSE) {
  drun<-res50$drun_train
  drun_tnorm<-res50$drun_tnorm_train
  ind01a_negall<-which(as.character(drun$label)=="0")
  ind01b_posall<-which(as.character(drun$label)=="1")
  drun_tnorm2a<-drun_tnorm[,ind01a_negall]
  drun_tnorm2b<-drun_tnorm[,ind01b_posall]
  #
  drun_tnorm3b<-(drun_tnorm2b-rowMeans(drun_tnorm2b))/apply(drun_tnorm2b,MARGIN=1,FUN=sd)
  dist02 <- dist(t(drun_tnorm3b)) # euclidean distances between the rows
  mds_fit <- cmdscale(dist02,eig=TRUE, k=2) # k is the number of dim
  
  ggplot(data=data.frame(x=mds_fit$points[,1],y=mds_fit$points[,2])) +
    geom_point(aes(x=x,y=y))
  
  rm(dist02,mds_fit,drun_tnorm3b)
  rm(drun,drun_tnorm,ind01a_negall,ind01b_posall,drun_tnorm2a,drun_tnorm2b)
}


##########
# choose some positive points as cluster heads
# cluster points to the nearest cluster heads by euclidean distance
prj_NNLG_cluster(clusnum_neg=25)


########## gam on each cluster
#
drun<-res50$drun_train
str_formulaA<-"label ~"
str_formulaB<-"label ~"
for (cname in setdiff(colnames(drun),c("label","user_id"))) {
  if (is.numeric(drun[[cname]])) {
    #
    str_formulaA<-paste0(str_formulaA," + s(",cname,",bs='cr',k=3)")
    # str_formulaA<-paste0(str_formulaA," + ",cname)
    
    #
    # str_formulaB<-paste0(str_formulaB," + s(",cname,",bs='cr',k=3)")
    str_formulaB<-paste0(str_formulaB," + ",cname)
  } else {
    #
    str_formulaA<-paste0(str_formulaA," + s(",cname,",bs='re')")
    # str_formulaA<-paste0(str_formulaA," + ",cname)
    
    #
    str_formulaB<-paste0(str_formulaB," + ",cname)
  }
}
rm(cname)
str_formulaA<-sub("~ [+]","~",str_formulaA)
str_formulaB<-sub("~ [+]","~",str_formulaB)
res50$str_formulaA<-str_formulaA
res50$str_formulaB<-str_formulaB


set.seed(0)
bam_control<-gam.control()
bam_control$maxit<-500
bam_control$trace<-TRUE

drun_train_pred<-rep(0,nrow(res50$drun_train))
drun_val_pred<-rep(0,nrow(res50$drun_val))
drun_test_pred<-rep(0,nrow(res50$drun_test))
gamreslist<-list()
weight_mul<-1.5
smooth_min_unum<-8



x1b_train<-as.matrix(drun3_train[,9:39])
x1b_val<-as.matrix(drun3_val[,9:39])


x2b_train<-as.matrix(drun3_train[,9:39])
x2b_val<-as.matrix(drun3_val[,9:39])



x1_head<-as.numeric(res50$drun_train[res50$clus_pos_headind[1],9:39])
x2_head<-as.numeric(res50$drun_train[res50$clus_pos_headind[2],9:39])

x1b_train_2head1<-colSums((t(x1b_train)-x1_head)^2)
x1b_val_2head1<-colSums((t(x1b_val)-x1_head)^2)

x1b_train_2head2<-colSums((t(x1b_train)-x2_head)^2)
x1b_val_2head2<-colSums((t(x1b_val)-x2_head)^2)

x2b_train_2head1<-colSums((t(x2b_train)-x1_head)^2)
x2b_val_2head1<-colSums((t(x2b_val)-x1_head)^2)

x2b_train_2head2<-colSums((t(x2b_train)-x2_head)^2)
x2b_val_2head2<-colSums((t(x2b_val)-x2_head)^2)

#
for (i in 1:length(res50$clus_pos_headind)) {
  # ind01a<-which((res50$clus_neg_memind_train==i)|(as.character(res50$drun_train$label)=="1"))
  ind01a<-which((res50$clus_neg_memind_train==i))
  ind01b<-which((res50$clus_neg_memind_val==i))
  ind01c<-which((res50$clus_neg_memind_test==i))
  
  drun3_train<-res50$drun_train[ind01a,]
  drun3_val<-res50$drun_val[ind01b,]
  drun3_test<-res50$drun_test[ind01c,]
  
  
  if (any(table(drun3_train$label)<=0)) {
    print(sprintf("Cluster %d: No positive sample",i))
    gamreslist[[i]]<-NULL
    
    # 
    drun3_train_pred<-rep(0,nrow(drun3_train))
    # drun_train_pred[ind01a]<-drun_train_pred[ind01a]+drun3_train_pred
    drun_train_pred[ind01a]<-drun3_train_pred
    print(sprintf("   Train_neg: (%9d / %9d) . Quant(0,..,100): %s",
      sum(drun3_train_pred[as.character(drun3_train$label)=="0"]>0.5),sum(as.character(drun3_train$label)=="0"),
      toString(round(quantile(drun3_train_pred[as.character(drun3_train$label)=="0"],probs=0:10/10),2) ) ) )
    print(sprintf("   Train_pos: (%9d / %9d) cases. Quant(0,..,100): %s",
      sum(drun3_train_pred[as.character(drun3_train$label)=="1"]<=0.5),sum(as.character(drun3_train$label)=="1"),
      toString(round(quantile(drun3_train_pred[as.character(drun3_train$label)=="1"],probs=0:10/10),2) ) ) )
    
    
    #
    drun3_val_pred<-rep(0,nrow(drun3_val))
    drun_val_pred[ind01b]<-drun3_val_pred
    print(sprintf("   Val_neg: (%9d / %9d) . Quant(0,..,100): %s",
      sum(drun3_val_pred[as.character(drun3_val$label)=="0"]>0.5),sum(as.character(drun3_val$label)=="0"),
      toString(round(quantile(drun3_val_pred[as.character(drun3_val$label)=="0"],probs=0:10/10),2) ) ) )
    print(sprintf("   Val_pos: (%9d / %9d) cases. Quant(0,..,100): %s",
      sum(drun3_val_pred[as.character(drun3_val$label)=="1"]<=0.5),sum(as.character(drun3_val$label)=="1"),
      toString(round(quantile(drun3_val_pred[as.character(drun3_val$label)=="1"],probs=0:10/10),2) ) ) )
    
    #
    drun3_test_pred<-rep(0,nrow(drun3_test))
    drun_test_pred[ind01c]<-drun3_test_pred
  }
  
  # make sure that there is one observation for any factor level
  # modify str_formulaA to satisfy the number of knots, if needed
  str_formulaA_i<-str_formulaA
  for (cname in setdiff(colnames(drun3_train),"label")) {
    if (is.factor(drun3_train[[cname]])) {
      temptab<-table(drun3_train[[cname]])
      ind02a<-which(temptab<=0)
      if (length(ind02a)>0) {
        for (j in 1:length(ind02a)) {
          ind02b<-sample.int(nrow(drun3_train),1)
          drun3_train[ind02b,cname]<-names(ind02a)[j]
        }
      }
    } else if ((is.numeric(drun3_train[[cname]]))&&(length(unique(drun3_train[[cname]]))<smooth_min_unum)) {
      str_formulaA_i<-gsub(paste0("s\\(",cname,",bs='cr',k=\\d+\\)"),cname,str_formulaA_i)
    }
    
  }
  
  
  #
  t01_temptab<-table(drun3_train$label)
  gam_weights<-rep(1,nrow(drun3_train))
  gam_weights[which(as.character(drun3_train$label)=="1")]<-ceiling(t01_temptab["0"]*weight_mul/t01_temptab["1"])
  
  
  
  print(sprintf("Cluster %d:",i))
  
  
  # gam
  if (FALSE) {
    print(sprintf("    Formula: %s",str_formulaA_i))
    tryCatch(
      gamres<-gam(formula(str_formulaA_i),data=drun3_train,family="binomial",
        weights=gam_weights,control=bam_control),
      error=function(e) {
        print("    Gam error: try linear formula")
        print(sprintf("    Formula: %s",str_formulaB))
        gamres<-gam(formula(str_formulaB),data=drun3_train,family="binomial",
          weights=gam_weights,control=bam_control)
      })
    
    #
    drun3_train_pred<-predict(gamres,drun3_train,type="response")
    # drun_train_pred[ind01a]<-drun_train_pred[ind01a]+drun3_train_pred
    
    #
    drun3_val_pred<-predict(gamres,drun3_val,type="response")
    
    #
    drun3_test_pred<-predict(gamres,drun3_test,type="response")
  }
  
  # randomForest
  if (TRUE) {
    gamres<-randomForest(label ~ ., data=drun3_train,ntree=1000,do.trace=10,
      sampsize=c(1800,3800))
    
    #
    drun3_train_pred<-predict(gamres,drun3_train,type="prob")[,"1"]
    
    #
    drun3_val_pred<-predict(gamres,drun3_val,type="prob")[,"1"]
    
    #
    drun3_test_pred<-predict(gamres,drun3_test,type="prob")[,"1"]
  }
  
  gamreslist[[i]]<-gamres
  
  
  #
  drun_train_pred[ind01a]<-drun3_train_pred
  print(sprintf("   Train_neg: (%9d / %9d) . Quant(0,..,100): %s",
    sum(drun3_train_pred[as.character(drun3_train$label)=="0"]>0.5),sum(as.character(drun3_train$label)=="0"),
    toString(round(quantile(drun3_train_pred[as.character(drun3_train$label)=="0"],probs=0:10/10),2) ) ) )
  print(sprintf("   Train_pos: (%9d / %9d) cases. Quant(0,..,100): %s",
    sum(drun3_train_pred[as.character(drun3_train$label)=="1"]<=0.5),sum(as.character(drun3_train$label)=="1"),
    toString(round(quantile(drun3_train_pred[as.character(drun3_train$label)=="1"],probs=0:10/10),2) ) ) )
  
  #
  drun_val_pred[ind01b]<-drun3_val_pred
  print(sprintf("   Val_neg: (%9d / %9d) . Quant(0,..,100): %s",
    sum(drun3_val_pred[as.character(drun3_val$label)=="0"]>0.5),sum(as.character(drun3_val$label)=="0"),
    toString(round(quantile(drun3_val_pred[as.character(drun3_val$label)=="0"],probs=0:10/10),2) ) ) )
  print(sprintf("   Val_pos: (%9d / %9d) cases. Quant(0,..,100): %s",
    sum(drun3_val_pred[as.character(drun3_val$label)=="1"]<=0.5),sum(as.character(drun3_val$label)=="1"),
    toString(round(quantile(drun3_val_pred[as.character(drun3_val$label)=="1"],probs=0:10/10),2) ) ) )
  
  #
  drun_test_pred[ind01c]<-drun3_test_pred
}










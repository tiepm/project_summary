
################################################################################

Sysconf<-list(
  FN_DATA01_BANK="./dataset/bank_info.csv",
  FN_DATA01_USER2014="./dataset/users_2014.csv",
  FN_DATA01_USER2015="./dataset/users_2015.csv",
  FN_DATA01_TRAIN2014="./dataset/train_2014.csv",
  FN_DATA01_TRAIN2015="./dataset/train_2015.csv")

################################################################################

#################################
prj_POSIX2date<-function(posix_str) {
    return( round((as.numeric(as.POSIXct(as.character(posix_str)))-as.numeric(as.POSIXct("2014-01-01")))/86400) )
}


#################################
# Perform some transformations 
#   format integer/numeric
#   convert posix str to date
# Add some variables c0106, etc
# Sort some data frames
prj_d01_preprocessing<-function(df,dfname) {
  names(df)<-unlist(lapply(names(df),FUN=tolower))
  
  list01<-c("geo_x","geo_y","loc_geo_x","loc_geo_y")
  list02<-c("geo_x","geo_y")
  if (dfname=="bank") {
    for (name in names(df)) {
      if (name=="poi_id") {
        df[,name]<-as.integer(df[,name])
      } else if (name%in%list01) {
        df[,name]<-as.numeric(df[,name])
      }
    }
    # order by poi_id
    orderind<-order(df$poi_id)
    df<-df[orderind,]
  } else if (dfname=="user") {
    for (name in names(df)) {
      if (name=="poi_id") {
        df[,name]<-as.integer(df[,name])
      } else if (name%in%list01) {
        df[,name]<-as.numeric(df[,name])
      }
    }
    
    # add combined factor for c201401/c201406 w201401/w201406
    df_ind01a<-grep("c201\\d01",names(df))
    df_ind01b<-grep("c201\\d06",names(df))
    df_ind01c<-grep("w201\\d01",names(df))
    df_ind01d<-grep("w201\\d06",names(df))
    if  ( (length(df_ind01a)>0)&&(length(df_ind01b)>0) ) {
      #
      df$c0106<-as.factor(paste(df[,df_ind01a],df[,df_ind01b],sep=""))
      df$w0106<-as.factor(paste(df[,df_ind01c],df[,df_ind01d],sep=""))
    }
    
    # add combined factor for c201407/c201412 w201407/w201412
    df_ind02a<-grep("c201\\d07",names(df))
    df_ind02b<-grep("c201\\d12",names(df))
    df_ind02c<-grep("w201\\d07",names(df))
    df_ind02d<-grep("w201\\d12",names(df))
    if  ( (length(df_ind02a)>0)&&(length(df_ind02b)>0) ) {
      #
      df$c0712<-as.factor(paste(df[,df_ind02a],df[,df_ind02b],sep=""))
      df$w0712<-as.factor(paste(df[,df_ind02c],df[,df_ind02d],sep=""))
    }
    
    # make age_cat as factor of four levels
    flnames_age<-levels(df$age_cat)
    if (!("-"%in%flnames_age)) {
      df$age_cat<-factor(df$age_cat,levels=c("-",flnames_age))
    }
    
    # process target_task_2
    if ("target_task_2"%in%names(df)) {
      # order by (target_task_2,user_id)
      key01<-as.character(df[,"target_task_2"])
      key01[key01=="-"]<-"z"
      key01<-paste(key01,sprintf("%012d",df$user_id),sep="")
      orderind<-order(key01)
      df<-df[orderind,]
      
      # 
      tt2<-as.character(df$target_task_2)
      df$tt2_14a<-(tt2>="2014.01.01")&(tt2<="2014.06.31")
      df$tt2_14b<-(tt2>="2014.07.01")&(tt2<="2014.12.31")
      df$tt2_15a<-(tt2>="2015.01.01")&(tt2<="2015.06.31")
      df$tt2_none<-(tt2=="-")
      
    } else {
      # order by user_id
      orderind<-order(df$user_id)
      df<-df[orderind,]
    }
    
  } else if (dfname=="train") {
    for (name in names(df)) {
      if (name=="poi_id") {
        df[,name]<-as.integer(df[,name])
      } else if (name=="date") {
        # t01<-as.numeric(as.POSIXct(as.character(df[,name])))
        # df[,name]<-round((t01-as.numeric(as.POSIXct("2014-01-01")))/86400)
        df[,name]<-prj_POSIX2date(df[,name])
      } else if (name%in%list02) {
        df[,name]<-as.character(df[,name])
        df[df[,name]=="-",name]<- "-1"
        df[,name]<-as.numeric(df[,name])
      }
    }
    # order by (user_id,date)
    orderind<-order(df$user_id+df$date/1000)
    df<-df[orderind,]
  }
  
  return(df)
}

#################################
# global function: match d01_train14$geo_x, geo_y of bank actvities to bank$geo_x, geo_y
prj_match_bank_poi_id<-function() {
  ind01a<-match(d01_train14$poi_id,d01_bank$poi_id)
  ind01b<-which(is.finite(ind01a))
  ind01c<-ind01a[ind01b]
  d01_train14$geo_x[ind01b]<-d01_bank$geo_x[ind01c]
  d01_train14$geo_y[ind01b]<-d01_bank$geo_y[ind01c]
}


#################################
# get the list of poiid of userlistA from d01_train satisfying
# userlistA is extractedf from d01_user with flag poi_id_userflag
prj_get_table_poiid_A<-function(d01_user,poi_id_userflag,
    d01_train,date_min,date_max) {
  
  #
  d01_train_lim<-d01_train[(d01_train$date>=date_min)&(d01_train$date<=date_max),]
  
  ind01_user_id<-d01_user$user_id[poi_id_userflag]
  ind02<-match(d01_train$user_id,ind01_user_id)
  ind02a<-which(is.finite(ind02))
  ind02b<-ind02[ind02a]
  
  #
  t01_poiid<-sort(table(d01_train14$poi_id[ind02a]),decreasing=TRUE)
  t01_poiid<-t01_poiid/sum(t01_poiid)
  
  return(t01_poiid)
}

# an ad hoc scheme to extract potential poiid from d01_user, d01_train
prj_get_potential_poiid<-function(d01_user,d01_train,poi_id_num) {
  
  #
  t01_poiid_tt214b_tr14a<-prj_get_table_poiid_A(d01_user14,d01_user14$tt2_14b,
    d01_train14,prj_POSIX2date("2014-01-01"),prj_POSIX2date("2014-06-30"))
  
  t01_poiid_negtt214b_tr14a<-prj_get_table_poiid_A(d01_user14,!d01_user14$tt2_14b,
    d01_train14,prj_POSIX2date("2014-01-01"),prj_POSIX2date("2014-06-30"))
  
  #
  t01_poiid_tt215a_tr14b<-prj_get_table_poiid_A(d01_user14,d01_user14$tt2_15a,
    d01_train14,prj_POSIX2date("2014-07-01"),prj_POSIX2date("2014-12-31"))
  
  t01_poiid_negtt215a_tr14b<-prj_get_table_poiid_A(d01_user14,!d01_user14$tt2_15a,
    d01_train14,prj_POSIX2date("2014-07-01"),prj_POSIX2date("2014-12-31"))
  
  
  #
  t02_poiid<-rep(0,max(d01_train$poi_id))
  t02_poiid[as.integer(names(t01_poiid_tt214b_tr14a))]<-t02_poiid[as.integer(names(t01_poiid_tt214b_tr14a))] +
    t01_poiid_tt214b_tr14a
  t02_poiid[as.integer(names(t01_poiid_tt215a_tr14b))]<-t02_poiid[as.integer(names(t01_poiid_tt215a_tr14b))] +
    t01_poiid_tt215a_tr14b
  
  t02_poiid[as.integer(names(t01_poiid_negtt214b_tr14a))]<-t02_poiid[as.integer(names(t01_poiid_negtt214b_tr14a))] -
    t01_poiid_negtt214b_tr14a
  t02_poiid[as.integer(names(t01_poiid_negtt215a_tr14b))]<-t02_poiid[as.integer(names(t01_poiid_negtt215a_tr14b))] -
    t01_poiid_negtt215a_tr14b
  
  #
  ind01<-order(t02_poiid,decreasing=TRUE)
  ind02a<-ind01[1:poi_id_num]
  ind02a<-ind02a[t02_poiid[ind02a]>0]
  ind01<-rev(ind01)
  ind02b<-ind01[1:poi_id_num]
  ind02b<-ind02b[t02_poiid[ind02b]<0]
  
  ind02c<-c(ind02a,ind02b)
  
  return(list(ind_all=ind02c,ind_pos=ind02a,ind_neg=ind02b))
}


#################################
# Summarize activity info of each user from date_min to date_max

prj_get_userB<-function(d01_user,d01_train,
  date_min,date_max,
  poiid) {
  
  user2_names<-c("user_id","avt_cnt",
    paste("ach_cnt",levels(d01_train$channel),sep="_"),
    paste("ati_cnt",levels(d01_train$time_cat),sep="_"),
    paste("aloc_cnt",levels(d01_train$loc_cat),sep="_"),
    paste("amc_cnt",levels(d01_train$mc_cat),sep="_"),
    paste("acrd_cnt",levels(d01_train$card_cat),sep="_"),
    paste("aamt_cnt",levels(d01_train$amt_cat),sep="_"))
  user2_names<-sub("-","na",user2_names)
  ind02<-matrix(c(grep("ach_cnt",user2_names)[1],length(grep("ach_cnt",user2_names)),
      grep("ati_cnt",user2_names)[1],length(grep("ati_cnt",user2_names)),
      grep("aloc_cnt",user2_names)[1],length(grep("aloc_cnt",user2_names)),
      grep("amc_cnt",user2_names)[1],length(grep("amc_cnt",user2_names)),
      grep("acrd_cnt",user2_names)[1],length(grep("acrd_cnt",user2_names)),
      grep("aamt_cnt",user2_names)[1],length(grep("aamt_cnt",user2_names))),
    nrow=2)
  ind02_cnames<-c("channel","time_cat","loc_cat","mc_cat","card_cat","amt_cat")
  dimnames(ind02)<-list(NULL,ind02_cnames)
  
  user2<-matrix(-1,nrow=nrow(d01_user),ncol=length(user2_names))
  dimnames(user2)<-list(NULL,user2_names)
  user2B<-matrix(0,nrow=nrow(d01_user),ncol=length(poiid))
  dimnames(user2B)<-list(NULL,poiid)
  poiid_names<-colnames(user2B)
  
  #
  d01_train2<-d01_train[(d01_train$date>=date_min)&(d01_train$date<=date_max),]
  ind10<-match(d01_train2$poi_id,poiid)
  d01_train3<-d01_train2[is.finite(ind10),c("user_id","poi_id")]
  
  # train2 indexing
  ind20a<-which( d01_train2$user_id>c(0,d01_train2$user_id[1:(length(d01_train2$user_id)-1)])  )
  ind20b<-c(ind20a[2:length(ind20a)]-1,length(d01_train2$user_id))
  ind20ab<-cbind(ind20a,ind20b)
  ind20c<- d01_train2$user_id[ind20a]
  ind20d<-new.env()
  for (i in 1:length(ind20c)) {
    ind20d[[as.character(ind20c[i])]]<-ind20ab[i,]
  }
  # train3 indexing
  ind30a<-which( d01_train3$user_id>c(0,d01_train3$user_id[1:(length(d01_train3$user_id)-1)])  )
  ind30b<-c(ind30a[2:length(ind30a)]-1,length(d01_train3$user_id))
  ind30ab<-cbind(ind30a,ind30b)
  ind30c<- d01_train3$user_id[ind30a]
  ind30d<-new.env()
  for (i in 1:length(ind30c)) {
    ind30d[[as.character(ind30c[i])]]<-ind30ab[i,]
  }
  
  for (i in 1:nrow(d01_user)) {
    if (i%%10000==0) {
      print(sprintf("Processed %d users",i))
    }
    
    useri<-d01_user[i,]
    uidi<-useri$user_id
    ind20di<-ind20d[[as.character(uidi)]]
    ind30di<-ind30d[[as.character(uidi)]]
    
    
    user2[i,1]<-uidi
    
    if (!is.null(ind20di)) {
      user2[i,2]<-ind20di[2]-ind20di[1]+1
      traini2<-d01_train2[ind20di[1]:ind20di[2],]
      for (cname in ind02_cnames) {
        ind02a<-ind02[1,cname]
        ind02b<-ind02[2,cname]
        
        user2[i,ind02a:(ind02a+ind02b-1)]<-table(traini2[,cname])
        
      }
    }
    
    # poiid
    if (!is.null(ind30di)) {
      traini3<-d01_train3[ind30di[1]:ind30di[2],]
      poiid01<-table(traini3$poi_id)
      user2B[i,names(poiid01)]<-poiid01
    }
  }
  
  colnames(user2B)<-paste0("poi_id_",colnames(user2B))
  user2C<-cbind(user2,user2B)
  
  return(user2C)
}


#################################
# compute summary statistics for users in each partition
prj_get_userstat_per_partition<-function(d01_user,d01_userB,
  d01_user_cname_extra,
  partition) {
  
  
  statlist<-list()
  for (i in 1:length(partition)) {
    listi<-list()
    pari<-partition[[i]]
    useri<-d01_user[pari,]
    userBi<-d01_userB[pari,]
    
    pari_n<-length(pari)
    for (cname in c("age_cat","loc_cat","inc_cat","gen",d01_user_cname_extra)) {
      listi[[cname]]<-table(useri[,cname])/pari_n
    }
    for (cname in c("loc_geo_x","loc_geo_y")) {
      listi[[cname]]<-quantile(useri[,cname],probs=0:10/10)
    }
    cnames2<-colnames(userBi)
    for (j in 2:length(cnames2)) {
      cname2<-cnames2[j]
      listi[[cname2]]<-quantile(userBi[,cname2],probs=0:10/10)
    }
    statlist[[i]]<-listi
  }
  
  
  statlist<-matrix(unlist(statlist),ncol=length(partition))
  dimnames(statlist)<-list(names(unlist(listi)),names(partition))
  
  return(statlist)
  
}

#################################
# merge user and userB into a single dataframe
#   df user: columns with names in cnames will be used and renamed
#   df userB: all columns will be used except columns user_id, ach_cnt_b (not available in the testset)
prj_get_userD_aux<-function(d01_user,d01_userB,
  cnames01,
  samratio,samseed) {
  
  
  cnames01a<-names(cnames01)
  cnames01b<-unlist(cnames01)
  cnames02<-setdiff(colnames(d01_userB),c("user_id","ach_cnt_b"))
  
  
  d02<-cbind(d01_user[,cnames01a],d01_userB[,cnames02])
  names(d02)<-c(cnames01b,cnames02)
  if ("gen"%in%names(d02)) {
    d02$gen<-as.factor(d02$gen)
  }
  if ("label"%in%names(d02)) {
    d02$label<-as.factor(as.numeric(d02$label))
  }
  
  if ((!is.null(samratio)) && (samratio>=0) && (samratio<=1)) {
    set.seed(samseed)
    ind02a<-as.logical(sample.int(2,nrow(d01_user),replace=TRUE,prob=c(samratio,1-samratio))-2)
    result<-list(d02[ind02a,],d02[!ind02a,],ind02a,!ind02a)
  } else {
    result<-list(d02,NULL,NULL,NULL)
  }
  
  return(result)
}



#################################
# global function: merge user and userB data and split into different datasets
prj_get_userD<-function(label_names,samratio_14a,samratio_14b,seed14a=0,seed14b=1) {
  
  res14a_cnames01<-list(user_id="user_id",label="label",c0106="c6m",w0106="w6m",
    age_cat="age_cat",loc_cat="loc_cat",inc_cat="inc_cat",gen="gen",
    loc_geo_x="geo_x_home",loc_geo_y="geo_y_home")
  names(res14a_cnames01)[2]<-label_names[1]
  res14a<-prj_get_userD_aux(d01_user14,d01_userB14a,
    cnames01=res14a_cnames01,
    samratio=samratio_14a,samseed=seed14a)
  
  # sum(abs(res14a[[1]]$label-d01_user14$tt2_14b))
  # sum(abs(as.numeric(res14a[[1]]$c6m)-as.numeric(d01_user14$c0106)))
  # sum(abs(res14a[[1]]$avt_cnt-d01_userB14a[,"avt_cnt"]))
  
  res14b_cnames01<-list(user_id="user_id",label="label",c0712="c6m",w0712="w6m",
    age_cat="age_cat",loc_cat="loc_cat",inc_cat="inc_cat",gen="gen",
    loc_geo_x="geo_x_home",loc_geo_y="geo_y_home")
  names(res14b_cnames01)[2]<-label_names[2]
  res14b<-prj_get_userD_aux(d01_user14,d01_userB14b,
    cnames01=res14b_cnames01,
    samratio=samratio_14b,samseed=seed14b)
  
  # sum(abs(res14b[[1]]$label-d01_user14$tt2_15a[res14b[[3]]]) )
  # sum(abs(as.numeric(res14b[[1]]$c6m)-as.numeric(d01_user14$c0712[res14b[[3]]])))
  # sum(abs(res14b[[1]]$avt_cnt-d01_userB14b[,"avt_cnt"][res14b[[3]]]))
  
  # sum(abs(res14b[[2]]$label-d01_user14$tt2_15a[res14b[[4]]]) )
  # sum(abs(as.numeric(res14b[[2]]$c6m)-as.numeric(d01_user14$c0712[res14b[[4]]])))
  # sum(abs(res14b[[2]]$avt_cnt-d01_userB14b[,"avt_cnt"][res14b[[4]]]))
  
  res15a<-prj_get_userD_aux(d01_user15,d01_userB15a,
    cnames01=list(user_id="user_id",c0106="c6m",w0106="w6m",
      age_cat="age_cat",loc_cat="loc_cat",inc_cat="inc_cat",gen="gen",
      loc_geo_x="geo_x_home",loc_geo_y="geo_y_home"),
    samratio=NULL,samseed=0)
  
  # sum(abs(res15a[[1]]$user_id-d01_user15$user_id))
  # sum(abs(as.numeric(res15a[[1]]$c6m)-as.numeric(d01_user15$c0106)))
  # sum(abs(res15a[[1]]$avt_cnt-d01_userB15a[,"avt_cnt"]))
  
  
  return(list(dtrain=rbind(res14a[[1]],res14b[[1]]),
      dval=rbind(res14a[[2]],res14b[[2]]),
      dtest=res15a[[1]]))
}


#################################
prj_get_auc<-function(poslab,posprob,remove_dup=TRUE) {
  
  ind01<-order(posprob)
  posprob<-posprob[ind01]
  poslab<-poslab[ind01]
  stat_fn<-cumsum(poslab)
  stat_tn<-cumsum(1-poslab)
  stat_tp<-sum(poslab)-stat_fn
  stat_fp<-sum(1-poslab)-stat_tn
  
  if (remove_dup==TRUE) {
    ind03<-which(posprob<c(posprob[2:length(posprob)],2))
    posprob<-posprob[ind03]
    stat_fn<-stat_fn[ind03]
    stat_tn<-stat_tn[ind03]
    stat_tp<-stat_tp[ind03]
    stat_fp<-stat_fp[ind03]
  }
  
  rrate_fp<-rev(stat_fp/sum(1-poslab))
  rrate_tp<-rev(stat_tp/sum(poslab))
  
  ind02a<-which(rrate_fp-c(0,rrate_fp[1:(length(rrate_fp)-1)])<0)
  ind02b<-which(rrate_tp-c(0,rrate_tp[1:(length(rrate_tp)-1)])<0)
  if ((length(ind02a)>0) && (length(ind02b)>=0)) {
    print("ERROR: rrate_fp or rrate_tp do not have non-negative increment")
    stop_here
  }
  
  dx<-diff(rrate_fp)
  auc<-sum(rrate_tp[1:(length(rrate_tp)-1)]*dx)
  
  return(auc)
}


#################################
prj_get_confmat<-function(lab,predr) {
  
  lab<-as.character(lab)
  predr<-as.character(predr)
  confmat<-matrix(0,nrow=2,ncol=2)
  confmat[1,1]<-sum((lab=="0")&(predr=="0"))
  confmat[1,2]<-sum((lab=="0")&(predr=="1"))
  confmat[2,1]<-sum((lab=="1")&(predr=="0"))
  confmat[2,2]<-sum((lab=="1")&(predr=="1"))
  dimnames(confmat)<-list(c("0","1"),c("0","1"))
  return(confmat)
}


################################################################################
# NN + local GAM
################################################################################



#################################
# 
# remove poi_id 
# divide data into two parts: 
#   drun_train, drun_val, drun_test: have factor variables
#   drun_tnorm_AAA, ...: do not have factors and are in matrix format
# drun_tnorm_AAA are normalized by colMeans of drun_tnorm_train
prj_NNLG_get_numdata<-function(dtemp_train,dtemp_val,dtemp_test) {
  
  #
  get_drun<-function(dtemp_train) {
    drun<-dtemp_train
    drun_rm<-c(
      colnames(drun)[grep("poi_id",colnames(drun))]
      # colnames(drun)[setdiff(grep("poi_id",colnames(drun)),grep("poi_id_9119",colnames(drun)))]
      )
    for (cname in drun_rm) {
      drun[[cname]]<-NULL
    }
    rm(cname,drun_rm)
    
    return(drun)
  }
  
  # clone and remove some unwanted features
  drun_train<-get_drun(dtemp_train)
  drun_val<-get_drun(dtemp_val)
  drun_test<-get_drun(dtemp_test)
  
  #
  drun_tnorm_train<-drun_train
  drun_tnorm_train$user_id<-NULL
  for (cname in colnames(drun_tnorm_train)) {
    if (is.factor(drun_tnorm_train[[cname]])) {
      drun_tnorm_train[[cname]]<-NULL
    }
  }
  drun_tnorm_train<-as.matrix(drun_tnorm_train)
  drun_tnorm_cm<- colMeans(drun_tnorm_train)
  drun_tnorm_sd<- apply(drun_tnorm_train,MARGIN=2,FUN=sd)
  drun_tnorm_train<-(t(drun_tnorm_train)-drun_tnorm_cm)/drun_tnorm_sd
  
  #
  ind01<-rownames(drun_tnorm_train)
  drun_tnorm_val<-as.matrix(drun_val[,ind01])
  drun_tnorm_val<-(t(drun_tnorm_val)-drun_tnorm_cm)/drun_tnorm_sd
  
  #
  drun_tnorm_test<-as.matrix(drun_test[,ind01])
  drun_tnorm_test<-(t(drun_tnorm_test)-drun_tnorm_cm)/drun_tnorm_sd
  
  #
  res<-list(drun_train=drun_train,drun_val=drun_val,drun_test=drun_test,
      drun_tnorm_train=drun_tnorm_train,drun_tnorm_val=drun_tnorm_val,
      drun_tnorm_test=drun_tnorm_test)
  
  return(res)
}



#################################
prj_NNLG_cluster<-function(clusnum_neg) {
  
  # global var: res50
  res50$clusnum_neg<<-clusnum_neg
  
  #
  drun<-res50$drun_train
  drun_tnorm<-res50$drun_tnorm_train
  ind01a_negall<-which(as.character(drun$label)=="0")
  ind01b_posall<-which(as.character(drun$label)=="1")
  drun_tnorm2a<-drun_tnorm[,ind01a_negall]
  drun_tnorm2b<-drun_tnorm[,ind01b_posall]
  
  # assign each negative point to the nearest positive point
  ind03a<-rep(0,ncol(drun_tnorm2a))
  for (i in 1:ncol(drun_tnorm2a)) {
    ind03a[i]<-which.min(colSums((drun_tnorm2b-drun_tnorm2a[,i])^2))
    if (i%%10000==0) {
      print(sprintf("Process %i items",i))
    }
  }
  rm(i)
  
  # select clusnum_neg positive points with the most associated negative points
  ind03b<-table(ind03a)
  ind03c<-as.integer(names(ind03b)[order(ind03b,decreasing=TRUE)[1:clusnum_neg]])
  ind03d<-sort(ind01b_posall[ind03c])
  
  #
  res50$clus_pos_headind<<-ind03d
  
  
  # re-run the assignment with only the selected positive points
  get_memind<-function(t01_head,t01_drun_tnorm) {
    ind04a<-rep(0,ncol(t01_drun_tnorm))
    for (i in 1:ncol(t01_drun_tnorm)) {
      ind04a[i]<-which.min(colSums((t01_head-t01_drun_tnorm[,i])^2))
      if (i%%10000==0) {
        print(sprintf("Process %i items",i))
      }
    }
    return(ind04a)
  }
  
  
  t01_head<-drun_tnorm[,ind03d]
  res50$clus_neg_memind_train<<-get_memind(t01_head,res50$drun_tnorm_train)
  res50$clus_neg_memind_val<<-get_memind(t01_head,res50$drun_tnorm_val)
  res50$clus_neg_memind_test<<-get_memind(t01_head,res50$drun_tnorm_test)
  
  return()
}



#################################
misc_df_conv_fac2num<-function(df,
    skip_colnames,ignore_colnames,
    return_matrix=FALSE) {
  
  df2<-df[,c(),drop=FALSE]
  for (colname in colnames(df)) {
    if (colname%in%skip_colnames) {
      next
    }
    
    if ((colname%in%ignore_colnames)||(is.numeric(df[[colname]]))){
      df2[[colname]]<-df[[colname]]
      next
    }
    
    if (is.factor(df[[colname]])) {
      cnlevels<-levels(df[[colname]])
      cnlevels2<-paste(colname,cnlevels,sep="_")
      for (i in 1:length(cnlevels)) {
        df2[[cnlevels2[i]]]<-as.numeric(df[[colname]]==cnlevels[i])
      }
    }
  }
  
  if (return_matrix==TRUE) {
    df2<-as.matrix(df2)
  }
  
  return(df2)
}


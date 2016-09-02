--------------------------------------------------------------------------------
to=require 'torch'
cto=require 'cutorch'
ag=require 'autograd'
nn=require 'nn'
cunn=require 'cunn'
csv=require 'csvigo'
require 'lfs';
gp=require 'gnuplot'
date=require "date"


--  Some modifications to 3rd-libraries
dofile("./001_base.lua")

--
to.setdefaulttensortype('torch.FloatTensor')


--------------------------------------------------------------------------------
--  Main
--------------------------------------------------------------------------------
metadata={}
metadata.data01_path="./data"
metadata.time_period={"2013-01-01 00:00:00", "2016-01-31 23:00:00"}
--  metadata.test_periods[2] is the original test set of R code
metadata.test_periods={{"2015-11-01 00:00:00", "2015-12-31 23:00:00"},
  {"2015-10-01 00:00:00", "2015-10-31 23:00:00"}}
metadata.train_periods={}
for k, v in pairs(metadata.test_periods) do
  metadata.train_periods[k]={}
  metadata.train_periods[k][1]=metadata.time_period[1]
  metadata.train_periods[k][2]=date(v[1]):addhours(-1):fmt("%Y-%m-%d %H:%M:%S")
  if metadata.train_periods[k][2]>metadata.time_period[2] then 
    metadata.train_periods[k][2]=metadata.time_period[2]
  end
end

--  Get the list of entities
metadata.entities= prj:get_metadata_entities(metadata.data01_path)

--  Assumptions
--    Data is hourly and contiguous (no missing/redundant values)
d01=prj:get_entity_data(metadata.entities[1])
d01.traintest_ind=2 --  the index in use of metadata.test_periods/metadata.train_periods


--  prediction is made at 8am everyday
--  So, may get historical series (0am,8am) and future series (8am,nextday:8am)
--  or get (0am:11pm) and (nextday:0am,nextday:11pm)
prj:get_targetsegments_on_signature(d01,
    "hour",23,-23,0,1,24,
    "demand","time",
    metadata.test_periods[d01.traintest_ind], metadata.train_periods[d01.traintest_ind],
    false);


--------------------------------------------------------------------------------
-- Misc
--------------------------------------------------------------------------------

--------------------------------------------------
--  plot: daily series
--------------------------------------------------
do 
  if false then
    local x1_prev=to.range(1,d01.demand_prev:size(2))
    local x1_next=to.range(1,d01.demand_next:size(2))
    
    local tab_plot_prev={}
    local tab_plot_next={}
    local n1=41 -- from/to indices of plotting series
    local n2=50 -- d01.demand_prev:size(1)
    
    for i=n1,n2 do
      tab_plot_prev[#tab_plot_prev+1]={tostring(i),x1_prev,d01.demand_prev[{{i}}]:resize(d01.demand_prev:size(2)),'-'}
      tab_plot_next[#tab_plot_next+1]={tostring(i),x1_next,d01.demand_next[{{i}}]:resize(d01.demand_next:size(2)),'-'}
    end
    
    gp.figure(1)
    gp.plot(tab_plot_prev)
    gp.figure(2)
    gp.plot(tab_plot_next)
  end
end

--------------------------------------------------
--  Test cutorch 
--  The bigger Y, the higher speedup factor cutorch/torch
--  if Y is small, cutorch may be slower than torch (calling overhead)
--------------------------------------------------

do
  if false then
    local Y, Y1
    local timer
    
    Y=to.randn(10000,1000)
    
    Y1=Y:cuda()
    
    timer=to.Timer()
    print(timer:time().real)
    for i=1,1000 do
      Y:add(i)
      Y:add(-i)
    end
    print(timer:time().real)
    
    timer:reset()
    print(timer:time().real)
    for i=1,1000 do
      Y1:add(i)
      Y1:add(-i)
    end
    print(timer:time().real)
  end
  
end


--------------------------------------------------
-- sparse coding
--------------------------------------------------

--  Get the data
--  demand_prev
Y_orig=d01.demand_prev[d01.demand_cp_train:resize(d01.demand_cp_train:nElement(),1):expandAs(d01.demand_prev)]
Y_orig=Y_orig:resize(d01.demand_cp_train:sum(),d01.demand_prev:size(2)):t();

Y_test_orig=d01.demand_prev[d01.demand_cp_test:resize(d01.demand_cp_test:nElement(),1):expandAs(d01.demand_prev)]
Y_test_orig=Y_test_orig:resize(d01.demand_cp_test:sum(),d01.demand_prev:size(2)):t();


--  demand_next
-- Y_orig=d01.demand_next[d01.demand_cp_train:resize(d01.demand_cp_train:nElement(),1):expandAs(d01.demand_next)]
-- Y_orig=Y_orig:resize(d01.demand_cp_train:sum(),d01.demand_next:size(2)):t();
-- 
-- Y_test_orig=d01.demand_next[d01.demand_cp_test:resize(d01.demand_cp_test:nElement(),1):expandAs(d01.demand_next)]
-- Y_test_orig=Y_test_orig:resize(d01.demand_cp_test:sum(),d01.demand_next:size(2)):t();

--  Input data transformation
center=false
if (center==true) then
  Y_orig_mean=Y_orig:mean(2):resize(Y_orig:size(1),1)
  Y_orig:add(-Y_orig_mean:expandAs(Y_orig))
  Y_test_orig:add(-Y_orig_mean:expandAs(Y_test_orig))
end


--  Set up arguent for sparse coding
--  state data: include everything
sdata_CPU={}

sdata_CPU.seed=1
--  cuda is not good for this algorithm (iterative)
--  This option is NOT FULLY TESTED YET due to some updates
sdata_CPU.useCuda=false 
sdata_CPU.lambdaB=2.0
sdata_CPU.lambdaC=2.0
sdata_CPU.m=100

--  estimate one column of Beta at one time (For FISTA, columnwise optim is usually better)
sdata_CPU.Beta_columnwise_est=true 

--  FISTALS parameters for the estimation of Beta
--  optional. See optim: FISTALS
sdata_CPU.Beta_Fista_par={
  L=1,Lstep=1.5,  -- stepsize=1/L; Lnext=Lcurrent.Lstep
  maxiter=5,  
  maxline=50,  -- number of linesearch iterations within each main iteration. 
  errthres=1e-4,
  doFistaUpdate=true,
  verbose=false}

--  fixed user defined dictionary
if (center==true) then
  sdata_CPU.X1=to.ones(Y_orig:size(1),1)
else
  sdata_CPU.X1=to.cat({Y_orig:mean(2), to.ones(Y_orig:size(1),1)})
end
--  sdata_CPU.X1=nil

--  The penalty type for each column of X2 (All are scaled with lambdaC)
--    "L2-RW" (default): L2 Random walk, starting at zero.
--    "L1-RW": L1 Random walk, starting at zero
sdata_CPU.X2_penaltyType="L2-RW"

--  estX_L1RW_npL is only useful for X2_penaltyType="L1-RW"
--    number of iterations for each pL step
sdata_CPU.estX_L1RW_npL=3

--  FISTALS parameters for the estimation of X2 of penalty type "L1-RW"
--  optional. See optim: FISTALS
sdata_CPU.estX_L1RW_Fista_par={
  L=1,Lstep=1.5,  -- stepsize=1/L; Lnext=Lcurrent.Lstep
  maxiter=5,  
  maxline=50,  -- number of linesearch iterations within each main iteration. 
  errthres=1e-4,
  doFistaUpdate=true,
  verbose=false}

--  Initialize a sparse coding object/table sdata_CPU
--    This object holds all the information of sparse coding
met:sparsecode_ts(Y_orig,sdata_CPU,Y_test_orig,false);


--  Function: do sparse coding in k iterations
-- sdata_CPU:estBetaX_dokiter(self,50,10,true)

--  Function: do sparse coding on train data by procedure A
sdata_CPU:estBetaX_dowhileA(10,500,10,50,1,0.25,true)

--  Function: do sparse coding on train data 
--    and validate on test data by procedure B
-- sdata_new=sdata_CPU:estBetaX_dowhileB(false, 0.5, 0.5, 0.1, 0.1, 
--   0, 0, 
--   10, 200, 10, 200, 
--   10, 20, 1, 0.25, 50)

--  Function: estimate Beta_new for Y_new, using the current dictionary 
Betanew=sdata_CPU:estBeta_for_Ynew(sdata_CPU.Y_test:clone(),50)

--  Function: clone a sdata_new from sdata_old
--    These two sdata shares some variable such as Y_train, Y_test
-- sdata_clone=met:sparsecode_ts(Y_orig,sdata_CPU,Y_test_orig,{X2_addcol=5});


--  Function: swith to training dataset. Do not re-estimate Beta
--    Print out summary statistics for training dataset
sdata_CPU:switch_YBeta("train",false,100)
misc:strt(sdata_CPU:SC_eval())


--  Function: swith to test dataset. Re-estimate Beta
--    Print out summary statistics for test dataset
sdata_CPU:switch_YBeta("test",true,100)
misc:strt(sdata_CPU:SC_eval())

--  Function: plot Y, X and Beta
sdata_CPU:plotYXBeta(1,2,3,"train",{401,430})
sdata_CPU:plotYXBeta(1,2,3,"test")
sdata_CPU:plotYXBeta(1,2,3,"new",nil,sdata_CPU.Y_test:clone(), Betanew)







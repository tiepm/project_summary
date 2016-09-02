--------------------------------------------------------------------------------
to=require 'torch'
cto=require 'cutorch'
ag=require 'autograd'
nn=require 'nn'
csv=require 'csvigo'
require 'lfs';
gp=require 'gnuplot'
date=require "date"
require 'string.color'

--------------------------------------------------------------------------------
--  general misc functions
--------------------------------------------------------------------------------
misc={}

--  summarize a table
function misc:strt(tab,num)
  
  if num==nil then num=10 end
  local tab2={}
  
  function get_tensorsize_str(x)
    local s1="size "
    local xt=x:size():totable()
    table.foreach(xt,function(k,v) s1=s1.."x"..tostring(v) return end)
    return s1
  end
  
  for k,v in pairs(tab) do
    --
    if type(v)=="table"then
      i2=0
      local tab2_v=""
      
      
      for k2,v2 in pairs(v) do
        if type(v2)=="table" then
          tab2_v=tab2_v .. string.format("%s:%s, ", k2,"TABLE")
        elseif type(v2)=="userdata"then
          --  Assume v2.type() doesn't require input
          if (v2.type and (type(v2.type)=="function") and string.match(v2:type(), "Tensor")) then
            tab2_v=tab2_v .. string.format("%s:%s-%s, ", k2, v2:type(), get_tensorsize_str(v2) )
          else
            tab2_v=tab2_v .. string.format("%s:%s, ", k2,"USERDATA")
          end
        else
          tab2_v=tab2_v .. string.format("%s:%s, ", k2,tostring(v2))
        end
        i2=i2+1
        if (i2>=num) then
          tab2_v=tab2_v .. "..."
          break
        end
      end
      
      tab2[k]=string.format("[%s]{%s}, ", string.match(tostring(v),"0x%w+"),tab2_v)
    elseif type(v)=="userdata"then
      --  Assume v2.type() doesn't require input
      if (v.type and (type(v.type)=="function") and string.match(v:type(), "Tensor")) then
        tab2[k]=string.format("[%s] %s-%s, ", 
          string.match(tostring(v:data()),"0x%w+"), 
          v:type(), 
          get_tensorsize_str(v) )
      else
        tab2[k]=v
      end
    else
      tab2[k]=v
    end
    
  end
  tab2_keys={}
  table.foreach(tab2, function(k,v) tab2_keys[#tab2_keys+1]=k end)
  table.sort(tab2_keys)
  for k1, k2 in pairs(tab2_keys) do
    v2=tab2[k2]
    if (type(v2)=="string") then v2s=string.color(tostring(v2),"green")
    elseif (type(v2)=="number") then v2s=string.color(tostring(v2),"blue")
    elseif (type(v2)=="function") then v2s=string.color(tostring(v2),"cyan")
    else v2s=string.color(tostring(v2),"blue")
    end
    print( string.format("  %s: %s",k2,v2s)  )
  end
end

--  convert anything to one or zero
--  this is convenient but slow
function misc:tooz(var)
  return var and 1 or 0
end



--------------------------------------------------------------------------------
-- My optim
--------------------------------------------------------------------------------
optim=optim or {}

--------------------------------------------------------------------------------
-- Modifying optim.FistaLS
--------------------------------------------------------------------------------
function optim.FistaLS(f, g, pl, xinit, params)
  
  local params = params or {}
  local L = params.L or 0.1
  local Lstep = params.Lstep or 1.5
  local maxiter = params.maxiter or 50
  local maxline = params.maxline or 20
  local errthres = params.errthres or 1e-4
  local doFistaUpdate = params.doFistaUpdate
  local verbose = params.verbose 
  
  -- temporary allocations
  params.xkm = params.xkm or torch.Tensor()
  params.y   = params.y   or torch.Tensor()
  params.ply = params.ply or torch.Tensor()
  local xkm = params.xkm  -- previous iteration
  local y   = params.y    -- fista iteration
  local ply = params.ply  -- soft shrinked y
  
  
  local xk = xinit
  -- <tiepm-mod>
  -- we start from all zeros
  -- xkm:resizeAs(xk):zero()
  -- ply:resizeAs(xk):zero()
  -- y:resizeAs(xk):zero()
  xkm=xinit:clone()
  ply:resizeAs(xk):zero()
  y=xinit:clone()
  -- </tiepm-mod>
  
  local history = {} -- keep track of stuff
  local niter = 0    -- number of iterations done
  local converged = false  -- are we done?
  local tk = 1      -- momentum param for FISTA
  local tkp = 0
  
  -- <tiepm-mod>
  -- local gy = g(y)
  -- </tiepm-mod>
  local fval = math.huge -- fval = f+g
  while not converged and niter < maxiter do
    
    -- run through smooth function (code is input, input is target)
    -- get derivatives from smooth function
    local fy,gfy = f(y,'dx')
    --local gfy = f(y)
    -- <tiepm-mod>
    local gy = g(y)
    -- </tiepm-mod>
    
    local fply = 0
    local gply = 0
    local Q = 0
    
    ----------------------------------------------
    -- do line search to find new current location starting from fista loc
    local nline = 0
    local linesearchdone = false
    while not linesearchdone do
      -- take a step in gradient direction of smooth function
      ply:copy(y)
      ply:add(-1/L,gfy)
      
      -- and solve for minimum of auxiliary problem
      pl(ply,L)
      -- this is candidate for new current iteration
      xk:copy(ply)
      
      -- evaluate this point F(ply)
      fply = f(ply)
      --<tiepm-mod>
      gply = g(ply)
      --</tiepm-mod>
      
      
      -- ply - y
      ply:add(-1, y)
      -- <ply-y , \Grad(f(y))>
      local Q2 = gfy:dot(ply)
      -- L/2 ||beta-y||^2
      local Q3 = L/2 * ply:dot(ply)
      -- Q(beta,y) = F(y) + <beta-y , \Grad(F(y))> + L/2||beta-y||^2 + G(beta)
      -- <tiepm-mod>
      Q = fy + Q2 + Q3 + gply
      -- </tiepm-mod>
      
      if verbose then
        print(string.format('nline=%d L=%g fply=%g Q=%g fy=%g Q2=%g Q3=%g gply=%g',nline,L,fply,Q,fy,Q2,Q3,gply))
        print(string.format('  fy=%g gy=%g fply=%g gply=%g',fy,gy,fply,gply))
      end
      -- check if F(beta) < Q(pl(y),\t)
      -- <tiepm-mod>: Need to make sure that Q(beta,y)<Q(y,y) or fply+gply<f
      -- if fply <= Q --and Fply + Gply <= F then
      
      -- Note that in the paper Q(y,ply) is guaranteed to be smaller than Q(y,y) thanks to the operator argmin
      -- However, we just do a gradient step size+pL; and do not guarantee the above condition
      -- Hence, we need to check that Q(y,ply)<=Q(y,y), or equivalently Q<=fy+gy
      if fply+gply <= Q and Q<fy+gy then
        -- </tiepm-mod>
        -- now evaluate G here
        linesearchdone = true
      elseif  nline >= maxline then
        linesearchdone = true
        xk:copy(xkm) -- if we can't find a better point, current iter = previous iter
        --print('oops')
      else
        L = L * Lstep
      end
      nline = nline + 1
    end
    -- end line search
    ---------------------------------------------
    
    ---------------------------------------------
    -- FISTA
    ---------------------------------------------
    if doFistaUpdate then
       -- do the FISTA step
       tkp = (1 + math.sqrt(1 + 4*tk*tk)) / 2
       -- x(k-1) = x(k-1) - x(k)
       xkm:add(-1,xk)
       -- y(k+1) = x(k) + (1-t(k)/t(k+1))*(x(k-1)-x(k))
       y:copy(xk)
       y:add( (1-tk)/tkp , xkm)
       -- store for next iterations
       -- x(k-1) = x(k)
       xkm:copy(xk)
    else
       y:copy(xk)
    end
    -- t(k) = t(k+1)
    tk = tkp
    fply = f(y)
    gply = g(y)
    if verbose then
      print(string.format('iter=%d eold=%g enew=%g',niter,fval,fply+gply))
      print("------------------------------------------------------------")
    end
    
    niter = niter + 1
    
    -- bookeeping
    fval = fply + gply
    history[niter] = {}
    history[niter].nline = nline
    history[niter].L  = L
    history[niter].F  = fval
    history[niter].Fply = fply
    history[niter].Gply = gply
    history[niter].Q  = Q
    params.L = L
    if verbose then
      history[niter].xk = xk:clone()
      history[niter].y  = y:clone()
    end
    
    -- are we done?
    if niter > 1 and math.abs(history[niter].F - history[niter-1].F) <= errthres then
      converged = true
      xinit:copy(y)
      return y,history
    end
    
    if niter >= maxiter then
      xinit:copy(y)
      return y,history
    end
    
    --if niter > 1 and history[niter].F > history[niter-1].F then
    --print(niter, 'This was supposed to be a convex function, we are going up')
    --converged = true
    --return xk,history
    --end
  end
  error('not supposed to be here')
end


--------------------------------------------------------------------------------
--  Modified version: work with matrix xinit
--    Assume that xinit is a tensor of size s1
--    f(x2), g(x2), pl(gx,L) are functions with x2,gx are tensor of size s1
--------------------------------------------------------------------------------
function optim.FistaLS_Tensor(f, g, pl, xinit, params)
  
  local params = params or {}
  local L = params.L or 0.1
  local Lstep = params.Lstep or 1.5
  local maxiter = params.maxiter or 50
  local maxline = params.maxline or 20
  local errthres = params.errthres or 1e-4
  local doFistaUpdate = params.doFistaUpdate
  local verbose = params.verbose 
  
  -- temporary allocations
  params.xkm = params.xkm or torch.Tensor()
  params.y   = params.y   or torch.Tensor()
  params.ply = params.ply or torch.Tensor()
  local xkm = params.xkm  -- previous iteration
  local y   = params.y    -- fista iteration
  local ply = params.ply  -- soft shrinked y
  
  -- we start from all zeros
  local xk = xinit
  -- <tiepm-mod>
  -- xkm:resizeAs(xk):zero()
  -- ply:resizeAs(xk):zero()
  -- y:resizeAs(xk):zero()
  xkm=xinit:clone()
  ply:resizeAs(xk):zero()
  y=xinit:clone()
  -- </tiepm-mod>
  
  local history = {} -- keep track of stuff
  local niter = 0    -- number of iterations done
  local converged = false  -- are we done?
  local tk = 1      -- momentum param for FISTA
  local tkp = 0
  
  -- <tiepm-mod>
  -- local gy = g(y)
  -- </tiepm-mod>
  local fval = math.huge -- fval = f+g
  niter=0
  while not converged and niter < maxiter do
    
    -- run through smooth function (code is input, input is target)
    -- get derivatives from smooth function
    local fy,gfy = f(y,'dx')
    --local gfy = f(y)
    -- <tiepm-mod>
    local gy = g(y)
    -- </tiepm-mod>
    
    local fply = 0
    local gply = 0
    local Q = 0
    
    ----------------------------------------------
    -- do line search to find new current location starting from fista loc
    local nline = 0
    local linesearchdone = false
    while not linesearchdone do
      -- take a step in gradient direction of smooth function
      ply:copy(y)
      ply:add(-1/L,gfy)
      
      -- and solve for minimum of auxiliary problem
      pl(ply,L)
      -- this is candidate for new current iteration
      xk:copy(ply)
      
      -- evaluate this point F(ply)
      fply = f(ply)
      --<tiepm-mod>
      gply = g(ply)
      --</tiepm-mod>
      
      
      -- ply - y
      ply:add(-1, y)
      -- <ply-y , \Grad(f(y))>
      --<tiepm-mod>
      -- local Q2 = gfy:dot(ply)
      local Q2 = to.cmul(gfy,ply):sum()
      --</tiepm-mod>
      -- L/2 ||beta-y||^2
      local Q3 = L/2 * (ply:norm(2)^2)
      -- Q(beta,y) = F(y) + <beta-y , \Grad(F(y))> + L/2||beta-y||^2 + G(beta)
      -- <tiepm-mod>
      Q = fy + Q2 + Q3 + gply
      -- </tiepm-mod>
      
      if verbose then
        print(string.format('nline=%d L=%g fply=%g Q=%g fy=%g Q2=%g Q3=%g gply=%g',nline,L,fply,Q,fy,Q2,Q3,gply))
        print(string.format('  fy=%g gy=%g fply=%g gply=%g',fy,gy,fply,gply))
      end
      -- check if F(beta) < Q(pl(y),\t)
      -- <tiepm-mod>: Need to make sure that Q(beta,y)<Q(y,y) or fply+gply<f
      -- if fply <= Q --and Fply + Gply <= F then
      if fply+gply <= Q and Q<fy+gy then
        -- </tiepm-mod>
        -- now evaluate G here
        linesearchdone = true
      elseif  nline >= maxline then
        linesearchdone = true
        xk:copy(xkm) -- if we can't find a better point, current iter = previous iter
        --print('oops')
      else
        L = L * Lstep
      end
      nline = nline + 1
    end
    -- end line search
    ---------------------------------------------
    
    ---------------------------------------------
    -- FISTA
    ---------------------------------------------
    if doFistaUpdate then
       -- do the FISTA step
       tkp = (1 + math.sqrt(1 + 4*tk*tk)) / 2
       -- x(k-1) = x(k-1) - x(k)
       xkm:add(-1,xk)
       -- y(k+1) = x(k) + (1-t(k)/t(k+1))*(x(k-1)-x(k))
       y:copy(xk)
       y:add( (1-tk)/tkp , xkm)
       -- store for next iterations
       -- x(k-1) = x(k)
       xkm:copy(xk)
    else
       y:copy(xk)
    end
    -- t(k) = t(k+1)
    tk = tkp
    fply = f(y)
    gply = g(y)
    if verbose then
      print(string.format('iter=%d eold=%g enew=%g',niter,fval,fply+gply))
      print("------------------------------------------------------------")
    end
    
    niter = niter + 1
    
    -- bookeeping
    fval = fply + gply
    history[niter] = {}
    history[niter].nline = nline
    history[niter].L  = L
    history[niter].F  = fval
    history[niter].Fply = fply
    history[niter].Gply = gply
    history[niter].Q  = Q
    params.L = L
    if verbose then
      history[niter].xk = xk:clone()
      history[niter].y  = y:clone()
    end
    
    -- are we done?
    if niter > 1 and math.abs(history[niter].F - history[niter-1].F) <= errthres then
      converged = true
      xinit:copy(y)
      return y,history
    end
    
    if niter >= maxiter then
      xinit:copy(y)
      return y,history
    end
    
    --if niter > 1 and history[niter].F > history[niter-1].F then
    --print(niter, 'This was supposed to be a convex function, we are going up')
    --converged = true
    --return xk,history
    --end
  end
  error('not supposed to be here')
end





--------------------------------------------------------------------------------
--  Methodology
--------------------------------------------------------------------------------
met={}

--------------------------------------------------------------------------------
--  Sparse coding: C=||Y-X.Beta||_F + lambda_B.RW_from_zero(X2) + lambda_C.L1(Beta2)
--    CA=||Y-X.Beta||_F
--      where X=concat_columns(X1,X2)
--      X1 is a user-defined dictionary; X1 is fixed
--      X2 is unknown dictionary
--    X.Beta = X1.Beta1+X2.Beta2
--      Beta1 are the first m1 rows of Beta
--      Beta2 are the last (m-m1) rows of Beta
--    CB=lambda_B.RW_from_zero(X2)
--    CC=lambda_C.L1(Beta2)
--    Y:size()=(d,n)
--    Beta:size()=(m,n)
--    X:size()=(d,m)
--    X1:size()=(d,m1)
--  Assumptions:
--    m>2+m1
--  Solution
--    /Use FISTA to solve Beta (both Beta1 and Beta2), given X: 
--      Minimize smooth CA + non-smooth CC
--      Check Amir Beck, 2009
--    /Solve X2, given Beta: minimize CA+CB
--      d(CA+CB)/dX => Sylvester equation => Linear solution
--------------------------------------------------------------------------------


--  To clone another sparse coder from the same data
--    Pass a table to opt_clone
--    opt_clone.X2_addcol: the number of random columns added to X2

function met:sparsecode_ts(Y_orig,sdata_ori,Y_test_orig,opt_clone)
  local sdata_ori=sdata_ori
  local sdata
  
  local m=sdata_ori.m
  local d,n,m1
  local t01_Yrmean
  
  n=Y_orig:size(2)
  d=Y_orig:size(1)
  
  ------------------------------------------------------------------------------
  --  Initialization
  ------------------------------------------------------------------------------
  
  if opt_clone then
    opt_clone.X2_addcol=opt_clone.X2_addcol or 0
    sdata={
      seed=sdata_ori.seed,
      useCuda=sdata_ori.useCuda,
      lambdaB=sdata_ori.lambdaB,
      lambdaC=sdata_ori.lambdaC,
      m=sdata_ori.m + opt_clone.X2_addcol ,
      Beta_columnwise_est=sdata_ori.Beta_columnwise_est,
      seed=sdata_ori.seed,
      X2_penaltyType=sdata_ori.X2_penaltyType,
      estX_L1RW_npL=sdata_ori.estX_L1RW_npL,
      
      m1=sdata_ori.X1:size(2),
      initial_tensor_type=sdata_ori.initial_tensor_type,
      status_Y="train",
      
      
      -- reference to sdata_ori
      Beta_Fista_par=sdata_ori.Beta_Fista_par, 
      estX_L1RW_Fista_par=sdata_ori.estX_L1RW_Fista_par,
      
      Y_orig=sdata_ori.Y_orig,
      Y_test_orig=sdata_ori.Y_test_orig,
      Y_train=sdata_ori.Y_train,
      Y_test=sdata_ori.Y_test,
      Y=sdata_ori.Y_train,
      
    }
    
    m1=sdata.m1
    if (opt_clone.X2_addcol>0) then
      m=m+opt_clone.X2_addcol
      
      sdata.X=to.cat(sdata_ori.X,to.randn(d,opt_clone.X2_addcol),2)
      sdata.X1=sdata_ori.X1:clone()
      sdata.X2=sdata.X[{{},{m1+1,m}}] -- reference
      
      sdata.Beta_train=to.cat(sdata_ori.Beta_train,to.zeros(opt_clone.X2_addcol,sdata_ori.Beta_train:size(2)),1)
      sdata.Beta=sdata.Beta_train -- reference
      if (sdata.Y_test~=nil) then
        sdata.Beta_test=to.cat(sdata_ori.Beta_test,to.zeros(opt_clone.X2_addcol,sdata_ori.Beta_test:size(2)),1)
      end
      if (m1>0) then
        sdata.Beta1=sdata.Beta[{{1,m1},{}}] -- reference
      else
        sdata.Beta1=nil
      end
      sdata.Beta2=sdata.Beta[{{m1+1,m},{}}] -- reference
      
      
    else
      
      -- 
      sdata.X=sdata_ori.X:clone()
      sdata.X1=sdata_ori.X1:clone()
      sdata.X2=sdata.X[{{},{m1+1,m}}] -- reference
      
      sdata.Beta_train=sdata_ori.Beta_train:clone()
      sdata.Beta=sdata.Beta_train -- reference
      if (sdata.Y_test~=nil) then
        sdata.Beta_test=sdata_ori.Beta_test:clone()
      end
      if (m1>0) then
        sdata.Beta1=sdata.Beta[{{1,m1},{}}] -- reference
      else
        sdata.Beta1=nil
      end
      sdata.Beta2=sdata.Beta[{{m1+1,m},{}}] -- reference
      
    end
    
  
  else
    sdata=sdata_ori
    sdata.initial_tensor_type=to.getdefaulttensortype()
    -- reference to original data
    sdata.Y_orig=Y_orig
    sdata.Y_test_orig=Y_test_orig
    
    
    if sdata.useCuda then
      cto.manualSeed(sdata.seed or 0,  cto.getDevice())
      to.setdefaulttensortype('torch.CudaTensor')
      sdata.Y=Y_orig:cuda()
      sdata.Y_train=sdata.Y -- reference
      if (sdata.Y_test_orig~=nil) then
        sdata.Y_test=sdata.Y_test_orig:cuda()
      end
    else
      to.manualSeed(sdata.seed or 0)
      to.setdefaulttensortype('torch.FloatTensor')
      sdata.Y=Y_orig:float()
      sdata.Y_train=sdata.Y -- reference
      if (sdata.Y_test_orig~=nil) then
        sdata.Y_test=sdata.Y_test_orig:float()
      end
    end
    sdata.status_Y="train"
    
    --  Initialize X, X1, X2 and Beta, Beta1 (can be nil), Beta2
    t01_Yrmean=sdata.Y:mean(2)
    if (sdata.X1 and sdata.X1.type and (string.match(sdata.X1:type(),"Tensor")) ) then
      if sdata.useCuda then
        sdata.X1=sdata.X1:cuda()
      else
        sdata.X1=sdata.X1:float()
      end
      sdata.m1=sdata.X1:size(2)
      
      --  sdata.X=to.cat({t01_Yrmean, to.ones(d,m-1), })
      sdata.X=to.cat({sdata.X1, t01_Yrmean, to.ones(d,1), to.randn(d,m-2-sdata.m1)})
      --  sdata.X=to.cat({t01_Yrmean, to.ones(d,1), to.randn(d,m-2)+t01_Yrmean:expand(d,m-2)})
    else
      sdata.X1=nil
      sdata.m1=0
      
      sdata.X=to.cat({t01_Yrmean, to.ones(d,1), to.randn(d,m-2-m1)})
    end
    m1=sdata.m1
    
    sdata.X2=sdata.X[{{},{m1+1,m}}] -- reference
    sdata.Beta=to.Tensor(m,n):zero()
    sdata.Beta_train=sdata.Beta -- reference
    if (sdata.Y_test~=nil) then
      sdata.Beta_test=to.Tensor(m,sdata.Y_test:size(2)):zero()
    end
    if (m1>0) then
      sdata.Beta1=sdata.Beta[{{1,m1},{}}] -- reference
    else
      sdata.Beta1=nil
    end
    sdata.Beta2=sdata.Beta[{{m1+1,m},{}}] -- reference
  end
  
  
  
  ------------------------------------------------------------------------------
  --  Upvalues: Shared external variables
  --    Shared local variables declared in met:sparsecode_ts (not functions within met:sparsecode_ts)
  --  Keep them minimal to avoid confusion
  ------------------------------------------------------------------------------
  sdata.upval_print=function(self)
    m1=m1 or "nil"
    print(string.format("d: %d| n: %d| m: %d| m1: %s",d,n,m,tostring(m1)))
    print(string.format("sdata: %s| self: %s",tostring(sdata), tostring(self)))
  end
  
  sdata.upval_update=function(self,d_new,n_new,m_new,m1_new)
    d=d_new; n=n_new; m=m_new; m1=m1_new
  end
  
  ------------------------------------------------------------------------------
  --  Evaluation of the current estimation
  ------------------------------------------------------------------------------
  sdata.SC_eval=function(self)
    local residual,CA, CB, CC, X2_nzcolcount, Beta2_sparseratio
    local m2=m-m1
    
    residual=to.Tensor(d,n)
    to.addmm(residual,1,self.Y,-1,self.X,self.Beta)
    CA=to.pow(residual,2):sum()
    if (self.X2_penaltyType=="L2-RW") then
      CB=to.mm(self.estX_L2RW_E,self.X2):cmul(self.X2):sum()  -- Note that E=lambdaB.D
    elseif (self.X2_penaltyType=="L1-RW") then
      CB=self.lambdaB*(self.X2[1]:norm(1) + (self.X2[{{2,d},}]-self.X2[{{1,d-1},}]):norm(1))
    else 
      CB=nil
    end
    
    CC=self.lambdaC*self.Beta2:norm(1)
    
    -- sparseness fraction of Beta2
    Beta2_sparseratio=to.abs(self.Beta2):gt(1e-6):sum()/self.Beta2:nElement()
    -- column sparseness fraction of X2
    X2_nzcolcount=to.gt(to.abs(self.X2):sum(1),1e-4):sum()
    
    return {residual=residual, 
      X2_penaltyType=self.X2_penaltyType,
      CA=CA,CA_by_n=CA/n,
      CB=CB,CB_by_X2nzCc=CB/X2_nzcolcount,
      CC=CC,
      C=CA+CB+CC,
      status_Y=self.status_Y,
      Beta2_sparseratio=Beta2_sparseratio,
      X2_nzcolcount=X2_nzcolcount}
  end
  
  ------------------------------------------------------------------------------
  --  FISTA: minimize smooth_f+nonsmooth_g wrt Beta
  --    Beta_f: evaluate f=CA(Beta) and df
  --    Beta_g: evaluate g=CC(Beta)=CC(Beta2)
  --  At one time, we only optimize a single observation
  --    CAi=t(ybi-X.betai).(ybi-X.betai)
  --    CCi=lambdaC.||beta2i||_1
  --      beta2i: parts belong to Beta2 of column betai
  --  beta_ind, betai, ybi and some other vars are saved as intermediate state vars
  --    Some temporary vars may be overwritten after any call of Betai_f Betai_g
  ------------------------------------------------------------------------------
  
  --
  sdata.Betai_f=function(betai_b)
    
    local ybi=sdata.ybi --
    local X=sdata.X
    local t01
    local dfval
    
    t01=to.addmv(-1,ybi,1,X,betai_b) -- (X.betai_b-ybi)
    fval=t01:norm()^2  -- f = t(ybi-X.betai_b).(ybi-X.betai_b)
    dfval=to.mv(X:t(),t01):mul(2) -- df/dbetai_b = 2.t(X).(X.betai_b-ybi)
    
    return fval, dfval
  end
  
  --
  sdata.Betai_g=function(betai_b)
    --  At the moment FISTALS only need gval
    local gval
    local beta2i_b = betai_b[{{m1+1,m}}]
    
    gval=sdata.lambdaC*beta2i_b:norm(1) -- g = lambdaC.||beta2i_b||_1
    return gval
  end
  
  
  --  new candidate of beta=pL(G(beta_prev,L),L)
  --    Note that this notation is different from the one in Amir Beck, 2009
  --    Check the source code fista.lua in package optim
  --  pL must modify gbetai in place
  --  G(beta_prev)=beta_prev - gradf(beta_prev)/L
  --  Define H(x)=g(x) + L/2*sum[ (x-G(beta_prev))^2 ]
  --  beta_new=pL(Gbeta=G(beta_prev,L),L)=argmin_x(H(x))
  --    This is easily solved if g(x) is separable
  --  In fact, if g(x)=lambdaC.||x||_1 (L1 penalty)
  --    then beta_new= (gbeta-alpha)_(+).sign(gbeta)
  --    with alpha=lambdaC/L=lambdaC*t
  --    t=1/L is the stepsize of SGD update Gbeta
  if (sdata.useCuda==true) then
    sdata.Betai_pL=function(gbetai,L)
      local alpha=sdata.lambdaC/L
      
      --  CudaTensor doesn't work with FFI
      --  local gbetaiD=gbetai:data()
      --  for i=m1,gbetai:nElement()-1 do
      --    if (gbetaiD[i]>alpha) then
      --      gbetaiD[i]=gbetaiD[i]-alpha
      --    elseif  (gbetaiD[i]< -alpha) then
      --      gbetaiD[i]=gbetaiD[i]+alpha
      --    else
      --      gbetaiD[i]=0
      --    end
      --  end
      
      gbeta2i=gbetai[{{m1+1,m}}]
      local t01=to.sign(gbeta2i)
      gbeta2i:abs():add(-alpha)
      gbeta2i:cmul(to.gt(gbeta2i,0)):cmul(t01)
      return gbetai
    end
  else
    sdata.Betai_pL=function(gbetai,L)
      local alpha=sdata.lambdaC/L
      
      gbeta2i=gbetai[{{m1+1,m}}]
      local t01=to.sign(gbeta2i)
      gbeta2i:abs():add(-alpha)
      gbeta2i:cmul(to.gt(gbeta2i,0):float()):cmul(t01)
      return gbetai
    end
  end
  
  sdata.betai=to.Tensor(m)
  sdata.ybi=to.Tensor(d)
  --  Update a single observation of betai and ybi 
  --    which is column i of matrix self.Beta and self.Y
  sdata.Betai_Fista_i=function(self,i)
    
    self.beta_ind=i
    self.betai:copy(self.Beta[{{},{i}}])
    self.ybi:copy(self.Y[{{},{i}}]):resize(d)
    local params={}
    --  clone FISTA Beta_Fista_par table as FISTA will modify this table param
    for k, v in pairs(self.Beta_Fista_par) do -- one-level cloning
      params[k]=v
    end 
    
    local result=optim.FistaLS(self.Betai_f, self.Betai_g, self.Betai_pL, self.betai, params)
    self.Beta[{{},{i}}]=result
    
    
    return result  --reference
  end
  
  --  Update the first k columns of self.Beta and self.Y
  sdata.Betai_Fista_all=function(self,k)
    k=k or n
    for i=1,k do
      self:Betai_Fista_i(i)
    end
    
  end
  
  ------------------------------------------------------------------------------
  --  This is a vectorized version of estimation of Beta
  --  Instead of optimizing one single optimization (CAi+ CCi, ybi, betai),
  --    this optimizes the whole (CA+CC, Y, Beta)
  --  However, compared to indpedent optimization (CAi+CCi),
  --    FISTA optim of CA+CC cannot utilize different stepsizes for different
  --    observations, resulting in slower convergence.
  ------------------------------------------------------------------------------
  --
  sdata.Beta_f=function(Beta_b)
    
    local Y=sdata.Y
    local X=sdata.X
    local t01, dfval
    
    t01=to.addmm(-1,Y,1,X,Beta_b) -- (X.Beta_b-y2)
    fval=t01:norm(2)^2 -- f = ||t01||_F^2 (Frobneous norm)
    dfval=to.mm(X:t(),t01):mul(2) -- df/dBeta_b = 2.t(X).(X.Beta_b-Y)
    
    return fval, dfval
  end
  
  --
  sdata.Beta_g=function(Beta_b)
    --  At the moment FISTALS only need gval
    local gval
    local Beta2_b = Beta_b[{{m1+1,m},{}}]
    
    gval=sdata.lambdaC*Beta2_b:norm(1) -- g = lambdaC.||Beta2_b||_1
    return gval
  end
  
  --
  if (sdata.useCuda==true) then
    sdata.Beta_pL=function(GBeta,L)
      --  not check yet
      local alpha=sdata.lambdaC/L
      
      GBeta2=GBeta[{{m1+1,m},{}}]
      local t01=to.sign(GBeta2)
      GBeta2:abs():add(-alpha)
      GBeta2:cmul(to.gt(GBeta2,0)):cmul(t01)
      return GBeta
    end
  else
    sdata.Beta_pL=function(GBeta,L)
      local alpha=sdata.lambdaC/L
      
      GBeta2=GBeta[{{m1+1,m},{}}]
      local t01=to.sign(GBeta2)
      GBeta2:abs():add(-alpha)
      GBeta2:cmul(to.gt(GBeta2,0):float()):cmul(t01)
      return GBeta
    end
  end
  
  --  Update self.Beta (vectorized version)
  sdata.Beta_Fista_all=function(self)
    local params={}
    
    --  clone FISTA Beta_Fista_par table as FISTA will modify this table param
    for k, v in pairs(self.Beta_Fista_par) do -- one-level cloning
      params[k]=v
    end 
    
    local result=optim.FistaLS_Tensor(self.Beta_f, self.Beta_g, self.Beta_pL,self.Beta, params)
    --  self.Beta will be modified in place
    
  end
  
  ------------------------------------------------------------------------------
  --  Given Beta, solve X with penalty L1-RW
  --  Minimize CA + CB wrt X
  --    CA=||Y-X.Beta||_F=||Y2-X2.Beta2||_F
  --      with Y2=Y-X1.Beta1
  --    CB=lambdaB.RW_from_zero(X2) 
  --      =lambdaB.sum_i[ sum_(j>1)[ (X2(j,i)-X2(j-1,i))^2 ] + X2(1,i)^2 ]
  --      =lambdaB.sum_i[ X2(.,i)^T.D.X2(.,i) ]  (X2(j,i): row j, column i)
  --    D=[ 2,-1, 0, 0,..]
  --      [-1, 2,-1, 0,..]
  --      [..,-1, 2,-1,..]
  --      [.., 0,-1, 2,-1]
  --      [.., 0, 0,-1, 1]
  --    size(D)=(d,d)
  --    dCA/dX2=-2(Y2-X2.Beta2).t(Beta2)
  --    dCB/dX2=2lambdaB.D.X2
  --    d(CA+CB)/dX2=0 <=> lambdaB.D.X2 + X2.Beta2.t(Beta2) = Y2.t(Beta2)
  --    <=> E.X2+X2.F=G
  --    <=> prod_K[I(m-m1,m-m1),E].colstack(X2) + prod_K(t(F),I(d,d)).colstack(X2) = colstrack(G)
  --      (prod_K: Kronecker product; colstack: stacking columns of X2)
  --    <=> (E2+F2).colstack(X2)=EF2.colstack(X2)=G2
  ------------------------------------------------------------------------------
  
  if (sdata.useCuda==true) then
    sdata.estX_L2RW_E=to.zeros(d,d)
    for i=1,d do
      sdata.estX_L2RW_E[i][i]=2*sdata.lambdaB
    end
  else
    sdata.estX_L2RW_E=to.eye(d,d)*2*sdata.lambdaB
  end
  sdata.estX_L2RW_E[d][d]=sdata.lambdaB
  for i=2,d do sdata.estX_L2RW_E[i-1][i]=-sdata.lambdaB; sdata.estX_L2RW_E[i][i-1]=-sdata.lambdaB end
  
  -- Update X (penalty version L2 with Random walk)
  sdata.estX_L2RW_solve = function(self) 
    local Beta2tBeta2, G2, EF2, val, ind1, ind2, Y2, m2
    
    m2=m-m1
    if (m1>0) then
      Y2=to.addmm(1,self.Y,-1,self.X1,self.Beta1)
    else
      Y2=self.Y
    end
    Beta2tBeta2=to.mm(self.Beta2,self.Beta2:t())
    G2=to.mm(self.Beta2,Y2:t()):resize(d*m2,1)
    
    
    EF2=to.zeros(d*m2,d*m2)
    --  F2 part
    for i=1,m2 do
      for j=i,m2 do
        val=Beta2tBeta2[i][j]
        ind1=(i-1)*d
        ind2=(j-1)*d
        for k=1,d do EF2[ind1+k][ind2+k]=val end
        if (j>=i) then
          for k=1,d do EF2[ind2+k][ind1+k]=val end
        end
      end
    end
    --  E2 part
    for i=1,m2 do EF2[{ {(i-1)*d+1,i*d},{(i-1)*d+1,i*d} }] = EF2[{ {(i-1)*d+1,i*d},{(i-1)*d+1,i*d} }] + self.estX_L2RW_E end
    -- torch.potrs(G2,to.potrf(EF2)):resize(m2,d):t()
    
    
    self.X2:copy(torch.potrs(G2,to.potrf(EF2)):resize(m2,d):t())
    return
  end
  
  ------------------------------------------------------------------------------
  --  Given Beta, solve X with penalty L1-RW
  --  Minimize CA + CB wrt X
  --    CA=||Y-X.Beta||_F=||Y2-X2.Beta2||_F
  --      with Y2=X1.Beta1
  --    CB=lambdaB.L1RW_from_zero(X2) 
  --      =lambdaB.sum_i[ sum_(j>1)[ |X2(j,i)-X2(j-1,i))| ] + |X2(1,i)| ]
  --    size(D)=(d,d)
  --    dCA/dX2=-2(Y2-X2.Beta2).t(Beta2)
  --  G(X2_prev)=X2_prev - gradf(X2_prev)/L
  --  Define H(X2)=g(X2) + L/2*sum[ (X2-G(X2_prev))^2]
  --  X2_new=pL(GX=G(X2_prev,L),L)=argmin_X2(H(X2))
  --  In this case g(X2)=CB(X2) is not separable for all elements
  --    but g(X2),H(X2) are column-separable
  --  Denote X2i as column i of matrix X2
  --    H(X2i)=lambdaB.L1RW_from_zero(X2i)+L/2*sum[ (X2i-GX2i)^2]
  --  Optimize H(X2i) wrt each element z of column X2i
  --  For the first (d-1) elements of X2i
  --    H(z) will have the form: alpha(|z-a|+|z-b|)+(z-gz)^2/2
  --    with alpha=lambdaB/L>0; b>a
  --    Solution pl(z)=argmin_z(H(z))
  --      gz>=b+2.alpha: pl(z)=gz-2alpha
  --      b+2.alpha>gz>=b: pl(z)=b
  --      b>gz>=a: pl(z)=gz
  --      a>gz>=a-2alpha: pl(z)=a
  --      a-2alpha>gz: pl(z)=gz+2aplha
  --      Denote this op by shrink(z,gz,2alpha,a,b)
  --  If z is the last element of X2i
  --    H(z) will have the form: alpha|z-a|+(z-gz)^2/2
  --    Solution: shrink gz to a by step=alpha
  --      Denote this op by shrink(z,gz,alpha,a)
  ------------------------------------------------------------------------------
  
  sdata.estX_L1RW_prep=function()
    local X1=sdata.X1
    local Beta1=sdata.Beta1
    
    if (m1>0) then
      sdata.estX_L1RW_X1Beta1=to.mm(X1,Beta1)
    end
  end
  
  --
  sdata.estX_L1RW_f=function(X2_b)
    local X1=sdata.X1
    local Y=sdata.Y
    local Beta=sdata.Beta
    local Beta1=sdata.Beta1
    local Beta2=sdata.Beta2
    local estX_L1RW_X1Beta1=sdata.estX_L1RW_X1Beta1 --should be updated already
    local Y2, t01, dfval
    
    --  t01=X2_b.Beta2-Y2=X2_b.Beta2+X1.Beta1-Y=X_b.Beta-Y
    if (m1>0) then
      t01=to.addmm(-1,Y,1,X2_b,Beta2):add(estX_L1RW_X1Beta1)
    else
      --  X2_b=X_b; Beta2=Beta
      t01=to.addmm(-1,Y,1,X2_b,Beta2)
    end
    
    fval=t01:norm(2)^2 -- f = ||t01||_F^2 (Frobneous norm)
    dfval=to.mm(t01,Beta2:t()):mul(2) -- df/dBeta_b = 2.t(X).(X.Beta_b-Y)
    
    return fval, dfval
  end
  
  --
  sdata.estX_L1RW_g=function(X2_b)
    --  At the moment FISTALS only need gval
    
    return sdata.lambdaB*(X2_b[1]:norm(1) + (X2_b[{{2,d},}]-X2_b[{{1,d-1},}]):norm(1))
  end
  
  --
  sdata.estX_L1RW_pL=function(GX2,L)
    local alpha=sdata.lambdaB/L
    local alpha2=2*alpha
    local m2=m-m1
    local npL=sdata.estX_L1RW_npL or 3
    local a,b, GX2i, gz, znew, c
    local GX2_ori=GX2:clone()
    local GX2_hash_prev, GX2_hash
    
    GX2_hash_prev=GX2:sum()
    for k=1,npL do
      for i=1,m2 do
        --  optimize each column i
        GX2i=GX2:select(2,i)
        GX2Oi=GX2_ori:select(2,i)
        for j=1,d do
          --  optimize each element
          if (j<d) then
            if (j==1) then
              if (GX2i[2]>0) then a=0; b=GX2i[2]
              else a=GX2i[2]; b=0 end
            else
              if (GX2i[j+1]>GX2i[j-1]) then a=GX2i[j-1]; b=GX2i[j+1]
              else a=GX2i[j+1]; b=GX2i[j-1] end
            end
            gz=GX2Oi[j]
            --  Do shrink(z,gz,2alpha,a,b)
            if (gz>b+alpha2) then znew=gz-alpha2
            elseif (gz>b) then znew=b
            elseif (gz>a) then znew=gz
            elseif (gz>a-alpha2) then znew=a
            else znew=gz+alpha2 end
            GX2i[j]=znew
          else
            a=GX2i[d-1]
            gz=GX2Oi[d]
            --  Do shrink(z,gz,alpha,a)
            if (gz>a+alpha) then znew=gz-alpha
            elseif (gz<a-alpha) then znew=gz+alpha
            else znew=a end
            GX2i[d]=znew
          end
        end
      end
      GX2_hash=GX2:sum()
      if (GX2_hash~=GX2_hash_prev) then
        GX2_hash_prev=GX2_hash
      else
        break
      end
    end
    
    return GX2
  end
  
  
  -- Update X (penalty version L1 with Random walk)
  sdata.estX_L1RW_Fista=function(self)
    local params={}
    
    --  clone FISTA estX_L1RW_Fista_par table as FISTA will modify this table param
    for k, v in pairs(self.estX_L1RW_Fista_par) do -- one-level cloning
      params[k]=v
    end
    
    -- Update some variables
    self:estX_L1RW_prep()
    
    local result=optim.FistaLS_Tensor(self.estX_L1RW_f, self.estX_L1RW_g, self.estX_L1RW_pL,self.X2, params)
    --  self.X2 will be modified in place
    
  end
  
  
  
  ------------------------------------------------------------------------------
  --  Shrinking zero columns of dictionary X2
  --  tol: a column of X2 is zero-column if sum(abs(col(X2)))<=tol
  ------------------------------------------------------------------------------
  sdata.shrink_X=function(self,tol)
    local X2_zcol_ind, X_new, i1, m1_new, ind01, Beta_test_new, Beta_train_new
    tol=tol or 1e-8
    
    --
    X_nzcol_ind=to.gt(to.abs(self.X):sum(1),tol):resize(m)
    if (m1>0) then
      X_nzcol_ind[{{1,m1}}]=1
    end
    
    --
    m_new=X_nzcol_ind:sum()
    --
    if (m_new==m) then
      return 0 -- return-code=0: there is no zero columns of X2; no need to shrink
    end
    
    ind01=to.nonzero(X_nzcol_ind):resize(m_new)
    X_new=to.Tensor(d,m_new)
    if (self.Beta_test~=nil) then
      Beta_test_new=to.Tensor(m_new,self.Beta_test:size(2))
    else
      Beta_test_new=nil
    end
    Beta_train_new=to.Tensor(m_new,self.Beta_train:size(2))
    
    --
    for i1=1,ind01:nElement() do
      X_new:select(2,i1):copy(self.X:select(2,ind01[i1]))
      if (Beta_test_new~=nil) then
        Beta_test_new:select(1,i1):copy(self.Beta_test:select(1,ind01[i1]))
      end
       Beta_train_new:select(1,i1):copy(self.Beta_train:select(1,ind01[i1]))
    end
    
    --
    self.X=X_new
    self.X2=self.X[{{},{m1+1,m_new}}] -- reference
    self.Beta_test=Beta_test_new
    self.Beta_train=Beta_train_new
    if (self.status_Y=="train") then
      self.Beta=self.Beta_train
    elseif (self.status_Y=="test") then
      self.Beta=self.Beta_test
    else
      self.Beta=nil
    end
    if (m1>0) then
      self.Beta1=self.Beta[{{1,m1},{}}] -- reference
    else
      self.Beta1=nil
    end
    self.Beta2=self.Beta[{{m1+1,m_new},{}}] -- reference
    self.betai=to.Tensor(m_new)
    self.m=m_new
    
    --
    self:upval_update(d,n,m_new,m1)
    
    return 10 -- return code=10: ok
  end
  
  
  ------------------------------------------------------------------------------
  --  Estimate Beta and X iteratively
  --  k: the number of iterations
  --  n_shrink: the interval length where X2 would be shrunk 
  --  verbose: print out evaluation measurement or not
  ------------------------------------------------------------------------------
  sdata.estBetaX_dokiter=function(self,k,n_shrink,verbose)
    local Beta_columnwise_est=self.Beta_columnwise_est or true
    local X2_penaltyType=self.X2_penaltyType or "L2-RW"
    for i=1, k do
      if (Beta_columnwise_est==true) then 
        self:Betai_Fista_all()
      else
        self:Beta_Fista_all()
      end
      if (verbose==true) then 
        print(string.format("----- Iteration: %d",i))
        print("----- After updating Beta")
        misc:strt(self:SC_eval()) 
      end
      
      if (X2_penaltyType=="L1-RW") then
        self:estX_L1RW_Fista()
      else 
        self:estX_L2RW_solve()
      end
      
      if (verbose==true) then 
        print("----- After updating X2")
        misc:strt(self:SC_eval()) 
      end
      
      if (math.fmod(i,n_shrink)==0) then
        self:shrink_X(1e-6*d)
        if (verbose==true) then 
          print("----- After shrinking X")
          misc:strt(self:SC_eval()) 
        end
      end
    end
  end
  
  
  
  
  ------------------------------------------------------------------------------
  --  Estimate Beta and X iteratively
  --  The iteative estimation is done for at least k1 iterations, and at most k2 iterations
  --  X2 would be shrunk once every n_shrink iterations
  --  The measurement evaluation is done once every n_SC iterations
  --  Denote std=math.sqrt(CA_by_n/d)
  --  Keep the estimation if the condition A is satisfied
  --  Condition A:
  --    last(X2_nzcolcount)-current(X2_nzcolcount)>=n_X2nzcc OR
  --    current std > upper bound std (std_ub)
  sdata.estBetaX_dowhileA=function(self,k1,k2,n_shrink,n_SC,n_X2nzcc,std_ub,verbose)
    local k2b, SC_tabc, dX2nzcc, metric_lX2nzcc, stdc
    local metric_lX2nzcc=nil
    
    -- like sdata.estBetaX_dokiter
    local Beta_columnwise_est=self.Beta_columnwise_est or true
    local X2_penaltyType=self.X2_penaltyType or "L2-RW"
    -- do at least k1 iterations
    for i=1, k1 do
      if (Beta_columnwise_est==true) then 
        self:Betai_Fista_all()
      else
        self:Beta_Fista_all()
      end
      if (verbose==true) then 
        print(string.format("----- Iteration: %d",i))
        print("----- After updating Beta")
        misc:strt(self:SC_eval()) 
      end
      
      if (X2_penaltyType=="L1-RW") then
        self:estX_L1RW_Fista()
      else 
        self:estX_L2RW_solve()
      end
      
      if (verbose==true) then 
        print("----- After updating X2")
        misc:strt(self:SC_eval()) 
      end
      
      -- shrink every n_shrink iterations
      if (math.fmod(i,n_shrink)==0) then
        self:shrink_X(1e-6*d)
        if (verbose==true) then 
          print("----- After shrinking X")
          misc:strt(self:SC_eval()) 
        end
      end
      
      -- evaluate every n_SC iterations
      if (math.fmod(i,n_SC)==0) then
        SC_tabc=self:SC_eval()
        -- keep that last number of non-zero columns of X2
        metric_lX2nzcc=SC_tabc.X2_nzcolcount
      end
    end
    
    -- do more iterations
    for i=k1+1, k2 do
      if (Beta_columnwise_est==true) then 
        self:Betai_Fista_all()
      else
        self:Beta_Fista_all()
      end
      if (verbose==true) then 
        print(string.format("----- Iteration: %d",i))
        print("----- After updating Beta")
        misc:strt(self:SC_eval()) 
      end
      
      if (X2_penaltyType=="L1-RW") then
        self:estX_L1RW_Fista()
      else 
        self:estX_L2RW_solve()
      end
      
      if (verbose==true) then 
        print("----- After updating X2")
        misc:strt(self:SC_eval()) 
      end
      
      if (math.fmod(i,n_shrink)==0) then
        self:shrink_X(1e-6*d)
        if (verbose==true) then 
          print("----- After shrinking X")
          misc:strt(self:SC_eval()) 
        end
      end
      
      -- evaluate every n_SC iterations
      if (math.fmod(i,n_SC)==0) then
        SC_tabc=self:SC_eval()
        -- keep that last number of non-zero columns of X2
        metric_lX2nzcc=metric_lX2nzcc or (SC_tabc.X2_nzcolcount+n_X2nzcc)
        
        -- evaluate stopping criteria
        dX2nzcc=metric_lX2nzcc-SC_tabc.X2_nzcolcount
        stdc=math.sqrt(SC_tabc.CA_by_n/d)
        conA=(stdc>std_ub) or (dX2nzcc>=n_X2nzcc)
        metric_lX2nzcc=SC_tabc.X2_nzcolcount
        
        if (verbose==true) then
          print("----- SC check point")
          print(string.format("  dX2nzcc: %d| stdc: %f",dX2nzcc,stdc))
        end
        
        if (conA==false) then
          return
        end
      end
    end
    
  end
  
  
  ------------------------------------------------------------------------------
  --  Switching between training dataset and testset
  --  typestr: if typestr=="train", use self.Y=self.Y_train
  --    if typestr=="test", use self.Y=self.Y_test
  --  Beta_est: if true, re-estimate self.Beta with the new self.Y
  --  Beta_est_k: how many iterations for the re-estimation of Beta
  ------------------------------------------------------------------------------
  sdata.switch_YBeta=function(self,typestr,Beta_est,Beta_est_k)
    local Beta_new, Y_new, n_new
    local Beta_columnwise_est=self.Beta_columnwise_est or true
    
    Beta_est=Beta_est or false
    Beta_est_k=Beta_est_k or 50
    
    if (typestr=="train") then
      Beta_new=self.Beta_train
      Y_new=self.Y_train
      self.status_Y="train"
    elseif (typestr=="test") then
      if (self.Y_test==nil) then
        print("ERROR: Ytest==nil")
        return
      end
      Beta_new=self.Beta_test
      Y_new=self.Y_test
      self.status_Y="test"
    end
    n_new=Y_new:size(2)
    
    --
    self.Y=Y_new
    self.Beta=Beta_new
    if (m1>0) then
      self.Beta1=self.Beta[{{1,m1},{}}] -- reference
    else
      self.Beta1=nil
    end
    self.Beta2=self.Beta[{{m1+1,m},{}}] -- reference
    
    self:upval_update(d,n_new,m,m1)
    
    -- re-estimation of Beta
    if (Beta_est==true) then
      for i=1,Beta_est_k do
        if (Beta_columnwise_est==true) then 
          self:Betai_Fista_all()
        else
          self:Beta_Fista_all()
        end
      end
    end
  end
  
  
  ------------------------------------------------------------------------------
  --
  ------------------------------------------------------------------------------
  sdata.estBeta_for_Ynew=function(self,Y_new,Beta_est_k)
    local Beta_new, n_new
    local Beta_columnwise_est=self.Beta_columnwise_est or true
    
    Beta_est_k=Beta_est_k or 50
    
    n_new=Y_new:size(2)
    Beta_new=to.Tensor(m,n_new):zero()
    
    --
    self.Y=Y_new
    self.Beta=Beta_new
    if (m1>0) then
      self.Beta1=self.Beta[{{1,m1},{}}] -- reference
    else
      self.Beta1=nil
    end
    self.Beta2=self.Beta[{{m1+1,m},{}}] -- reference
    
    self:upval_update(d,n_new,m,m1)
    
    -- re-estimation of Beta
    for i=1,Beta_est_k do
      if (Beta_columnwise_est==true) then 
        self:Betai_Fista_all()
      else
        self:Beta_Fista_all()
      end
    end
    
    -- 
    self:switch_YBeta(self.status_Y,false,0)
    
    --
    return Beta_new
  end
  
  
  ------------------------------------------------------------------------------
  --  estimate Beta and X with procedure B
  --  If flag_startup=true, perform estimation of Beta and X on self.Y_train, with
  --    self.lambdaB/lambdaC set to lambdaB_start/lambdaC_start 
  --    and by 
  --        estBetaX_dowhileA(k1_start,k2_start,n_shrink,n_SC,n_X2nzcc,std_ub,
  --                          verbose=true if verbose_cde>=10)
  --  [Loop_step] Try to increase self.lambdaB/lambdaC by lambdaB_by, lambdaC_by 
  --    (two independent instances of sparse coding)
  --  (Optional) Each new instance may add a number of randomly dictionary elements
  --    to X2. That number is defined by v
  --  Refit each new instance with 
  --        estBetaX_dowhileA(k1,k2,n_shrink,n_SC,n_X2nzcc,std_ub,
  --                          verbose=true if verbose_cde>=10)
  --  For new instances and the current instance, 
  --    fit Beta_test to Y_test and check CA_by_n of the test_set
  --  Select the instance of the smalllest CA_by_n 
  --    If the current instance changes, repeat from the [Loop_step]
  sdata.estBetaX_dowhileB=function(self,
      flag_startup, lambdaB_start, lambdaC_start, lambdaB_by, lambdaC_by, 
      X2_addcol, verbose_code,
      k1_start,k2_start, k1,k2,
      n_shrink,n_SC,n_X2nzcc,std_ub,
      k3_Beta_test)
    
    local sdata_cur, sdata_newlB, sdata_newlC, verbose2, verbose1
    
    
    if verbose_code>=10 then
      verbose2=true
      verbose1=true
    elseif verbose_code>=0 then
      verbose2=false
      verbose1=true
    else
      verbose2=false
      verbose1=false
    end
    sdata_cur=self
    
    
    
    if flag_startup==true then
      if (sdata_cur.status_Y=="test") then
        sdata_cur:switch_YBeta("train",false,0)
      end
      sdata_cur.lambdaB=lambdaB_start
      sdata_cur.lambdaC=lambdaC_start
      sdata_cur:estBetaX_dowhileA(k1_start,k2_start,n_shrink,n_SC,n_X2nzcc,std_ub,verbose2)
    end
    
    
    --
    sdata_cur:switch_YBeta("test",true,k3_Beta_test)
    tabSC_cur=sdata_cur:SC_eval()
    sdata_cur:switch_YBeta("train",false,k3_Beta_test)
    
    
    repeat 
      -- 
      sdata_newlB=met:sparsecode_ts(sdata_cur.Y_orig,sdata_cur,sdata_cur.Y_test_orig,{X2_addcol=X2_addcol});
      sdata_newlB.lambdaB=sdata_newlB.lambdaB+lambdaB_by
      
      sdata_newlC=met:sparsecode_ts(sdata_cur.Y_orig,sdata_cur,sdata_cur.Y_test_orig,{X2_addcol=X2_addcol});
      sdata_newlC.lambdaC=sdata_newlC.lambdaC+lambdaC_by
      
      --
      sdata_newlB:estBetaX_dowhileA(k1,k2,n_shrink,n_SC,n_X2nzcc,std_ub,verbose2)
      
      sdata_newlC:estBetaX_dowhileA(k1,k2,n_shrink,n_SC,n_X2nzcc,std_ub,verbose2)
      
      --
      sdata_newlB:switch_YBeta("test",true,k3_Beta_test)
      tabSC_newlB=sdata_newlB:SC_eval()
      sdata_newlB:switch_YBeta("train",false,k3_Beta_test)
      
      sdata_newlC:switch_YBeta("test",true,k3_Beta_test)
      tabSC_newlC=sdata_newlC:SC_eval()
      sdata_newlC:switch_YBeta("train",false,k3_Beta_test)
      
      if verbose1==true then
        print(string.format("----- lambdaB: %f | lambdaC: %f | test set",sdata_cur.lambdaB,sdata_cur.lambdaC))
        misc:strt(tabSC_cur)
        print(string.format("----- lambdaB: %f | lambdaC: %f | test set",sdata_newlB.lambdaB,sdata_newlB.lambdaC))
        misc:strt(tabSC_newlB)
        print(string.format("----- lambdaB: %f | lambdaC: %f | test set",sdata_newlC.lambdaB,sdata_newlC.lambdaC))
        misc:strt(tabSC_newlC)
      end
        
      if (tabSC_newlB.CA_by_n<tabSC_newlC.CA_by_n) and (tabSC_newlB.CA_by_n<tabSC_cur.CA_by_n) then 
        sdata_cur=sdata_newlB
        tabSC_cur=tabSC_newlB
        if verbose1==true then
          print("----- Adjust lambda B")
        end
      elseif (tabSC_newlC.CA_by_n<tabSC_newlB.CA_by_n) and (tabSC_newlC.CA_by_n<tabSC_cur.CA_by_n) then 
        sdata_cur=sdata_newlC
        tabSC_cur=tabSC_newlC
        if verbose1==true then
          print("----- Adjust lambda C")
        end
      else
        if verbose1==true then
          print("----- Cannot improve CA on test set. Quit")
        end
        return sdata_cur
      end
    
    until false
    
  end
  
  ------------------------------------------------------------------------------
  --  plot Y, X and Beta:
  --  Input: plot device IDs
  --    Use non-positive numbers to skip the corresponding plot
  --  If status_YB="new": plot Y_new, self.X, and Beta_new
  --    status_YB="train": plot self.Y_train, self.X, and self.Beta_train
  --    status_YB="test": plot self.Y_test, self.X, and self.Beta_test
  --  range_YB: a list of two indices that subsets plots of Y and Beta
  sdata.plotYXBeta=function(self,devY,devX,devBeta, 
      status_YB, range_YB, 
      Y_new,Beta_new)
    local tab_plotBeta, tab_plotX, X_ind, Beta_ind
    local temp_tstype=to.getdefaulttensortype()
    
    to.setdefaulttensortype('torch.FloatTensor')
    local devY=devY or 1
    local devX=devX or 2
    local devBeta=devBeta or 3
    local status_YB=status_YB or "train"
    
    X_plot=self.X
    if (status_YB=="new") then
      Y_plot=Y_new
      Beta_plot=Beta_new
    elseif (status_YB=="train") then
      Y_plot=self.Y_train
      Beta_plot=self.Beta_train
    elseif (status_YB=="test") then
      Y_plot=self.Y_test
      Beta_plot=self.Beta_test
    else
      print("Error code of status_YB")
      return
    end
    
    --
    if range_YB then
      range_YB[1]=math.max(range_YB[1],1)
      range_YB[2]=math.min(range_YB[2],Y_plot:size(2))
    else
      range_YB={1,Y_plot:size(2)}
    end
    
    print(string.format("YB: %s | devY: %d | devX: %d | devBeta: %d | range_min: %d | range_max: %d", 
        status_YB, devY, devX, devBeta,range_YB[1],range_YB[2]))
    
    if (Y_plot) and (devY>0) then
      tab_Y_plot={}
      Y_ind=to.range(1,Y_plot:size(1))
      for i=range_YB[1],range_YB[2] do
        tab_Y_plot[#tab_Y_plot+1]={tostring(i),Y_ind,Y_plot[{{},{i}}]:float():clone():resize(Y_ind:size(1)),'-'}
      end
      gp.figure(devY)
      gp.plot(tab_Y_plot)
    end
    
    if (X_plot) and (devX>0) then
      tab_X_plot={}
      X_ind=to.range(1,X_plot:size(1))
      for i=1,X_plot:size(2) do
        tab_X_plot[#tab_X_plot+1]={tostring(i),X_ind,X_plot[{{},{i}}]:float():clone():resize(X_ind:size(1)),'-'}
      end
      gp.figure(devX)
      gp.plot(tab_X_plot)
    end
    
    if (Beta_plot) and (devBeta>0) then
      tab_Beta_plot={}
      Beta_ind=to.range(1,Beta_plot:size(1))
      for i=range_YB[1],range_YB[2] do
        tab_Beta_plot[#tab_Beta_plot+1]={tostring(i),Beta_ind,Beta_plot[{{},{i}}]:float():clone():resize(Beta_ind:size(1)),'-'}
      end
      gp.figure(devBeta)
      gp.plot(tab_Beta_plot)
    end
    
    --
    to.setdefaulttensortype(temp_tstype)
  end
  
  return sdata
end


--------------------------------------------------------------------------------
--  project functions
--------------------------------------------------------------------------------
prj={}

--------------------------------------------------------------------------------
--  get entity names from a filepath
--------------------------------------------------------------------------------
function prj:get_metadata_entities(path)
  local entities={}
  for filename in lfs.dir(path) do
    if ((filename~=".") and (filename~="..")) then
      entities[#entities+1]=path .. "/" .. filename
    end
  end
  table.sort(entities)
  return entities
end

--------------------------------------------------------------------------------
--  get data of a single entity from a filepath
--  rename the keys (replace dot by underscore)
--  convert all negative values to (-1) (For this data, only nonnegative values are valid)
--  convert anomaly/season2 string to float (See the mapping table below)
--  all categorical vars are of type ByteTensor. Numerical vars are of type Tensor
--------------------------------------------------------------------------------
function prj:get_entity_data(datapath)
  
  local m=csvigo.load({path=datapath, mode="tidy",header=true});
  local mb={}
  local anostr2val={null=0, outlier=10, missing=11, redundant=12, dls_missing=21, dls_redundant=22}
  local season2val={winter=0, spring=1, summer=2, autumn=3}
  local tab_keys_catvar={"daytype", "specialday", "season"}
  local tab_keys_notensor={time= 1}
  
  for k,tab in pairs(m) do  --  for1
    local lckb=string.lower(string.gsub(k, "[.]", "_"))
    
    --
    if lckb=="anomaly_flag" then
      table.foreach(tab, function(k2,v2) tab[k2]=anostr2val[string.lower(v2)] end)
    elseif lckb=="season2" then
      table.foreach(tab, function(k2,v2) tab[k2]=season2val[string.lower(v2)] end)
    end
    
    --
    local flag_catvar=false
    for k2,v2 in pairs(tab_keys_catvar) do
      if string.match(lckb,v2)~=nil then
        flag_catvar=true
        break
      end
    end
    
    --
    if flag_catvar==true then
      local ten=to.ByteTensor(tab)
      mb[lckb]=ten
    elseif tab_keys_notensor[lckb]==nil then
      local ten=to.Tensor(tab)
      ten[to.lt(ten,0)]=-1
      mb[lckb]=ten
    else
      mb[lckb]=tab
    end
    
  end --  end for1
  
  return mb
end



--------------------------------------------------------------------------------
--  Take a table d01
--  Assume that d01 has series ts_signature with name=signname
--    and a coressponding (same size) series ts_target
--    and (optionally) a coressponding series of absolute time string ts_time with name=timename
--  This function cuts ts_target at points ts_signature=ind_at
--    then get two segments of ts_target at each cutpoint
--    ts_target[{{cutpoint+step_from1,cutpoint+step_to1}}]
--    ts_target[{{cutpoint+step_from2,cutpoint+step_to2}}]
--  The cutpoints, list of segments 1, list of segment2 are saved as  
--    tensor variables of d01 with names outnamecp, outname1, outname2
--  If test_period and train_period (list of two strings of absolute time) are available,
--    each cutpoint is marked if it belong to test_period or train_period.
--    The flag series are saved as tensors of d01 with names 
--      outnamecp_test, outnamecp_train
--  The absolute time format in ts_time, test_period, train_period
--    must be compatible and support comparison operators
--  return_new_table: 
--  if split_target_to_traintest==2, split series of names outname1/outname2 into train/test sets
--      and remove the merged series
--    elseif split_target_to_traintest==1, split series of names outname1/outname2 into train/test sets
--      and keep the merged series
--    else do not split
--------------------------------------------------------------------------------
function prj:get_targetsegments_on_signature(d01,
    signname, ind_at, step_from1, step_to1, step_from2, step_to2, 
    targetname, timename, 
    test_period, train_period,
    return_new_table, split_target_to_traintest, 
    outname1, outname2, outnamecp)
  
  --  default names
  outname1=outname1 or (targetname.."_prev")
  outname2=outname2 or (targetname.."_next")
  outnamecp=outnamecp or (targetname.."_cp")
  split_target_to_traintest=split_target_to_traintest or 0
  -- fixed names
  local outnamecp_train=(outnamecp.."_train")
  local outnamecp_test=(outnamecp.."_test")
  --
  local d02
  if return_new_table then
    d02={}
  else
    d02=d01
  end
  
  --
  local ts_signature=d01[signname]
  local ts_time=d01[timename]
  local ts_target=d01[targetname]
  
  
  local ts_cutpoints=to.nonzero(to.eq(ts_signature,ind_at))
  ts_cutpoints:resize((#ts_cutpoints)[1]);
  
  --  adjust cutpoints to satisfy the boundary condition
  ind01=((ts_cutpoints+math.max(step_from1,step_to1,step_from2,step_to2)):le(ts_target:size(1)) + 
    (ts_cutpoints+math.min(step_from1,step_to1,step_from2,step_to2)):ge(1)):ge(2)
  ts_cutpoints=ts_cutpoints[ind01]
  
  d02[outname1]=dt:get01_data_at_cps(ts_target,ts_cutpoints,step_from1,step_to1)
  d02[outname2]=dt:get01_data_at_cps(ts_target,ts_cutpoints,step_from2,step_to2)
  
  --  save the cutpoints
  d02[outnamecp]=ts_cutpoints
  
  if (test_period) and (train_period) then 
    --  mark cutpoints that belong to train dataset metadata.train_periods[d01.traintest_ind]
    --    test dataset metadata.test_periods[d01.traintest_ind]
    local function t01_getind(time1,time2)
      ind01=to.ByteTensor(ts_cutpoints:size(1)):zero()
      table.foreach(ts_cutpoints:totable(), function(k,v) 
        ind01[k]=((ts_time[v]>=time1) and  (ts_time[v]<=time2) and 1 or 0) 
        end)
      return ind01
    end
    
    d02[outnamecp_train]=t01_getind(train_period[1],train_period[2])
    d02[outnamecp_test]=t01_getind(test_period[1],test_period[2])
    
    if (split_target_to_traintest==1) or (split_target_to_traintest==2) then
        local outname1_train=(outname1.."_train")
        local outname1_test=(outname1.."_test")
        local outname2_train=(outname2.."_train")
        local outname2_test=(outname2.."_test")
        
        d02[outname1_train]=d02[outname1][d02[outnamecp_train]:resize(d02[outnamecp_train]:nElement(),1):expandAs(d02[outname1])]
        d02[outname1_train]:resize(d02[outnamecp_train]:sum(),d02[outname1]:size(2))
        
        d02[outname1_test]=d02[outname1][d02[outnamecp_test]:resize(d02[outnamecp_test]:nElement(),1):expandAs(d02[outname1])]
        d02[outname1_test]:resize(d02[outnamecp_test]:sum(),d02[outname1]:size(2))
        
        d02[outname2_train]=d02[outname2][d02[outnamecp_train]:resize(d02[outnamecp_train]:nElement(),1):expandAs(d02[outname2])]
        d02[outname2_train]:resize(d02[outnamecp_train]:sum(),d02[outname2]:size(2))
        
        d02[outname2_test]=d02[outname2][d02[outnamecp_test]:resize(d02[outnamecp_test]:nElement(),1):expandAs(d02[outname2])]
        d02[outname2_test]:resize(d02[outnamecp_test]:sum(),d02[outname2]:size(2))
        
        if split_target_to_traintest==2 then 
          d02[outname1]=nil
          d02[outname2]=nil
        end
    end
    
  end
  
  return d02
end

--------------------------------------------------------------------------------



--------------------------------------------------------------------------------
--  Data transform
--------------------------------------------------------------------------------
dt={}

--------------------------------------------------------------------------------
--  get data chunks of ts_input at cutpoints
--  cutpoints are indices of cutting points of ts_input
--  series of ts_input[{{cutpoint+step_from,cutpoint+step_to}}] are returned 
--    as rows of matrix result
--  cutpoints satisfying the boundary conditions will be flaged with ind01
--------------------------------------------------------------------------------
function dt:get01_data_at_cps(ts_input,ts_cutpoints,step_from,step_to)
  assert(step_to>=step_from);
  
  --  check boundary condition and filter
  local ind01=((ts_cutpoints+step_from):ge(1) + 
    (ts_cutpoints+step_to):le(ts_input:size(1))):ge(2)
  local ts_cutpoints=ts_cutpoints[ind01]
  
  local result=to.Tensor(ind01:size(1),step_to-step_from+1):zero()
  local t
  for i=1,ind01:nElement() do 
    if ind01[i]==1 then
      t=ts_cutpoints[i]
      result[{{i,}}]=ts_input[{{t+step_from,t+step_to}}]
    end
  end
  
  return result,ind01
end















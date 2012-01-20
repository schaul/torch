require 'torch'
require 'random'

local mytester 
local labtest = {}
local msize = 100

local function maxdiff(x,y)
   local d = x-y
   if x:type() == 'torch.DoubleTensor' or x:type() == 'torch.FloatTensor' then
      return d:abs():maxall()
   else
      local dd = torch.Tensor():resize(d:size()):copy(d)
      return dd:abs():maxall()
   end
end

function labtest.max()
   local x = torch.rand(msize,msize)
   local mx,ix = torch.max(x,1)
   local mxx = torch.Tensor()
   local ixx = torch.LongTensor()
   torch.max(mxx,ixx,x,1)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.max value')
   mytester:asserteq(maxdiff(ix,ixx),0,'torch.max index')
end
function labtest.min()
   local x = torch.rand(msize,msize)
   local mx,ix = torch.min(x)
   local mxx = torch.Tensor()
   local ixx = torch.LongTensor()
   torch.min(mxx,ixx,x)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.min value')
   mytester:asserteq(maxdiff(ix,ixx),0,'torch.min index')
end
function labtest.sum()
   local x = torch.rand(msize,msize)
   local mx = torch.sum(x)
   local mxx = torch.Tensor()
   torch.sum(mxx,x)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.sum value')
end
function labtest.prod()
   local x = torch.rand(msize,msize)
   local mx = torch.prod(x)
   local mxx = torch.Tensor()
   torch.prod(mxx,x)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.prod value')
end
function labtest.cumsum()
   local x = torch.rand(msize,msize)
   local mx = torch.cumsum(x)
   local mxx = torch.Tensor()
   torch.cumsum(mxx,x)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.cumsum value')
end
function labtest.cumprod()
   local x = torch.rand(msize,msize)
   local mx = torch.cumprod(x)
   local mxx = torch.Tensor()
   torch.cumprod(mxx,x)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.cumprod value')
end
function labtest.cross()
   local x = torch.rand(msize,3,msize)
   local y = torch.rand(msize,3,msize)
   local mx = torch.cross(x,y)
   local mxx = torch.Tensor()
   torch.cross(mxx,x,y)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.cross value')
end
function labtest.zeros()
   local mx = torch.zeros(msize,msize)
   local mxx = torch.Tensor()
   torch.zeros(mxx,msize,msize)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.zeros value')
end
function labtest.ones()
   local mx = torch.ones(msize,msize)
   local mxx = torch.Tensor()
   torch.ones(mxx,msize,msize)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.ones value')
end
function labtest.diag()
   local x = torch.rand(msize,msize)
   local mx = torch.diag(x)
   local mxx = torch.Tensor()
   torch.diag(mxx,x)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.diag value')
end
function labtest.eye()
   local mx = torch.eye(msize,msize)
   local mxx = torch.Tensor()
   torch.eye(mxx,msize,msize)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.eye value')
end
function labtest.range()
   local mx = torch.range(0,1)
   local mxx = torch.Tensor()
   torch.range(mxx,0,1)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.range value')
end
function labtest.randperm()
   local t=os.time()
   random.manualSeed(t)
   local mx = torch.randperm(msize)
   local mxx = torch.Tensor()
   random.manualSeed(t)
   torch.randperm(mxx,msize)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.randperm value')
end
function labtest.reshape()
   local x = torch.rand(10,13,23)
   local mx = torch.reshape(x,130,23)
   local mxx = torch.Tensor()
   torch.reshape(mxx,x,130,23)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.reshape value')
end
function labtest.sort()
   local x = torch.rand(msize,msize)
   local mx,ix = torch.sort(x)
   local mxx = torch.Tensor()
   local ixx = torch.LongTensor()
   torch.sort(mxx,ixx,x)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.sort value')
   mytester:asserteq(maxdiff(ix,ixx),0,'torch.sort index')
end
function labtest.tril()
   local x = torch.rand(msize,msize)
   local mx = torch.tril(x)
   local mxx = torch.Tensor()
   torch.tril(mxx,x)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.tril value')
end
function labtest.triu()
   local x = torch.rand(msize,msize)
   local mx = torch.triu(x)
   local mxx = torch.Tensor()
   torch.triu(mxx,x)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.tril value')
end
function labtest.cat()
   local x = torch.rand(13,msize,msize)
   local y = torch.rand(17,msize,msize)
   local mx = torch.cat(x,y,1)
   local mxx = torch.Tensor()
   torch.cat(mxx,x,y,1)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.cat value')
end
function labtest.sin()
   local x = torch.rand(msize,msize,msize)
   local mx = torch.sin(x)
   local mxx  = torch.Tensor()
   torch.sin(mxx,x)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.sin value')
end
function labtest.linspace()
   local from = math.random()
   local to = from+math.random()
   local mx = torch.linspace(from,to,137)
   local mxx = torch.Tensor()
   torch.linspace(mxx,from,to,137)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.linspace value')
end
function labtest.logspace()
   local from = math.random()
   local to = from+math.random()
   local mx = torch.logspace(from,to,137)
   local mxx = torch.Tensor()
   torch.logspace(mxx,from,to,137)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.logspace value')
end
function labtest.rand()
   random.manualSeed(123456)
   local mx = torch.rand(msize,msize)
   local mxx = torch.Tensor()
   random.manualSeed(123456)
   torch.rand(mxx,msize,msize)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.rand value')
end
function labtest.randn()
   random.manualSeed(123456)
   local mx = torch.randn(msize,msize)
   local mxx = torch.Tensor()
   random.manualSeed(123456)
   torch.randn(mxx,msize,msize)
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.randn value')
end
function labtest.gesv()
   if not torch.gesv then return end
   local a=torch.Tensor({{6.80, -2.11,  5.66,  5.97,  8.23},
			 {-6.05, -3.30,  5.36, -4.44,  1.08},
			 {-0.45,  2.58, -2.70,  0.27,  9.04},
			 {8.32,  2.71,  4.35, -7.17,  2.14},
			 {-9.67, -5.14, -7.26,  6.08, -6.87}}):t():clone()
   local b=torch.Tensor({{4.02,  6.19, -8.22, -7.57, -3.03},
			 {-1.56,  4.00, -8.67,  1.75,  2.86},
			 {9.81, -4.09, -4.57, -8.61,  8.99}}):t():clone()
   local mx = torch.gesv(b,a)
   local ta = torch.Tensor()
   local tb = torch.Tensor()
   local mxx = torch.gesv(tb,ta,b,a)
   local mxxx = torch.gesv(b,a,true)
   mytester:asserteq(maxdiff(mx,tb),0,'torch.gesv value temp')
   mytester:asserteq(maxdiff(mx,b),0,'torch.gesv value flag')
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.gesv value out1')
   mytester:asserteq(maxdiff(mx,mxxx),0,'torch.gesv value out2')
end
function labtest.gels()
   if not torch.gels then return end
   local a=torch.Tensor({{ 1.44, -9.96, -7.55,  8.34,  7.08, -5.45},
			 {-7.84, -0.28,  3.24,  8.09,  2.52, -5.70},
			 {-4.39, -3.24,  6.27,  5.28,  0.74, -1.19},
			 {4.53,  3.83, -6.64,  2.06, -2.47,  4.70}}):t():clone()
   local b=torch.Tensor({{8.58,  8.26,  8.48, -5.28,  5.72,  8.93},
			 {9.35, -4.43, -0.70, -0.26, -7.36, -2.52}}):t():clone()
   local mx = torch.gels(b,a)
   local ta = torch.Tensor()
   local tb = torch.Tensor()
   local mxx = torch.gels(tb,ta,b,a)
   local mxxx = torch.gels(b,a,true)
   mytester:asserteq(maxdiff(mx,tb),0,'torch.gels value temp')
   mytester:asserteq(maxdiff(mx,b),0,'torch.gels value flag')
   mytester:asserteq(maxdiff(mx,mxx),0,'torch.gels value out1')
   mytester:asserteq(maxdiff(mx,mxxx),0,'torch.gels value out2')
end
function labtest.eig()
   if not torch.eig then return end
   local a=torch.Tensor({{ 1.96,  0.00,  0.00,  0.00,  0.00},
			 {-6.49,  3.80,  0.00,  0.00,  0.00},
			 {-0.47, -6.39,  4.17,  0.00,  0.00},
			 {-7.20,  1.50, -1.51,  5.70,  0.00},
			 {-0.65, -6.34,  2.67,  1.80, -7.10}}):t():clone()
   local e = torch.eig(a)
   local ee,vv = torch.eig(a,'v')
   local te = torch.Tensor()
   local tv = torch.Tensor()
   local eee,vvv = torch.eig(te,tv,a,'v')
   mytester:assertlt(maxdiff(e,ee),1e-12,'torch.eig value')
   mytester:assertlt(maxdiff(ee,eee),1e-12,'torch.eig value')
   mytester:assertlt(maxdiff(ee,te),1e-12,'torch.eig value')
   mytester:assertlt(maxdiff(vv,vvv),1e-12,'torch.eig value')
   mytester:assertlt(maxdiff(vv,tv),1e-12,'torch.eig value')
end
function labtest.svd()
   if not torch.svd then return end
   local a=torch.Tensor({{8.79,  6.11, -9.15,  9.57, -3.49,  9.84},
			 {9.93,  6.91, -7.93,  1.64,  4.02,  0.15},
			 {9.83,  5.04,  4.86,  8.83,  9.80, -8.99},
			 {5.45, -0.27,  4.85,  0.74, 10.00, -6.02},
			 {3.16,  7.98,  3.01,  5.80,  4.27, -5.31}}):t():clone()
   local u,s,v = torch.svd(a)
   local uu = torch.Tensor()
   local ss = torch.Tensor()
   local vv = torch.Tensor()
   uuu,sss,vvv = torch.svd(uu,ss,vv,a)
   mytester:asserteq(maxdiff(u,uu),0,'torch.svd')
   mytester:asserteq(maxdiff(u,uuu),0,'torch.svd')
   mytester:asserteq(maxdiff(s,ss),0,'torch.svd')
   mytester:asserteq(maxdiff(s,sss),0,'torch.svd')
   mytester:asserteq(maxdiff(v,vv),0,'torch.svd')
   mytester:asserteq(maxdiff(v,vvv),0,'torch.svd')
end
function torch.test()
   math.randomseed(os.time())
   mytester = torch.Tester()
   mytester:add(labtest)
   mytester:run()
end

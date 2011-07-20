#include "luaT.h"
#include "THC.h"
#include "THLogAdd.h" /* DEBUG: WTF */

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

const void *torch_CudaTensor_id = NULL;

#include "HardTanh.cu"
#include "LogSoftMax.cu"
#include "TemporalLogSoftMax.cu"
#include "TemporalConvolution.cu"

DLL_EXPORT TH_API int luaopen_libcunn(lua_State *L)
{
  lua_newtable(L);

  torch_CudaTensor_id = luaT_checktypename2id(L, "torch.CudaTensor");

  cunn_HardTanh_init(L);
  cunn_LogSoftMax_init(L);
  cunn_TemporalLogSoftMax_init(L);
  cunn_TemporalConvolution_init(L);

  return 1;
}
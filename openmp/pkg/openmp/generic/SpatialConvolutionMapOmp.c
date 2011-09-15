#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolutionMapOmp.c"
#else

#include "omp.h"

static int nnOmp_(SpatialConvolutionMap_forwardOmp)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  setompnthread(L,1,"nThread");

  THTensor *connTable = luaT_getfieldcheckudata(L, 1, "connTable", torch_(Tensor_id));
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  luaL_argcheck(L, input->nDimension == 3, 2, "3D tensor expected");
  luaL_argcheck(L, input->size[0] == nInputPlane, 2, "invalid number of input planes");
  luaL_argcheck(L, input->size[2] >= kW && input->size[1] >= kH, 2, "input image smaller than kernel size");

  THTensor_(resize3d)(output, nOutputPlane,
                      (input->size[1] - kH) / dH + 1, 
                      (input->size[2] - kW) / dW + 1);

  // contiguous
  input = THTensor_(newContiguous)(input);
  output = THTensor_(newContiguous)(output);

  // get raw pointers
  real *input_data = THTensor_(data)(input);
  real *output_data = THTensor_(data)(output);
  real *weight_data = THTensor_(data)(weight);
  real *bias_data = THTensor_(data)(bias);

  // and dims
  long input_n = input->size[0];
  long input_h = input->size[1];
  long input_w = input->size[2];
  long output_n = output->size[0];
  long output_h = output->size[1];
  long output_w = output->size[2];
  long weight_n = weight->size[0];
  long weight_h = weight->size[1];
  long weight_w = weight->size[2];

  // add bias
  long p;
#pragma omp parallel for private(p)
  for (p = 0; p < nOutputPlane; p++) {
    real *ptr_output = output_data + p*output_w*output_h;
    long j;
    for(j = 0; j < output_h*output_w; j++)
      ptr_output[j] = bias_data[p];
  
    // convolve all maps
    int nweight = connTable->size[0];
    long k;
    for (k = 0; k < nweight; k++) {
      // get offsets for input/output
      int o = (int)THTensor_(get2d)(connTable,k,1)-1;
      int i = (int)THTensor_(get2d)(connTable,k,0)-1;
      
      if (o == p)
      {
	THLab_(validXCorr2Dptr)(output_data + o*output_w*output_h,
				1.0,
				input_data + i*input_w*input_h, input_h, input_w,
				weight_data + k*weight_w*weight_h, weight_h, weight_w,
				dH, dW);
      }
    }
  }

// clean up
  THTensor_(free)(input);
  THTensor_(free)(output);
  
  return 1;
}

static int nnOmp_(SpatialConvolutionMap_backwardOmp)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THTensor *connTable = luaT_getfieldcheckudata(L, 1, "connTable", torch_(Tensor_id));
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_(Tensor_id));
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  // contiguous
  gradInput = THTensor_(newContiguous)(gradInput);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  // Resize/Zero
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  // get raw pointers
  real *gradInput_data = THTensor_(data)(gradInput);
  real *gradOutput_data = THTensor_(data)(gradOutput);
  real *weight_data = THTensor_(data)(weight);
  real *gradWeight_data = THTensor_(data)(gradWeight);

  // and dims
  long input_n = input->size[0];
  long input_h = input->size[1];
  long input_w = input->size[2];
  long output_n = gradOutput->size[0];
  long output_h = gradOutput->size[1];
  long output_w = gradOutput->size[2];
  long weight_n = weight->size[0];
  long weight_h = weight->size[1];
  long weight_w = weight->size[2];


  long p;
#pragma omp parallel for private(p)
  for(p = 0; p < nInputPlane; p++)
  {
    long k;
    // backward all
    int nkernel = connTable->size[0];
    for(k = 0; k < nkernel; k++)
    {
      int o = (int)THTensor_(get2d)(connTable,k,1)-1;
      int i = (int)THTensor_(get2d)(connTable,k,0)-1;
      if (i == p)
      {
	
	// gradient to input
	THLab_(fullConv2Dptr)(gradInput_data + i*input_w*input_h,
			      1.0,
			      gradOutput_data + o*output_w*output_h,  output_h,  output_w,
			      weight_data + k*weight_w*weight_h, weight_h, weight_w,
			      dH, dW);
      }
    }
  }
  
  // clean up
  THTensor_(free)(gradInput);
  THTensor_(free)(gradOutput);
    
  return 1;
}

static int nnOmp_(SpatialConvolutionMap_accGradParametersOmp)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  real scale = luaL_optnumber(L, 4, 1);

  THTensor *connTable = luaT_getfieldcheckudata(L, 1, "connTable", torch_(Tensor_id));
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_(Tensor_id));
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  // contiguous
  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  // get raw pointers
  real *input_data = THTensor_(data)(input);
  real *gradOutput_data = THTensor_(data)(gradOutput);
  real *weight_data = THTensor_(data)(weight);
  real *gradWeight_data = THTensor_(data)(gradWeight);
  real *gradBias_data = THTensor_(data)(gradBias);

  // and dims
  long input_n = input->size[0];
  long input_h = input->size[1];
  long input_w = input->size[2];
  long output_n = gradOutput->size[0];
  long output_h = gradOutput->size[1];
  long output_w = gradOutput->size[2];
  long weight_n = weight->size[0];
  long weight_h = weight->size[1];
  long weight_w = weight->size[2];

  // gradients wrt bias
  long k;
#pragma omp parallel for private(k)
  for(k = 0; k < nOutputPlane; k++) {
    real *ptr_gradOutput = gradOutput_data + k*output_w*output_h;
    long l;
    for(l = 0; l < output_h*output_w; l++)
      gradBias_data[k] += scale*ptr_gradOutput[l];
  }

  // gradients wrt weight
  int nkernel = connTable->size[0];
#pragma omp parallel for private(k)
  for(k = 0; k < nkernel; k++)
  {
    int o = (int)THTensor_(get2d)(connTable,k,1)-1;
    int i = (int)THTensor_(get2d)(connTable,k,0)-1;

    // gradient to kernel
    THLab_(validXCorr2DRevptr)(gradWeight_data + k*weight_w*weight_h,
                               scale,
                               input_data + i*input_w*input_h, input_h, input_w,
                               gradOutput_data + o*output_w*output_h, output_h, output_w,
                               dH, dW);
  }

  // clean up
  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
  return 0;
}

static const struct luaL_Reg nnOmp_(SpatialConvolutionMapStuff__) [] = {
  {"SpatialConvolutionMap_forwardOmp", nnOmp_(SpatialConvolutionMap_forwardOmp)},
  {"SpatialConvolutionMap_backwardOmp", nnOmp_(SpatialConvolutionMap_backwardOmp)},
  {"SpatialConvolutionMap_accGradParametersOmp", nnOmp_(SpatialConvolutionMap_accGradParametersOmp)},
  {NULL, NULL}
};

static void nnOmp_(SpatialConvolutionMap_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  lua_getfield(L,-1,"nn");
  luaL_register(L, NULL, nnOmp_(SpatialConvolutionMapStuff__));
  lua_pop(L,1);
}

#endif

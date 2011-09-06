#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolutionMap.c"
#else

static int nn_(SpatialConvolutionMap_forward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

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
  THTensor *outputPlane = THTensor_(new)();
  int k;
  for (k = 0; k < nOutputPlane; k++) {
    THTensor_(select)(outputPlane,output,0,k);
    THTensor_(fill)(outputPlane, THTensor_(get1d)(bias, k));
  }
  THTensor_(free)(outputPlane);

  // convolve all maps
  int i,o;
  int nweight = connTable->size[0];
  for (k = 0; k < nweight; k++) {
    // get offsets for input/output
    o = (int)THTensor_(get2d)(connTable,k,1)-1;
    i = (int)THTensor_(get2d)(connTable,k,0)-1;

    // convolve each map
    THLab_(validXCorr2Dptr)(output_data + o*output_w*output_h,
                            1.0,
                            input_data + i*input_w*input_h, input_h, input_w,
                            weight_data + k*weight_w*weight_h, weight_h, weight_w,
                            dH, dW);
  }

  // clean up
  THTensor_(free)(input);
  THTensor_(free)(output);

  return 1;
}

static int nn_(SpatialConvolutionMap_backward)(lua_State *L)
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
  input = THTensor_(newContiguous)(input);
  gradInput = THTensor_(newContiguous)(gradInput);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  // Resize/Zero
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  // get raw pointers
  real *input_data = THTensor_(data)(input);
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

  // gradients wrt bias
  int k;
  THTensor *gradOutputPlane = THTensor_(new)();
  real *gradBias_data = THTensor_(data)(gradBias);
  for(k = 0; k < nOutputPlane; k++) {
    THTensor_(select)(gradOutputPlane, gradOutput, 0, k);
    gradBias_data[k] += THTensor_(sum)(gradOutputPlane);
  }
  THTensor_(free)(gradOutputPlane);

  // backward all
  int nkernel = connTable->size[0];
  for(k = 0; k < nkernel; k++)
  {
    int o = (int)THTensor_(get2d)(connTable,k,1)-1;
    int i = (int)THTensor_(get2d)(connTable,k,0)-1;

    // gradient to kernel
    THLab_(validXCorr2DRevptr)(gradWeight_data + k*weight_w*weight_h,
                               1.0,
                               input_data + i*input_w*input_h, input_h, input_w,
                               gradOutput_data + o*output_w*output_h, output_h, output_w,
                               dH, dW);
    
    // gradient to input
    THLab_(fullConv2Dptr)(gradInput_data + i*input_w*input_h,
                          1.0,
                          gradOutput_data + o*output_w*output_h,  output_h,  output_w,
                          weight_data + k*weight_w*weight_h, weight_h, weight_w,
                          dH, dW);
  }

  // clean up
  THTensor_(free)(input);
  THTensor_(free)(gradInput);
  THTensor_(free)(gradOutput);
  
  return 1;
}

static const struct luaL_Reg nn_(SpatialConvolutionMap__) [] = {
  {"SpatialConvolutionMap_forward", nn_(SpatialConvolutionMap_forward)},
  {"SpatialConvolutionMap_backward", nn_(SpatialConvolutionMap_backward)},
  {NULL, NULL}
};

static void nn_(SpatialConvolutionMap_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(SpatialConvolutionMap__), "nn");
  lua_pop(L,1);
}

#endif

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TemporalConvolution.c"
#else

static int nn_(TemporalConvolution_forward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int inputFrameSize = luaT_getfieldcheckint(L, 1, "inputFrameSize");
  int outputFrameSize = luaT_getfieldcheckint(L, 1, "outputFrameSize");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  THTensor *outputWindow, *inputWindow;
  int nInputFrame, nOutputFrame;
  long k;
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D tensor expected");
  luaL_argcheck(L, input->size[1] == inputFrameSize, 2, "invalid input frame size");
  luaL_argcheck(L, input->size[0] >= kW, 2, "input sequence smaller than kernel size");

  input = THTensor_(newContiguous)(input);
  outputWindow = THTensor_(new)();
  inputWindow = THTensor_(new)();

  nInputFrame = input->size[0];
  nOutputFrame = (nInputFrame - kW) / dW + 1;

  THTensor_(resize2d)(output,
                      nOutputFrame,
                      outputFrameSize);

  /* bias first */
  for(k = 0; k < nOutputFrame; k++)
  {
    THTensor_(select)(outputWindow, output, 0, k);
    THTensor_(copy)(outputWindow, bias);
  }

  /* ouch */
  for(k = 0; nOutputFrame > 0; k++)
  {
    long outputFrameStride = (kW-1)/dW+1;
    long inputFrameStride = outputFrameStride*dW;
    long nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
    nOutputFrame -= nFrame;

    THTensor_(setStorage2d)(inputWindow, input->storage,
                            input->storageOffset+k*dW*input->size[1],
                            nFrame, inputFrameStride*input->size[1],
                            kW*input->size[1], 1);

    THTensor_(setStorage2d)(outputWindow, output->storage, 
                            output->storageOffset + k*output->size[1],
                            nFrame, outputFrameStride*output->size[1],
                            output->size[1], 1);

    THTensor_(transpose)(weight, NULL, 0, 1);
    THTensor_(addmm)(outputWindow, 1, 1, inputWindow, weight);
    THTensor_(transpose)(weight, NULL, 0, 1);
  }

  THTensor_(free)(outputWindow);
  THTensor_(free)(inputWindow);
  THTensor_(free)(input);

  return 1;
}

static int nn_(TemporalConvolution_backward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  long nInputFrame = input->size[0];
  long nOutputFrame = gradOutput->size[0];

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  THTensor *gradOutputWindow;
  THTensor *gradInputWindow;
  long k;

  gradOutputWindow = THTensor_(new)();
  gradInputWindow = THTensor_(new)();

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  /* ouch */
  for(k = 0; nOutputFrame > 0; k++)
  {
    long outputFrameStride = (kW-1)/dW+1;
    long inputFrameStride = outputFrameStride*dW;
    long nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
    nOutputFrame -= nFrame;

    THTensor_(setStorage2d)(gradOutputWindow, gradOutput->storage,
                            gradOutput->storageOffset + k*gradOutput->size[1],
                            nFrame, outputFrameStride*gradOutput->size[1],
                            gradOutput->size[1], 1);

    THTensor_(setStorage2d)(gradInputWindow, gradInput->storage,
                            gradInput->storageOffset+k*dW*gradInput->size[1],
                            nFrame, inputFrameStride*gradInput->size[1],
                            kW*gradInput->size[1], 1);

    THTensor_(addmm)(gradInputWindow, 1, 1, gradOutputWindow, weight);
  }

  THTensor_(free)(gradOutputWindow);
  THTensor_(free)(gradInputWindow);

  return 1;
}

static int nn_(TemporalConvolution_accGradParameters)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));  
  real scale = luaL_optnumber(L, 4, 1);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  long nInputFrame = input->size[0];
  long nOutputFrame = gradOutput->size[0];

  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_(Tensor_id));
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_(Tensor_id));

  THTensor *gradOutputWindow;
  THTensor *inputWindow;
  long k;

  input = THTensor_(newContiguous)(input);
  gradOutputWindow = THTensor_(new)();
  inputWindow = THTensor_(new)();

  /* bias first */
  for(k = 0; k < nOutputFrame; k++)
  {
    THTensor_(select)(gradOutputWindow, gradOutput, 0, k);
    THTensor_(cadd)(gradBias, scale, gradOutputWindow);
  }

  /* ouch */
  for(k = 0; nOutputFrame > 0; k++)
  {
    long outputFrameStride = (kW-1)/dW+1;
    long inputFrameStride = outputFrameStride*dW;
    long nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
    nOutputFrame -= nFrame;

    THTensor_(setStorage2d)(inputWindow, input->storage,
                            input->storageOffset+k*dW*input->size[1],
                            nFrame, inputFrameStride*input->size[1],
                            kW*input->size[1], 1);

    THTensor_(setStorage2d)(gradOutputWindow, gradOutput->storage, 
                            gradOutput->storageOffset + k*gradOutput->size[1],
                            nFrame, outputFrameStride*gradOutput->size[1],
                            gradOutput->size[1], 1);

    THTensor_(transpose)(gradOutputWindow, NULL, 0, 1);
    THTensor_(addmm)(gradWeight, 1, scale, gradOutputWindow, inputWindow);
    THTensor_(transpose)(gradOutputWindow, NULL, 0, 1);
  }

  THTensor_(free)(gradOutputWindow);
  THTensor_(free)(inputWindow);
  THTensor_(free)(input);

  return 0;
}

static const struct luaL_Reg nn_(TemporalConvolution__) [] = {
  {"TemporalConvolution_forward", nn_(TemporalConvolution_forward)},
  {"TemporalConvolution_backward", nn_(TemporalConvolution_backward)},
  {"TemporalConvolution_accGradParameters", nn_(TemporalConvolution_accGradParameters)},
  {NULL, NULL}
};

static void nn_(TemporalConvolution_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(TemporalConvolution__), "nn");
  lua_pop(L,1);
}

#endif

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TemporalSubSampling.c"
#else

static int nn_(TemporalSubSampling_forward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int inputFrameSize = luaT_getfieldcheckint(L, 1, "inputFrameSize");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  THTensor *outputFrame, *inputWindow;
  int nInputFrame, nOutputFrame;
  long k;
  
  luaL_argcheck(L, input->nDimension == 2, 2, "2D tensor expected");
  luaL_argcheck(L, input->size[1] == inputFrameSize, 2, "invalid input frame size");
  luaL_argcheck(L, input->size[0] >= kW, 2, "input sequence smaller than kernel size");

  outputFrame = THTensor_(new)();
  inputWindow = THTensor_(new)();

  nInputFrame = input->size[0];
  nOutputFrame = (nInputFrame - kW) / dW + 1;

  THTensor_(resize2d)(output,
                      nOutputFrame,
                      inputFrameSize);
  
  for(k = 0; k < nOutputFrame; k++)
  {
    THTensor_(narrow)(inputWindow, input, 0, k*dW, kW);
    THTensor_(select)(outputFrame, output, 0, k);
    THLab_(sum)(outputFrame, inputWindow, 0);
    THTensor_(cmul)(outputFrame, weight);
    THTensor_(cadd)(outputFrame, 1, bias);
  }

  THTensor_(free)(outputFrame);
  THTensor_(free)(inputWindow);

  return 1;
}

static int nn_(TemporalSubSampling_backward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));  
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  THTensor *gradOutputFrame;
  THTensor *gradInputWindow, *buffer, *kwunit;
  long k;

  gradOutputFrame = THTensor_(new)();
  gradInputWindow = THTensor_(new)();
  buffer = THTensor_(new)();
  kwunit = THTensor_(newWithSize1d)(kW);

  THTensor_(fill)(kwunit, 1);
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  for(k = 0; k < gradOutput->size[0]; k++)
  {
    THTensor_(narrow)(gradInputWindow, gradInput, 0, k*dW, kW);
    THTensor_(select)(gradOutputFrame, gradOutput, 0, k);
    THLab_(cmul)(buffer, weight, gradOutputFrame);
    THTensor_(addr)(gradInputWindow, 1, kwunit, buffer);
  }

  THTensor_(free)(gradOutputFrame);
  THTensor_(free)(gradInputWindow);
  THTensor_(free)(buffer);
  THTensor_(free)(kwunit);

  return 1;
}

static int nn_(TemporalSubSampling_accGradParameters)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));  
  real scale = luaL_optnumber(L, 4, 1);

  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int dW = luaT_getfieldcheckint(L, 1, "dW");

  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_(Tensor_id));
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_(Tensor_id));

  THTensor *gradOutputFrame;
  THTensor *inputWindow, *buffer;
  long k;


  gradOutputFrame = THTensor_(new)();
  inputWindow = THTensor_(new)();
  buffer = THTensor_(new)();

  for(k = 0; k < gradOutput->size[0]; k++)
  {
    THTensor_(narrow)(inputWindow, input, 0, k*dW, kW);
    THTensor_(select)(gradOutputFrame, gradOutput, 0, k);
    THLab_(sum)(buffer, inputWindow, 0);
    THTensor_(addcmul)(gradWeight, scale, buffer, gradOutputFrame);
    THTensor_(cadd)(gradBias, scale, gradOutputFrame);
  }

  THTensor_(free)(gradOutputFrame);
  THTensor_(free)(inputWindow);
  THTensor_(free)(buffer);

  return 0;
}

static const struct luaL_Reg nn_(TemporalSubSampling__) [] = {
  {"TemporalSubSampling_forward", nn_(TemporalSubSampling_forward)},
  {"TemporalSubSampling_backward", nn_(TemporalSubSampling_backward)},
  {"TemporalSubSampling_accGradParameters", nn_(TemporalSubSampling_accGradParameters)},
  {NULL, NULL}
};

static void nn_(TemporalSubSampling_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(TemporalSubSampling__), "nn");
  lua_pop(L,1);
}

#endif

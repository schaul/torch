#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Threshold.c"
#else

static int nn_(Threshold_forward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  real val = luaT_getfieldchecknumber(L, 1, "val");
  real threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));
  
  THTensor_(resizeAs)(output, input);
  TH_TENSOR_APPLY2(real, output, real, input, \
                  *output_data = (*input_data > threshold) ? *input_data : val;);

  return 1;
}

static int nn_(Threshold_backward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  real threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,      \
                   if ((*input_data) > threshold) *gradInput_data = 1;  \
                   else *gradInput_data = 0;                            \
                   *gradInput_data = (*gradOutput_data) * (*gradInput_data););
  return 1;
}

static const struct luaL_Reg nn_(Threshold__) [] = {
  {"Threshold_forward", nn_(Threshold_forward)},
  {"Threshold_backward", nn_(Threshold_backward)},
  {NULL, NULL}
};

static void nn_(Threshold_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(Threshold__), "nn");
  lua_pop(L,1);
}

#endif

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/HardShrink.c"
#else

static int nn_(HardShrink_forward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  THTensor_(resizeAs)(output, input);
  
  TH_TENSOR_APPLY2(real, output, real, input,                       \
                   if ((*input_data) > 0.5) *output_data = *input_data - 0.5;    \
                   else if ((*input_data) < 0.5) *output_data = *input_data + 0.5; \
                   else *output_data = 0;);
  return 1;
}

static int nn_(HardShrink_backward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input, \
                   if ((*input_data) > 0.5) *gradInput_data = 1;              \
                   else if ((*input_data) < 0.5) *gradInput_data = 1;         \
                   else *gradInput_data = 0;                               \
                   *gradInput_data = (*gradOutput_data) * (*gradInput_data););
  return 1;
}

static const struct luaL_Reg nn_(HardShrink__) [] = {
  {"HardShrink_forward", nn_(HardShrink_forward)},
  {"HardShrink_backward", nn_(HardShrink_backward)},
  {NULL, NULL}
};

static void nn_(HardShrink_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(HardShrink__), "nn");
  lua_pop(L,1);
}

#endif

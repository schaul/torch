#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Abs.c"
#else

static int nn_(Abs_forward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  THTensor_(resizeAs)(output, input);

  TH_TENSOR_APPLY2(real, output, real, input, \
                   *output_data = fabs(*input_data);)
  return 1;
}

static int nn_(Abs_backward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input, \
                   real z = *input_data;                              \
                   *gradInput_data = *gradOutput_data * (z >= 0 ? 1 : -1);)
  return 1;
}

static const struct luaL_Reg nn_(Abs__) [] = {
  {"Abs_forward", nn_(Abs_forward)},
  {"Abs_backward", nn_(Abs_backward)},
  {NULL, NULL}
};

static void nn_(Abs_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(Abs__), "nn");
  lua_pop(L,1);
}

#endif

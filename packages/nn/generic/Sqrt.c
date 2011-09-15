#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Sqrt.c"
#else

static int nn_(Sqrt_forward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  THTensor_(resizeAs)(output, input);

  TH_TENSOR_APPLY2(real, output, real, input,		\
		   *output_data = sqrt(*input_data););

  return 1;
}

static int nn_(Sqrt_backward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  THTensor_(resizeAs)(gradInput, input);

  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,	\
		   *gradInput_data  = 0.5 * (*gradOutput_data / *output_data););
  
  return 1;
}

static const struct luaL_Reg nn_(Sqrt__) [] = {
  {"Sqrt_forward", nn_(Sqrt_forward)},
  {"Sqrt_backward", nn_(Sqrt_backward)},
  {NULL, NULL}
};

static void nn_(Sqrt_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(Sqrt__), "nn");
  lua_pop(L,1);
}

#endif

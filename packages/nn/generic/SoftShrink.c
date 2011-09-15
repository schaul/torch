#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SoftShrink.c"
#else

static int nn_(SoftShrink_forward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  real lambda = luaT_getfieldchecknumber(L, 1, "lambda");
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  THTensor_(resizeAs)(output, input);
  
  TH_TENSOR_APPLY2(real, output, real, input,				\
                   if ((*input_data) > lambda) *output_data = *input_data - lambda; \
                   else if ((*input_data) < lambda) *output_data = *input_data + lambda; \
                   else *output_data = 0;);
  return 1;
}

static int nn_(SoftShrink_backward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  real lambda = luaT_getfieldchecknumber(L, 1, "lambda");
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,	\
                   if ((*input_data) > lambda || (*input_data) < lambda) \
		     *gradInput_data = (*gradOutput_data);		\
		   else							\
		     *gradInput_data = 0;				\
    );
  return 1;
}

static const struct luaL_Reg nn_(SoftShrink__) [] = {
  {"SoftShrink_forward", nn_(SoftShrink_forward)},
  {"SoftShrink_backward", nn_(SoftShrink_backward)},
  {NULL, NULL}
};

static void nn_(SoftShrink_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(SoftShrink__), "nn");
  lua_pop(L,1);
}

#endif

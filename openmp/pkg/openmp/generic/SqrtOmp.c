#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SqrtOmp.c"
#else

static int nnOmp_(Sqrt_forwardOmp)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  setompnthread(L,1,"nThread");
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  THTensor_(resizeAs)(output, input);
  
  if (input->nDimension == 1 || !THTensor_(isContiguous)(input) || !THTensor_(isContiguous)(output))
  {
    TH_TENSOR_APPLY2(real, output, real, input,		\
		     *output_data = sqrt(*input_data););
  }
  else
  {
    real* output_data = THTensor_(data)(output);
    real* input_data  = THTensor_(data)(input);
    long k;

#pragma omp parallel for private(k)
    for (k = 0; k < input->size[0]; k++)
    {
      real* ptr_output = output_data + k*input->stride[0];
      real* ptr_input  = input_data  + k*input->stride[0];
      long i;
      for (i = 0; i < input->stride[0]; i++)
      {
	ptr_output[i] = sqrt(ptr_input[i]);
      }
    }
  }
  return 1;
}

static int nnOmp_(Sqrt_backwardOmp)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  setompnthread(L,1,"nThread");
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  THTensor_(resizeAs)(gradInput, input);

  if (output->nDimension == 1 || 
      !THTensor_(isContiguous)(output) || 
      !THTensor_(isContiguous)(gradOutput) ||
      !THTensor_(isContiguous)(gradInput))
  {
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,	\
		     *gradInput_data  = 0.5 * (*gradOutput_data / *output_data););
  }
  else
  {
    real* gradOutput_data = THTensor_(data)(gradOutput);
    real* gradInput_data  = THTensor_(data)(gradInput);
    real* output_data     = THTensor_(data)(output);
    long k;

#pragma omp parallel for private(k)
    for (k = 0; k < output->size[0]; k++)
    {
      real* ptr_gradOutput = gradOutput_data + k*output->stride[0];
      real* ptr_gradInput  = gradInput_data  + k*output->stride[0];
      real* ptr_output     = output_data     + k*output->stride[0];
      long i;
      for (i = 0; i < output->stride[0]; i++)
      {
	ptr_gradInput[i] = 0.5 * (ptr_gradOutput[i] / ptr_output[i]);
	//printf("%g %g %g\n",ptr_gradInput[i], ptr_gradOutput[i], ptr_output[i]);
      }
    }
  }
  return 1;
}

static const struct luaL_Reg nnOmp_(Sqrt__) [] = {
  {"Sqrt_forwardOmp", nnOmp_(Sqrt_forwardOmp)},
  {"Sqrt_backwardOmp", nnOmp_(Sqrt_backwardOmp)},
  {NULL, NULL}
};

static void nnOmp_(Sqrt_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  lua_getfield(L,-1,"nn");
  luaL_register(L, NULL, nnOmp_(Sqrt__));
  lua_pop(L,1);
}

#endif

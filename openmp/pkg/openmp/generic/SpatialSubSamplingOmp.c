#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialSubSamplingOmp.c"
#else

static int nnOmp_(SpatialSubSampling_forwardOmp)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  setompnthread(L,1,"nThread");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  real *weight_data = THTensor_(data)(weight);
  real *bias_data = THTensor_(data)(bias);
  real *output_data;
  real *input_data;

  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D(batch mode) tensor expected");

  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  if (input->nDimension == 4) {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }

  long inputWidth = input->size[dimw];
  long inputHeight = input->size[dimh];
  long outputWidth = (inputWidth - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;

  luaL_argcheck(L, input->size[dimh-1] == nInputPlane, 2, "invalid number of input planes");
  luaL_argcheck(L, inputWidth >= kW && inputHeight >= kH, 2, "input image smaller than kernel size");

  if (input->nDimension == 3)
    THTensor_(resize3d)(output, nInputPlane, outputHeight, outputWidth);
  else
    THTensor_(resize4d)(output, input->size[0], nInputPlane, outputHeight, outputWidth);
  
  input = THTensor_(newContiguous)(input);
  input_data = THTensor_(data)(input);
  output_data = THTensor_(data)(output);
  
  long k;
#pragma omp parallel for private(k)
  for(k = 0; k < nInputPlane; k++)
  {
    long p;
    for(p = 0; p < nbatch; p++)
    {
      long xx, yy;
      /* For all output pixels... */
      real *ptr_output = output_data + p*nInputPlane*outputWidth*outputHeight + k*outputWidth*outputHeight;
      /* Get the good mask for (k,i) (k out, i in) */
      real the_weight = weight_data[k];
      /* Initialize to the bias */
      real z = bias_data[k];
      long i;
      for(i = 0; i < outputWidth*outputHeight; i++)
	ptr_output[i] = z;
      
      for(yy = 0; yy < outputHeight; yy++)
      {
	for(xx = 0; xx < outputWidth; xx++)
	{
	  // Compute the mean of the input image...
	  real *ptr_input = input_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight + yy*dH*inputWidth+xx*dW;
	  real sum = 0;
	  long kx, ky;

	  for(ky = 0; ky < kH; ky++)
	  {
	    for(kx = 0; kx < kW; kx++)
	      sum += ptr_input[kx];
	    ptr_input += inputWidth; // next input line
	  }
	  // Update output
	  *ptr_output++ += the_weight*sum;
	}
      }
    }
  }
  THTensor_(free)(input);

  return 1;
}

static int nnOmp_(SpatialSubSampling_backwardOmp)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  setompnthread(L,1,"nThread");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  if (input->nDimension == 4) {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }

  long inputWidth = input->size[dimw];
  long inputHeight = input->size[dimh];
  long outputWidth = (inputWidth - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;

  real *weight_data = THTensor_(data)(weight);
  real *gradOutput_data = THTensor_(data)(gradOutput);
  real *input_data, *gradInput_data;

  input_data = THTensor_(data)(input);

  THTensor_(resizeAs)(gradInput, input);
  gradInput_data = THTensor_(data)(gradInput);
  gradOutput_data = THTensor_(data)(gradOutput);

  long k;
#pragma omp parallel for private(k)
  for(k = 0; k < nInputPlane; k++)
  {
    long p;
    for(p = 0; p < nbatch; p++)
    {
      real the_weight = weight_data[k];
      real *ptr_gradOutput = gradOutput_data + p*nInputPlane*outputHeight*outputWidth + k*outputWidth*outputHeight;
      long xx, yy;

      real* ptr_gi = gradInput_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight;
      long i;
      for(i=0; i<inputWidth*inputHeight; i++)
	ptr_gi[i] = 0.0;

      for(yy = 0; yy < outputHeight; yy++)
      {
	for(xx = 0; xx < outputWidth; xx++)
	{
	  real *ptr_gradInput = gradInput_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight + yy*dH*inputWidth+xx*dW;
	  real z = *ptr_gradOutput++ * the_weight;
	  long kx, ky;

	  for(ky = 0; ky < kH; ky++)
	  {
	    for(kx = 0; kx < kW; kx++)
	      ptr_gradInput[kx] += z;
	    ptr_gradInput += inputWidth;
	  }
	}
      }
    }
  }

  return 1;
}

static int nnOmp_(SpatialSubSampling_accGradParametersOmp)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  real scale = luaL_optnumber(L, 4, 1);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  setompnthread(L,1,"nThread");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_(Tensor_id));
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_(Tensor_id));
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_(Tensor_id));

  long nbatch = 1;
  long dimw = 2;
  long dimh = 1;
  if (input->nDimension == 4) {
    dimw++;
    dimh++;
    nbatch = input->size[0];
  }

  long inputWidth = input->size[dimw];
  long inputHeight = input->size[dimh];
  long outputWidth = (inputWidth - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;

  real *weight_data = THTensor_(data)(weight);
  real *gradWeight_data = THTensor_(data)(gradWeight);
  real *gradBias_data = THTensor_(data)(gradBias);
  real *gradOutput_data = THTensor_(data)(gradOutput);
  real *input_data, *gradInput_data;

  input = THTensor_(newContiguous)(input);
  input_data = THTensor_(data)(input);

  long k;
#pragma omp parallel for private(k)
  for(k = 0; k < nInputPlane; k++)
  {
    long p;
    for(p = 0; p < nbatch; p++)
    {
      real *ptr_gradOutput = gradOutput_data + p*nInputPlane*outputHeight*outputWidth + k*outputWidth*outputHeight;
      real sum;
      long xx, yy;

      sum = 0;
      long i;
      for(i = 0; i < outputWidth*outputHeight; i++)
	sum += ptr_gradOutput[i];
      gradBias_data[k] += scale*sum;

      sum = 0;
      for(yy = 0; yy < outputHeight; yy++)
      {
	for(xx = 0; xx < outputWidth; xx++)
	{
	  real *ptr_input = input_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight + yy*dH*inputWidth+xx*dW;
	  real z = *ptr_gradOutput++;
	  long kx, ky;

	  for(ky = 0; ky < kH; ky++)
	  {
	    for(kx = 0; kx < kW; kx++)
	      sum += z * ptr_input[kx];
	    ptr_input += inputWidth;
	  }
	}
      }
      gradWeight_data[k] += scale*sum;
    }
  }

  THTensor_(free)(input);

  return 0;
}

static const struct luaL_Reg nnOmp_(SpatialSubSampling__) [] = {
  {"SpatialSubSampling_forwardOmp", nnOmp_(SpatialSubSampling_forwardOmp)},
  {"SpatialSubSampling_backwardOmp", nnOmp_(SpatialSubSampling_backwardOmp)},
  {"SpatialSubSampling_accGradParametersOmp", nnOmp_(SpatialSubSampling_accGradParametersOmp)},
  {NULL, NULL}
};

static void nnOmp_(SpatialSubSampling_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  lua_getfield(L,-1,"nn");
  luaL_register(L, NULL, nnOmp_(SpatialSubSampling__));
  lua_pop(L,1);
}

#endif

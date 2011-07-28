#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THLabConv.h"
#else


TH_API void THLab_(validXCorr2Dptr)(real *r_,
				    real *t_, long ir, long ic, 
				    real *k_, long kr, long kc, 
				    long sr, long sc);
TH_API void THLab_(validConv2Dptr)(real *r_,
				   real *t_, long ir, long ic, 
				   real *k_, long kr, long kc, 
				   long sr, long sc);
			     
TH_API void THLab_(fullXCorr2Dptr)(real *r_,
				   real *t_, long ir, long ic, 
				   real *k_, long kr, long kc, 
				   long sr, long sc);
     
TH_API void THLab_(fullConv2Dptr)(real *r_,
				  real *t_, long ir, long ic, 
				  real *k_, long kr, long kc, 
				  long sr, long sc);

TH_API void THLab_(validXCorr2DRevptr)(real *r_,
				       real *t_, long ir, long ic, 
				       real *k_, long kr, long kc, 
				       long sr, long sc);

TH_API void THLab_(conv2DRevger)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long srow, long scol);
TH_API void THLab_(conv2Dger)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long srow, long scol, const char* type);
TH_API void THLab_(conv2Dmv)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long srow, long scol, const char *type);
TH_API void THLab_(conv2Dmul)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long srow, long scol, const char *type);

TH_API void THLab_(validXCorr3Dptr)(real *r_,
				    real *t_, long it, long ir, long ic, 
				    real *k_, long kt, long kr, long kc, 
				    long st, long sr, long sc);
TH_API void THLab_(validConv3Dptr)(real *r_,
				   real *t_, long it, long ir, long ic, 
				   real *k_, long kt, long kr, long kc, 
				   long st, long sr, long sc);
TH_API void THLab_(fullXCorr3Dptr)(real *r_,
				   real *t_, long it, long ir, long ic, 
				   real *k_, long kt, long kr, long kc, 
				   long st, long sr, long sc);
TH_API void THLab_(fullConv3Dptr)(real *r_,
				  real *t_, long it, long ir, long ic, 
				  real *k_, long kt, long kr, long kc, 
				  long st, long sr, long sc);

TH_API void THLab_(validXCorr3DRevptr)(real *r_,
				       real *t_, long it, long ir, long ic, 
				       real *k_, long kt, long kr, long kc, 
				       long st, long sr, long sc);

TH_API void THLab_(conv3DRevger)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol);
TH_API void THLab_(conv3Dger)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char* type);
TH_API void THLab_(conv3Dmv)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *type);
TH_API void THLab_(conv3Dmul)(THTensor *r_, real beta, THTensor *t_, THTensor *k_, long sdepth, long srow, long scol, const char *type);



#endif

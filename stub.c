/* Stand-in for ZLUDA's nvcuda.dll / nvml.dll: one CUDA 11.8 device, cc 8.6. */
#define X __declspec(dllexport)
X int cuInit(unsigned f){return 0;}
X int cuDriverGetVersion(int*v){*v=11080;return 0;}
X int cuDeviceGetCount(int*c){*c=1;return 0;}
X int cuDeviceGet(int*d,int i){*d=i;return 0;}
X int cuDeviceGetAttribute(int*v,int a,int d){*v=(a==75)?8:6;return 0;}
X int nvmlInit_v2(void){return 0;}
X int nvmlShutdown(void){return 0;}
X int nvmlSystemGetCudaDriverVersion_v2(int*v){*v=11080;return 0;}
X int nvmlDeviceGetCount_v2(unsigned*c){*c=1;return 0;}
X int nvmlDeviceGetHandleByIndex_v2(unsigned i,void**h){*h=(void*)1;return 0;}
X int nvmlDeviceGetCudaComputeCapability(void*h,int*a,int*b){*a=8;*b=6;return 0;}

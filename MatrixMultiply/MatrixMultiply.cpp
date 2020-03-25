#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/opencl.h>
#include <stdio.h>
#include <stdlib.h>

// Forward declaration of the OpenCL error checking function
void checkError(cl_int error, int line);

const char* kernelSource2 = "__kernel void simpleMultiply (const int M, const int N, const int K, \n"\
"const __global float* A,                           \n"\
"const __global float* B,                           \n"\
"__global float* C){                               \n"\
"   int TS = 32;                                 \n"\
"   const int row = get_local_id(0);                   \n"\
"   const int col = get_local_id(1);                   \n"\

"   const int globalRow = TS * get_group_id(0) + row;  \n"\
"   const int globalCol = TS * get_group_id(1) + col;  \n"\

"   __local float Asub[32][32];                        \n"\
"   __local float Bsub[32][32];                        \n"\
"   float acc = 0.0f;                                  \n"\
"   const int numTiles = K / TS;                       \n"\

"   for (int t = 0; t < numTiles; t++) {               \n"\
"       const int tiledRow = TS * t + row;                 \n"\
"       const int tiledCol = TS * t + col;                 \n"\
"       Asub[col][row] = A[tiledCol * M + globalRow];      \n"\
"       Bsub[col][row] = B[globalCol * K + tiledRow];      \n"\
"       barrier(CLK_LOCAL_MEM_FENCE);                      \n"\

"       for (int k = 0; k < TS; k++) {                     \n"\
"            acc += Asub[k][row] * Bsub[col][k];            \n"\
"       }                                                  \n"\
"       barrier(CLK_LOCAL_MEM_FENCE);                      \n"\
"   }                                                  \n"\
"   C[globalCol * M + globalRow] = acc;                \n"\
"}                                                  \n"\
"                                                   \n";


const char* kernelSource = "                    \n"\
"__kernel void simpleMultiply(__global float *outputC, \n"\
"                       int widthA,             \n"\
"                       int heightA,            \n"\
"                       int widthB,             \n"\
"                       int heightB,            \n"\
"                     __global float *inputA,   \n"\
"                     __global float *inputB){  \n"\
"   int row = get_global_id(1);                 \n"\
"   int col = get_global_id(0);                 \n"\
"   float sum = 0.0f;                           \n"\
//"   printf(\"%d \",row);\n"\
"   printf(\"%d%d \",row, col);                  \n"\

//"   printf(\"%d\", size);\n"\
"   printf(\"%d\", size2);\n"\

"   for(int i = 0; i<widthA; i++){              \n"\
"       sum += inputA[row * widthA + i] * inputB[i * widthB + col];\n"\
"   }                                           \n"\
"   outputC[row * widthB + col] = sum;          \n"\

"}                                              \n"\
"                                               \n";


const char* kernelSource3 = "                    \n"\
"__kernel void simpleMultiply(__global float *outputC, \n"\
"                       int widthA,             \n"\
"                       int heightA,            \n"\
"                       int widthB,             \n"\
"                       int heightB,            \n"\
"                     __global float *inputA,   \n"\
"                     __global float *inputB){  \n"\
"   int row = get_global_id(0);                 \n"\
"   int col = get_global_id(1);                 \n"\
"   float sum = 0.0f;                           \n"\
//"   printf(\"%d \",row);\n"\
"   printf(\"%d%d \",row, col);                  \n"\

//"   printf(\"%d\", size);\n"\
"   printf(\"%d\", size2);\n"\

"   for(int i = 0; i<widthA; i++){              \n"\
"       sum += inputA[row * widthA + i] * inputB[i * widthB + col];\n"\
"   }                                           \n"\
"   outputC[row * widthB + col] = sum;          \n"\

"}                                              \n"\
"                                               \n";

int main()
{

    //Device input buffers
    cl_mem bufferA;
    cl_mem bufferB;
    //Device output buffer
    cl_mem bufferC;
    cl_uint platformCount;
    cl_platform_id platforms;
    cl_device_id device_id;
    cl_uint ret_num_devices;
    cl_context context;
    cl_command_queue command_queue;
    cl_program my_program;
    cl_kernel my_kernel;
    cl_int ret;

    cl_event prof_event;
    cl_ulong time_start, time_end, total_time;

    int M = 1024;
    int N = 1024;
    int K = 1024;


    size_t bytes = M * N * sizeof(double);
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);

    for (int i = 0; i < M * K; i++) {
        h_a[i] = 1;
    }

    for (int j = 0; j < K * N; j++) {
        h_b[j] = 1;
    }


    ret = clGetPlatformIDs(1, &platforms, NULL);
    ret = clGetDeviceIDs(platforms, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &ret);
    command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

    float* h_c = (float*)calloc(M * N, sizeof(float));

    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, M * K * sizeof(float), NULL, NULL);
    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, K * N * sizeof(float), NULL, NULL);
    bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, M * N * sizeof(float), NULL, NULL);

    ret = clEnqueueWriteBuffer(command_queue, bufferA, CL_TRUE, 0, M * K * sizeof(float), (void*)h_a, 0, NULL, NULL);
    ret |= clEnqueueWriteBuffer(command_queue, bufferB, CL_TRUE, 0, K * N * sizeof(float), (void*)h_b, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf("Failed to enqueuewritebuffers.\n");
        exit(1);
    }

    my_program = clCreateProgramWithSource(context, 1, &kernelSource2, NULL, &ret);
    if (ret != CL_SUCCESS) {
        printf("Failed to create Program with Source.\n");
        exit(1);
    }

    ret = clBuildProgram(my_program, 0, NULL, NULL, NULL, NULL);
    /*if (ret != CL_SUCCESS) {
        printf("Error in Building Program\n");
        printf("%d", ret);
        exit(1);
    }*/


    // Check for compilation errors
    size_t logSize;
    ret = clGetProgramBuildInfo(my_program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    checkError(ret, __LINE__);
    char* messages = (char*)malloc((1 + logSize) * sizeof(char));
    ret = clGetProgramBuildInfo(my_program, device_id, CL_PROGRAM_BUILD_LOG, logSize, messages, NULL);
    checkError(ret, __LINE__);
    messages[logSize] = '\0';
    if (logSize > 10) { printf("## Compiler message: %s\n", messages); }
    free(messages);

    my_kernel = clCreateKernel(my_program, "simpleMultiply", &ret);


    //Kernel 1
    /*ret = clSetKernelArg(my_kernel, 0, sizeof(cl_mem), (void*)&bufferC);
    ret |= clSetKernelArg(my_kernel, 1, sizeof(cl_int), (void*)&K);
    ret |= clSetKernelArg(my_kernel, 2, sizeof(cl_int), (void*)&M);
    ret |= clSetKernelArg(my_kernel, 3, sizeof(cl_int), (void*)&N);
    ret |= clSetKernelArg(my_kernel, 4, sizeof(cl_int), (void*)&K);
    ret |= clSetKernelArg(my_kernel, 5, sizeof(cl_mem), (void*)&bufferA);
    ret |= clSetKernelArg(my_kernel, 6, sizeof(cl_mem), (void*)&bufferB);*/




    //Kernel 2
    ret = clSetKernelArg(my_kernel, 0, sizeof(int), (void*)&M);
    ret |= clSetKernelArg(my_kernel, 1, sizeof(int), (void*)&N);
    ret |= clSetKernelArg(my_kernel, 2, sizeof(int), (void*)&K);
    ret |= clSetKernelArg(my_kernel, 3, sizeof(cl_mem), (void*)&bufferA);
    ret |= clSetKernelArg(my_kernel, 4, sizeof(cl_mem), (void*)&bufferB);
    ret |= clSetKernelArg(my_kernel, 5, sizeof(cl_mem), (void*)&bufferC);


    const int TS = 32;
    size_t globalSize[2] = { (size_t)M, (size_t)N };
    size_t localSize[2] = { TS, TS };

    ret = clEnqueueNDRangeKernel(command_queue, my_kernel, 2, NULL, globalSize, localSize, 0, NULL, &prof_event);



    if (ret != CL_SUCCESS) {
        printf("Failed to enqueueNDRangeKernel.\n");
        exit(1);
    }

    clFinish(command_queue);

    clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    total_time = time_end - time_start;

    printf("%d", total_time);
    ret = clEnqueueReadBuffer(command_queue, bufferC, CL_TRUE, 0, M * N * sizeof(float), (void*)h_c, 0, NULL, NULL);


    free(h_c);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseProgram(my_program);
    clReleaseKernel(my_kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;



    /*
    // CL_DEVICE_MAX_COMPUTE_UNITS
    cl_uint compute_units;
    ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    printf("  CL_DEVICE_MAX_COMPUTE_UNITS:\t\t%u\n", compute_units);


    // CL_DEVICE_MAX_WORK_ITEM_SIZES
    size_t workitem_size[3];
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
    printf("  CL_DEVICE_MAX_WORK_ITEM_SIZES:\t%u / %u / %u \n", workitem_size[0], workitem_size[1], workitem_size[2]);


    // CL_DEVICE_MAX_WORK_GROUP_SIZE
    size_t workgroup_size;
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL);
    printf("  CL_DEVICE_MAX_WORK_GROUP_SIZE:\t%u\n", workgroup_size);


    // CL_DEVICE_GLOBAL_MEM_SIZE
    cl_ulong mem_size;
    clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
    printf("  CL_DEVICE_GLOBAL_MEM_SIZE:\t\t%u MByte\n", (unsigned int)(mem_size / (1024 * 1024)));


    // CL_DEVICE_LOCAL_MEM_SIZE
    clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
    printf("  CL_DEVICE_LOCAL_MEM_SIZE:\t\t%u KByte\n", (unsigned int)(mem_size / 1024));
    */
}



// Print an error message to screen (only if it occurs)
void checkError(cl_int error, int line) {
    if (error != CL_SUCCESS) {
        switch (error) {
        case CL_DEVICE_NOT_FOUND:                 printf("-- Error at %d:  Device not found.\n", line); break;
        case CL_DEVICE_NOT_AVAILABLE:             printf("-- Error at %d:  Device not available\n", line); break;
        case CL_COMPILER_NOT_AVAILABLE:           printf("-- Error at %d:  Compiler not available\n", line); break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:    printf("-- Error at %d:  Memory object allocation failure\n", line); break;
        case CL_OUT_OF_RESOURCES:                 printf("-- Error at %d:  Out of resources\n", line); break;
        case CL_OUT_OF_HOST_MEMORY:               printf("-- Error at %d:  Out of host memory\n", line); break;
        case CL_PROFILING_INFO_NOT_AVAILABLE:     printf("-- Error at %d:  Profiling information not available\n", line); break;
        case CL_MEM_COPY_OVERLAP:                 printf("-- Error at %d:  Memory copy overlap\n", line); break;
        case CL_IMAGE_FORMAT_MISMATCH:            printf("-- Error at %d:  Image format mismatch\n", line); break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:       printf("-- Error at %d:  Image format not supported\n", line); break;
        case CL_BUILD_PROGRAM_FAILURE:            printf("-- Error at %d:  Program build failure\n", line); break;
        case CL_MAP_FAILURE:                      printf("-- Error at %d:  Map failure\n", line); break;
        case CL_INVALID_VALUE:                    printf("-- Error at %d:  Invalid value\n", line); break;
        case CL_INVALID_DEVICE_TYPE:              printf("-- Error at %d:  Invalid device type\n", line); break;
        case CL_INVALID_PLATFORM:                 printf("-- Error at %d:  Invalid platform\n", line); break;
        case CL_INVALID_DEVICE:                   printf("-- Error at %d:  Invalid device\n", line); break;
        case CL_INVALID_CONTEXT:                  printf("-- Error at %d:  Invalid context\n", line); break;
        case CL_INVALID_QUEUE_PROPERTIES:         printf("-- Error at %d:  Invalid queue properties\n", line); break;
        case CL_INVALID_COMMAND_QUEUE:            printf("-- Error at %d:  Invalid command queue\n", line); break;
        case CL_INVALID_HOST_PTR:                 printf("-- Error at %d:  Invalid host pointer\n", line); break;
        case CL_INVALID_MEM_OBJECT:               printf("-- Error at %d:  Invalid memory object\n", line); break;
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  printf("-- Error at %d:  Invalid image format descriptor\n", line); break;
        case CL_INVALID_IMAGE_SIZE:               printf("-- Error at %d:  Invalid image size\n", line); break;
        case CL_INVALID_SAMPLER:                  printf("-- Error at %d:  Invalid sampler\n", line); break;
        case CL_INVALID_BINARY:                   printf("-- Error at %d:  Invalid binary\n", line); break;
        case CL_INVALID_BUILD_OPTIONS:            printf("-- Error at %d:  Invalid build options\n", line); break;
        case CL_INVALID_PROGRAM:                  printf("-- Error at %d:  Invalid program\n", line); break;
        case CL_INVALID_PROGRAM_EXECUTABLE:       printf("-- Error at %d:  Invalid program executable\n", line); break;
        case CL_INVALID_KERNEL_NAME:              printf("-- Error at %d:  Invalid kernel name\n", line); break;
        case CL_INVALID_KERNEL_DEFINITION:        printf("-- Error at %d:  Invalid kernel definition\n", line); break;
        case CL_INVALID_KERNEL:                   printf("-- Error at %d:  Invalid kernel\n", line); break;
        case CL_INVALID_ARG_INDEX:                printf("-- Error at %d:  Invalid argument index\n", line); break;
        case CL_INVALID_ARG_VALUE:                printf("-- Error at %d:  Invalid argument value\n", line); break;
        case CL_INVALID_ARG_SIZE:                 printf("-- Error at %d:  Invalid argument size\n", line); break;
        case CL_INVALID_KERNEL_ARGS:              printf("-- Error at %d:  Invalid kernel arguments\n", line); break;
        case CL_INVALID_WORK_DIMENSION:           printf("-- Error at %d:  Invalid work dimensionsension\n", line); break;
        case CL_INVALID_WORK_GROUP_SIZE:          printf("-- Error at %d:  Invalid work group size\n", line); break;
        case CL_INVALID_WORK_ITEM_SIZE:           printf("-- Error at %d:  Invalid work item size\n", line); break;
        case CL_INVALID_GLOBAL_OFFSET:            printf("-- Error at %d:  Invalid global offset\n", line); break;
        case CL_INVALID_EVENT_WAIT_LIST:          printf("-- Error at %d:  Invalid event wait list\n", line); break;
        case CL_INVALID_EVENT:                    printf("-- Error at %d:  Invalid event\n", line); break;
        case CL_INVALID_OPERATION:                printf("-- Error at %d:  Invalid operation\n", line); break;
        case CL_INVALID_GL_OBJECT:                printf("-- Error at %d:  Invalid OpenGL object\n", line); break;
        case CL_INVALID_BUFFER_SIZE:              printf("-- Error at %d:  Invalid buffer size\n", line); break;
        case CL_INVALID_MIP_LEVEL:                printf("-- Error at %d:  Invalid mip-map level\n", line); break;
        case -1024:                               printf("-- Error at %d:  *clBLAS* Functionality is not implemented\n", line); break;
        case -1023:                               printf("-- Error at %d:  *clBLAS* Library is not initialized yet\n", line); break;
        case -1022:                               printf("-- Error at %d:  *clBLAS* Matrix A is not a valid memory object\n", line); break;
        case -1021:                               printf("-- Error at %d:  *clBLAS* Matrix B is not a valid memory object\n", line); break;
        case -1020:                               printf("-- Error at %d:  *clBLAS* Matrix C is not a valid memory object\n", line); break;
        case -1019:                               printf("-- Error at %d:  *clBLAS* Vector X is not a valid memory object\n", line); break;
        case -1018:                               printf("-- Error at %d:  *clBLAS* Vector Y is not a valid memory object\n", line); break;
        case -1017:                               printf("-- Error at %d:  *clBLAS* An input dimension (M,N,K) is invalid\n", line); break;
        case -1016:                               printf("-- Error at %d:  *clBLAS* Leading dimension A must not be less than the size of the first dimension\n", line); break;
        case -1015:                               printf("-- Error at %d:  *clBLAS* Leading dimension B must not be less than the size of the second dimension\n", line); break;
        case -1014:                               printf("-- Error at %d:  *clBLAS* Leading dimension C must not be less than the size of the third dimension\n", line); break;
        case -1013:                               printf("-- Error at %d:  *clBLAS* The increment for a vector X must not be 0\n", line); break;
        case -1012:                               printf("-- Error at %d:  *clBLAS* The increment for a vector Y must not be 0\n", line); break;
        case -1011:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix A is too small\n", line); break;
        case -1010:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix B is too small\n", line); break;
        case -1009:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix C is too small\n", line); break;
        case -1008:                               printf("-- Error at %d:  *clBLAS* The memory object for Vector X is too small\n", line); break;
        case -1007:                               printf("-- Error at %d:  *clBLAS* The memory object for Vector Y is too small\n", line); break;
        case -1001:                               printf("-- Error at %d:  Code -1001: no GPU available?\n", line); break;
        default:                                  printf("-- Error at %d:  Unknown with code %d\n", line, error);
        }
        exit(1);
    }
}
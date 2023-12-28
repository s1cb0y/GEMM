#include <CL/opencl.h>

// OpenCL Kernel for BLOCK m a t r i x m u l t i p l y . C = A ∗ B
//
const char *KernelSource = "\n" \
" __kernel void myGEMM1(const __global float* A, \n" \
"                       const __global float* B, \n" \
"                       __global float* C, \n" \
"                       const int N) { \n" \
"      \n" \
"     // Thread identifiers \n" \
"     const int globalRow = get_global_id(0); // Row ID of C (0..N) \n" \
"     const int globalCol = get_global_id(1); // Col ID of C (0..N) \n" \
"     \n" \
"     // Compute a single element (loop over N) \n" \
"     float acc = 0.0f; \n" \
"     for (int k=0; k<N; k++) { \n" \
"         acc += A[globalRow*N + k] * B[N * k + globalCol]; \n" \
"     } \n" \
"     \n" \
"     // Store the result \n" \
"     C[globalRow*N + globalCol] = acc; \n" \
" }   \n" \
"";

int multiplyGPU(float* A, float* B, float *C, uint32_t N){
    int err = 0;
   
    const int TS = 32;
    const size_t local[2] = { TS, TS };
    const size_t global[2] = { N, N };
                  // local domain size for our calculation
    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    
    cl_mem cl_A;                       // device memory used for the input array
    cl_mem cl_B;                       // device memory used for the input array
    cl_mem cl_C;                       // device memory used for the output array

    // Connect to a compute device
    //
    int gpu = 1;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }

    // Create a compute context 
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    //
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, "myGEMM1", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

     // Create the input and output arrays in device memory for our calculation
    //
    cl_A = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * N*N, NULL, NULL);
    cl_B = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * N*N, NULL, NULL);
    cl_C = clCreateBuffer(context,  CL_MEM_WRITE_ONLY, sizeof(float) * N*N, NULL, NULL);
    if (!cl_A || !cl_B || !cl_C)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }   

    // Write our data set into the input array in device memory 
    //
    err = clEnqueueWriteBuffer(commands, cl_A, CL_TRUE, 0, sizeof(float) * N*N, A, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, cl_B, CL_TRUE, 0, sizeof(float) * N*N, B, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }

    // Set the arguments to our compute kernel
    //
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_A);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_B);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_C);    
    err |= clSetKernelArg(kernel, 3, sizeof(int), &N);    
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }


      // Get the maximum work group size for executing the kernel on the device
    //
    size_t _local;
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(_local), &_local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global, local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands);

    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer( commands, cl_C, CL_TRUE, 0, sizeof(float) * N, C, 0, NULL, NULL );  
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    return err;
    
}
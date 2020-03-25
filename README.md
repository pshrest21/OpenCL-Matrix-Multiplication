# OpenCL-Matrix-Multiplication
This is a GPU programming in OpenCL for matrix multiplication that uses two approaches.
1. Using global memory
For this approach, comment out the void "kernelSource2" function (line 10) and replace "kernelSource2" with "kernelSource" when
creating you program in line 151. Then, comment out the code for clSetKernelArg for "//Kernal 2" at line 191, and
uncomment the code for "//Kernel 1" in line 179.
Once you do this, your code should run. In this program, I measuring the time taken to complete the matrix
multiplication using two methods. The time taken to perform the operation using global memory would be
around 22 million nanoseconds

2. Using local memory
For this approach, do the exact opposite of before. Comment out the "kernelSource" function (line 43) and replace "kernelSource" with "kernelSource2" when
creating you program in line 151. Then, comment out the code for clSetKernelArg for "//Kernel 1" at line 179, and
uncomment the code for "//Kernel 2" in line 191.
Once you do this, your code should run.The time taken to perform the operation using local memory would be
around 9 million nanoseconds

So you can see that the performance increased significantly as the time taken to perform the same operation
is reduced by more than half. You can play with the code and see different results. Be sure you set your
local work group size more sensibly or else you can freeze your whole system and you might have to hard
reset your system.
Have fun!

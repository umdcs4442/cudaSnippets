#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cout << "No CUDA Capable device(s) detected.\n";
        return 0;
    }

    std::cout << "./deviceQuery Starting...\n\n";
    std::cout << "Detected " << deviceCount << " CUDA Capable device(s)\n";

    for (int dev = 0; dev < deviceCount; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        int driverVersion = 0, runtimeVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);

        std::cout << "\nDevice " << dev << ": \"" << prop.name << "\"\n";
        std::cout << "  CUDA Driver Version / Runtime Version          "
                  << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << " / "
                  << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10 << "\n";
        std::cout << "  CUDA Capability Major/Minor version number:    "
                  << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total amount of global memory:                 "
                  << prop.totalGlobalMem / (1024 * 1024) << " MBytes ("
                  << prop.totalGlobalMem << " bytes)\n";
        std::cout << "  (" << prop.multiProcessorCount << ") Multiprocessors\n";
        std::cout << "  GPU Max Clock rate:                            "
                  << prop.clockRate * 1e-3f << " MHz ("
                  << prop.clockRate * 1e-6f << " GHz)\n";
        std::cout << "  Memory Clock rate:                             "
                  << prop.memoryClockRate * 1e-3f << " Mhz\n";
        std::cout << "  Memory Bus Width:                              "
                  << prop.memoryBusWidth << "-bit\n";
        std::cout << "  L2 Cache Size:                                 "
                  << prop.l2CacheSize << " bytes\n";
        std::cout << "  Total amount of constant memory:               "
                  << prop.totalConstMem << " bytes\n";
        std::cout << "  Total amount of shared memory per block:       "
                  << prop.sharedMemPerBlock << " bytes\n";
        std::cout << "  Total shared memory per multiprocessor:        "
                  << prop.sharedMemPerMultiprocessor << " bytes\n";
        std::cout << "  Total number of registers available per block: "
                  << prop.regsPerBlock << "\n";
        std::cout << "  Warp size:                                     "
                  << prop.warpSize << "\n";
        std::cout << "  Maximum number of threads per multiprocessor:  "
                  << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "  Maximum number of threads per block:           "
                  << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Max dimension size of a thread block (x,y,z):  ("
                  << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << ")\n";
        std::cout << "  Max dimension size of a grid size (x,y,z):      ("
                  << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << ")\n";
        std::cout << "  Concurrent copy and kernel execution:          "
                  << (prop.deviceOverlap ? "Yes with copy engine(s)" : "No") << "\n";
        std::cout << "  Support host page-locked memory mapping:       "
                  << (prop.canMapHostMemory ? "Yes" : "No") << "\n";
        std::cout << "  Device has ECC support:                        "
                  << (prop.ECCEnabled ? "Enabled" : "Disabled") << "\n";
        std::cout << "  Device supports Unified Addressing (UVA):      "
                  << (prop.unifiedAddressing ? "Yes" : "No") << "\n";
        std::cout << "  Device supports Managed Memory:                "
                  << (prop.managedMemory ? "Yes" : "No") << "\n";
    }

    std::cout << "\nResult = PASS\n";
    return 0;
}

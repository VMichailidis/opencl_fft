#include "CL/cl.h"
#include "common.h"
#include "fft.hpp"
#include <CL/opencl.h>
#include <assert.h>
#include <chrono>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
// #include <vector>
const double PI = acos(-1);

#define KERNEL_NAME "libra"

#define FLOAT_ULP 10

#define CL_CHECK(_expr)                                              \
    do {                                                             \
        cl_int _err = _expr;                                         \
        if (_err == CL_SUCCESS)                                      \
            break;                                                   \
        printf("OpenCL Error: '%s' returned %d!\n", #_expr,          \
               (int)_err);                                           \
        cleanup();                                                   \
        exit(-1);                                                    \
    } while (0)

#define CL_CHECK2(_expr)                                             \
    ({                                                               \
        cl_int _err = CL_INVALID_VALUE;                              \
        decltype(_expr) _ret = _expr;                                \
        if (_err != CL_SUCCESS) {                                    \
            printf("OpenCL Error: '%s' returned %d!\n", #_expr,      \
                   (int)_err);                                       \
            cleanup();                                               \
            exit(-1);                                                \
        }                                                            \
        _ret;                                                        \
    })

static int read_kernel_file(const char *filename, uint8_t **data,
                            size_t *size) {
    if (nullptr == filename || nullptr == data || 0 == size)
        return -1;

    FILE *fp = fopen(filename, "r");
    if (NULL == fp) {
        fprintf(stderr, "Failed to load kernel.");
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    rewind(fp);

    *data = (uint8_t *)malloc(fsize);
    *size = fread(*data, 1, fsize, fp);

    fclose(fp);

    return 0;
}

template <typename Type> class Comparator {};

template <> class Comparator<int> {
  public:
    static const char *type_str() { return "integer"; }
    static int generate() { return rand(); }
    static bool compare(int a, int b, int index, int errors) {
        if (a != b) {
            if (errors < 100) {
                printf("*** error: [%d] expected=%d, actual=%d\n",
                       index, a, b);
            }
            return false;
        }
        return true;
    }
};

template <> class Comparator<float> {
  public:
    static const char *type_str() { return "float"; }
    static int generate() {
        return static_cast<float>(rand()) / RAND_MAX;
    }
    static bool compare(float a, float b, int index, int errors) {
        union fi_t {
            float f;
            int32_t i;
        };
        fi_t fa, fb;
        fa.f = a;
        fb.f = b;
        auto d = std::abs(fa.i - fb.i);
        if (d > FLOAT_ULP) {
            if (errors < 100) {
                printf("*** error: [%d] expected=%f, got=%f\n", index,
                       a, b);
            }
            return false;
        }
        return true;
    }
};
template <> class Comparator<std::vector<float>> {
  public:
    static const char *type_str() { return "float"; }
    static std::vector<float> generate(int i, int duration,
                                       int samples) {
        std::vector<float> f = {
            // cos(2 * PI * 5 * i * duration / samples), 0};
            (float)i / 2.0, (float)i / 2.0};
        return f;
    }
    static bool compare(float a, float b, int index, int errors) {
        union fi_t {
            float f;
            int32_t i;
        };
        fi_t fa, fb;
        fa.f = a;
        fb.f = b;
        auto d = std::abs(fa.i - fb.i);
        if (d > FLOAT_ULP) {
            if (errors < 100) {
                printf("*** error: [%d] expected=%f, actual=%f\n",
                       index, a, b);
            }
            return false;
        }
        return true;
    }
};

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue command_queue = NULL;
cl_program program = NULL;
uint8_t *kernel_bin = NULL;
cl_mem s_memobj = NULL;
static void cleanup() {
    if (command_queue)
        clReleaseCommandQueue(command_queue);
    if (program)
        clReleaseProgram(program);
    if (s_memobj)
        clReleaseMemObject(s_memobj);
    if (context)
        clReleaseContext(context);
    if (device_id)
        clReleaseDevice(device_id);
    if (kernel_bin)
        free(kernel_bin);
}

static void init_gpu(cl_platform_id *platform_id,
                     cl_device_id *device_id, cl_context *context,
                     cl_command_queue *command_queue) {

    // Getting platform and device information
    CL_CHECK(clGetPlatformIDs(1, platform_id, NULL));
    CL_CHECK(clGetDeviceIDs(*platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
                            device_id, NULL));

    printf("Create context\n");
    *context = CL_CHECK2(
        clCreateContext(NULL, 1, device_id, NULL, NULL, &_err));
    *command_queue = CL_CHECK2(
        clCreateCommandQueue(*context, *device_id, 0, &_err));
}

static double benchmark_fft(cl_device_id *device_id,
                            cl_command_queue *command_queue,
                            TYPE *data, size_t len,
                            size_t block_size) {
    // Init buffer
    size_t nbytes = len * sizeof(TYPE);
    s_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        nbytes, NULL, &_err));

    printf("Create program from kernel source\n");
    size_t kernel_size;
    if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
        return -1.0;
    program = CL_CHECK2(clCreateProgramWithSource(
        context, 1, (const char **)&kernel_bin, &kernel_size, &_err));

    // Build program
    CL_CHECK(clBuildProgram(program, 1, device_id, NULL, NULL, NULL));

    // Create kernels
    cl_kernel fft_k, bit_reverse_k = NULL;
    fft_k = CL_CHECK2(clCreateKernel(program, "fft", &_err));
    bit_reverse_k =
        CL_CHECK2(clCreateKernel(program, "libra", &_err));

    // Calculate correct kernel arguements
    size_t word_width = log2(len) - 1;
    size_t half_width = word_width / 2;
    size_t hlen = 1 << half_width;
    //

    // Set kernel arguments for bitreverse
    CL_CHECK(clSetKernelArg(bit_reverse_k, 0, sizeof(cl_mem),
                            (void *)&s_memobj));
    CL_CHECK(
        clSetKernelArg(bit_reverse_k, 1, sizeof(int), (void *)&hlen));
    CL_CHECK(clSetKernelArg(bit_reverse_k, 2, sizeof(int),
                            (void *)&half_width));

    // Set kernel arguements for fft
    CL_CHECK(
        clSetKernelArg(fft_k, 0, sizeof(cl_mem), (void *)&s_memobj));
    CL_CHECK(
        clSetKernelArg(fft_k, 1, sizeof(int), (void *)&block_size));

    printf("Upload source buffers\n");
    CL_CHECK(clEnqueueWriteBuffer(*command_queue, s_memobj, CL_TRUE,
                                  0, nbytes, data, 0, NULL, NULL));

    printf("Execute the kernel\n");
    size_t bit_reverse_size[1] = {hlen};
    size_t fft_size[1] = {block_size};
    size_t local_work_size[1] = {1};
    auto time_start = std::chrono::high_resolution_clock::now();
    CL_CHECK(clEnqueueNDRangeKernel(*command_queue, bit_reverse_k, 1,
                                    NULL, bit_reverse_size,
                                    local_work_size, 0, NULL, NULL));
    CL_CHECK(clEnqueueNDRangeKernel(*command_queue, fft_k, 1, NULL,
                                    fft_size, local_work_size, 0,
                                    NULL, NULL));
    CL_CHECK(clFinish(*command_queue));
    auto time_end = std::chrono::high_resolution_clock::now();

    double elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            time_end - time_start)
            .count();

    printf("Download destination buffer\n");
    CL_CHECK(clEnqueueReadBuffer(*command_queue, s_memobj, CL_TRUE, 0,
                                 nbytes, data, 0, NULL, NULL));
    return elapsed;
}

uint32_t len = 64; //
uint32_t duration = 10;
size_t block_size = len / 2;
static void show_usage() { printf("Usage: [-n size] [-h: help]\n"); }

static void parse_args(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "n:h")) != -1) {
        switch (c) {
        case 'n':
            len = atoi(optarg);
            block_size = len / 2;
            break;
        case 'h':
            show_usage();
            exit(0);
            break;
        default:
            show_usage();
            exit(-1);
        }
    }

    printf("Workload size=%d\n", len);
}

int main(int argc, char **argv) {
    // parse command arguments
    parse_args(argc, argv);

    cl_platform_id platform_id;
    init_gpu(&platform_id, &device_id, &context, &command_queue);
    // Important to set the work group to maximum size for maximum
    // compute unit unitilization

    // Generate input values
    std::vector<TYPE> h_s(2 * len);
    for (uint32_t i = 0; i < 2 * len; i += 2) {
        std::vector<TYPE> temp =
            Comparator<std::vector<TYPE>>::generate(i, duration, len);
        h_s[i] = temp.data()[0];
        h_s[i + 1] = temp.data()[1];
    }
    CArray h_ref(len);
    for (uint32_t i = 0; i < len; ++i) {
        Complex tmp = Complex(h_s[2 * i], h_s[2 * i + 1]);
        h_ref[i] = tmp;
    }

    // Benchmark
    double elapsed =
        benchmark_fft(&device_id, &command_queue, h_s.data(),
                      h_s.size(), block_size);
    if (elapsed == -1.0)
        return -1;
    printf("Elapsed time: %lg ms\n", elapsed);

    printf("Verify result\n");

    // Validate
    fft(h_ref);
    int errors = 0;
    for (uint32_t i = 0; i < len; ++i) {
        if (!Comparator<TYPE>::compare(real(h_ref[i]), h_s[2 * i], i,
                                       errors) ||
            !Comparator<TYPE>::compare(imag(h_ref[i]), h_s[2 * i + 1],
                                       i, errors)) {
            ++errors;
        }
    }
    if (0 == errors) {
        printf("PASSED!\n");
    } else {
        printf("FAILED! - %d errors\n", errors);
    }

    // Clean up
    cleanup();

    return errors;
}

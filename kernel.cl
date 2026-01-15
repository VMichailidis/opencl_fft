
#include "common.h"

__kernel void fft_gpu (__global TYPE * A)
{
  int gid = get_global_id(0);
  A[2 * gid] = A[2 * gid] + 2;
  A[2 * gid+1] = A[2 * gid+1] + 2;
}

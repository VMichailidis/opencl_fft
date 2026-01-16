
#include "common.h"

int reverse(int x, int w);

void swap(__global TYPE * X, __global TYPE * Y);

__kernel void fft_gpu (__global TYPE * A, int hlen, int hw)
{
  int gid = get_global_id(0); // gid is the column
  for (int lo = 1; lo < hlen; lo++){
    int rlo = reverse(lo, hw);
    if(gid < rlo) {
      int ru = reverse(gid, hw);
      int index = (gid << hw) | lo;
      int xendi = (rlo << hw) | ru;
      swap(&(A[2 * index]), &(A[2 * xendi]));
      swap(&(A[2 * index + 1]), &(A[2 * xendi + 1]));
    }
  }

  // A[2 * gid] = A[2 * gid] + 2;
  // A[2 * gid+1] = A[2 * gid+1] + 2;
}

__kernel void fft_gpu_vanilla(__global TYPE * A, int len, int width) {
  int gid = get_global_id(0);
  int dig = reverse(gid, width);
  if (gid < dig) {
    swap(&(A[2 * gid]), &(A[2 * dig]));
    swap(&(A[2 * gid + 1]), &(A[2 * dig + 1]));
  }
}

int reverse(int x, int w) {
  int reversed = 0;
  for (int j = 0; j < w; j++) // log2(N) = 3 bits needed to represent indices
    reversed = (reversed << 1) | (x >> j & 1);
  return reversed;
}

void swap(__global TYPE * X, __global TYPE * Y){
  TYPE tmp = *X;
  *X = *Y;
  *Y = tmp;
}

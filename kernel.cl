
#include "common.h"
#define PI 3.1415926535897932
int reverse(int x, int w);

void swap(__global TYPE * X, __global TYPE * Y);

__kernel void libra (__global TYPE * A, int hlen, int hw)
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
}

__kernel void bit_reverse(__global TYPE * A, int len, int width) {
  int gid = get_global_id(0);
  int dig = reverse(gid, width);
  if (gid < dig) {
    swap(&(A[2 * gid]), &(A[2 * dig]));
    swap(&(A[2 * gid + 1]), &(A[2 * dig + 1]));
  }
}
__kernel void fft (__global TYPE * A, int len){
  // inspired by https://github.com/dimitarkyurtov/fft-gpu/blob/main/src/opencl/fft.cpp#L8
  int gid = get_global_id(0);
  for (int stride = 2; stride <= len; stride <<= 1) {
    int group = gid / (stride / 2) * stride;
    int offset = gid % (stride / 2);
    TYPE angle = -2 * PI * offset /stride;
    TYPE W_r = cos(angle);
    TYPE W_i = sin(angle);

    TYPE u_r = A[2*(group + offset)];
    TYPE u_i = A[2*(group + offset) + 1];
    TYPE v_r = A[2*(group + offset + (stride / 2))];
    TYPE v_i = A[2*(group + offset + (stride / 2)) + 1];

    TYPE c_r = W_r * v_r - W_i * v_i;
    TYPE c_i = W_r * v_i + W_i * v_r;

    A[2*(group + offset)] = u_r + c_r;
    A[2*(group + offset) + 1] = u_i + c_i;
    A[2*(group + offset + (stride / 2))] = u_r - c_r;
    A[2*(group + offset + (stride / 2)) + 1] = u_i - c_i;
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

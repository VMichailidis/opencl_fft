
#include "common.h"

__kernel void vecadd (__global TYPE * A)
{
  int gid = get_global_id(0);
  A[2*gid] = A[2*gid] + 2;
  A[2*gid+sizeof(TYPE)] = A[gid+sizeof(TYPE)] + 2;
}

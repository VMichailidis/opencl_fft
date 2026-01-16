// Modified version from here:
// https://www.w3computing.com/articles/how-to-implement-a-fast-fourier-transform-fft-in-cpp/
#include "fft.hpp"
const double PI = acos(-1);

int log_2(int x);
int reverse(int x, int w);

int reverse(int x, int w) {
    int reversed = 0;
    for (int j = 0; j < w;
         j++) // log2(N) = 3 bits needed to represent indices
        reversed = (reversed << 1) | (x >> j & 1);
    return reversed;
}

int log2(int x) {
    int log = 0;
    while (x >>= 1)
        log++;
    return log;
}

void libra_even(CArray &s) {
    size_t len = s.size();
    size_t w = log2(len);
    size_t hw = w / 2;
    size_t hlen = 1 << hw;
    for (int lo = 0; lo < hlen; lo++) {
        int rlo = reverse(lo, hw);
        for (int u = 0; u < rlo; u++) {
            int ru = reverse(u, hw);
            int index = (u << hw) | lo;
            int xendi = (rlo << hw) | ru;

            Complex temp = s[index];
            s[index] = s[xendi];
            s[xendi] = temp;
        }
    }
}

void libra_odd(CArray &s) {
    size_t len = s.size();
    size_t w = log2(len);
    size_t hw = w / 2;
    int mp = 1 << hw;
    size_t hlen = 1 << hw;
    for (int lo = 0; lo < hlen; lo++) {
        int rlo = reverse(lo, hw);
        for (int u = 0; u < rlo; u++) {
            int ru = reverse(u, hw);
            {
                int index = (u << (hw + 1)) | lo;
                int xendi = (rlo << (hw + 1)) | ru;

                Complex temp = s[index];
                s[index] = s[xendi];
                s[xendi] = temp;
            }
            {
                int index = (u << (hw + 1)) | mp | lo;
                int xendi = (rlo << (hw + 1)) | mp | ru;

                Complex temp = s[index];
                s[index] = s[xendi];
                s[xendi] = temp;
            }
        }
    }
}

void libra(CArray &s) {
    // CArray: Array to be placed in bitreverse order
    int len = s.size();
    if (log2(len) % 2) {
        libra_odd(s);
    } else {
        libra_even(s);
    }
    return;
}

void fft(CArray &s) {
    const size_t N = s.size();
    if (N <= 1) {
        return;
    }
    libra(s);
    // Iterative FFT
    // for (size_t len = 2; len <= N; len <<= 1) {
    //     double angle = -2 * PI / len;
    //     Complex wlen(cos(angle), sin(angle));
    //     for (size_t i = 0; i < N; i += len) {
    //         Complex w(1);
    //         for (size_t j = 0; j < len / 2; ++j) {
    //             Complex u = s[i + j];
    //             Complex v = s[i + j + len / 2] * w;
    //             s[i + j] = u + v;
    //             s[i + j + len / 2] = u - v;
    //             w *= wlen;
    //         }
    //     }
    // }
}

#ifndef _UTILS_
#define _UTILS_

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

static inline int u64_to_str_with_sep(uint64_t x, char sep, char *out) {
  if (x < 1000) {
    return sprintf(out, "%llu", (unsigned long long)x);
  }

  // x >= 1000
  int c1 = u64_to_str_with_sep(x / 1000, sep, out);
  out[c1] = sep;

  uint64_t y = x % 1000;
  assert(0 <= y && y <= 999);
  int c2;
  if (y >= 100) {
    c2 = sprintf(out + c1 + 1, "%llu", (unsigned long long)y);
  } else if (y >= 10) {
    c2 = sprintf(out + c1 + 1, "0%llu", (unsigned long long)y);
  } else {
    c2 = sprintf(out + c1 + 1, "00%llu", (unsigned long long)y);
  }
  assert(c2 == 3);

  return c1 + 4;
}

#endif // _UTILS_

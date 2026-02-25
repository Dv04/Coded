#ifndef _WINDOW_BIT_COUNT_APX_
#define _WINDOW_BIT_COUNT_APX_

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

uint64_t N_MERGES = 0; // keep track of how many bucket merges occur

/*
    DGIM approximation algorithm for counting 1's over a sliding window.

    Efficient O(1) amortized design:
    - Separate circular buffer (ring) per power level, holding timestamps.
    - Each ring has capacity r+2 (max buckets before merge = r+2).
    - Insert at power-0 ring. If ring overflows (r+2 entries), pop 2 oldest
      from power p, push 1 onto power (p+1). Cascade up.
    - Expiry: the oldest bucket overall is the tail of the highest occupied
      power level. Check and pop if expired.
    - Query: total_sum - oldest_size + 1.

    All rings are allocated from a single contiguous block of memory.
*/

#define MAX_POWER_LEVELS 34 // covers window sizes up to 2^32

typedef struct {
  uint32_t *timestamps; // pointer into pooled allocation
  uint32_t capacity;    // r + 2
  uint32_t head;        // index of newest entry
  uint32_t tail;        // index of oldest entry
  uint32_t count;       // number of entries
} Ring;

typedef struct {
  uint32_t wnd_size;
  uint32_t k;
  uint32_t r;
  uint32_t timestamp;
  uint32_t total_sum;
  uint32_t num_power_levels;
  int32_t highest_power; // highest power level with any bucket (-1 if none)
  Ring rings[MAX_POWER_LEVELS];
  uint32_t *pool; // single malloc for all ring buffers
} StateApx;

static inline void ring_init(Ring *ring, uint32_t *buf, uint32_t cap) {
  ring->timestamps = buf;
  ring->capacity = cap;
  ring->head = 0;
  ring->tail = 0;
  ring->count = 0;
}

static inline void ring_push(Ring *ring, uint32_t ts) {
  assert(ring->count < ring->capacity);
  if (ring->count == 0) {
    ring->head = 0;
    ring->tail = 0;
  } else {
    ring->head = (ring->head + 1) % ring->capacity;
  }
  ring->timestamps[ring->head] = ts;
  ring->count++;
}

static inline uint32_t ring_pop_tail(Ring *ring) {
  assert(ring->count > 0);
  uint32_t ts = ring->timestamps[ring->tail];
  ring->count--;
  if (ring->count > 0) {
    ring->tail = (ring->tail + 1) % ring->capacity;
  }
  return ts;
}

static inline uint32_t ring_peek_tail(Ring *ring) {
  assert(ring->count > 0);
  return ring->timestamps[ring->tail];
}

// k = 1/eps
uint64_t wnd_bit_count_apx_new(StateApx *self, uint32_t wnd_size, uint32_t k) {
  assert(wnd_size >= 1);
  assert(k >= 1);

  self->wnd_size = wnd_size;
  self->k = k;
  self->r = k;
  self->timestamp = 0;
  self->total_sum = 0;
  self->highest_power = -1;

  // number of power levels needed
  uint32_t max_p = 0;
  {
    uint32_t v = wnd_size;
    while (v > 1) {
      v >>= 1;
      max_p++;
    }
  }
  self->num_power_levels = max_p + 3; // safety margin
  if (self->num_power_levels > MAX_POWER_LEVELS) {
    self->num_power_levels = MAX_POWER_LEVELS;
  }

  // Each ring has capacity r+2
  uint32_t ring_cap = self->r + 2;
  uint64_t total_slots = (uint64_t)ring_cap * self->num_power_levels;
  uint64_t memory = total_slots * sizeof(uint32_t);

  self->pool = (uint32_t *)malloc(memory);
  assert(self->pool != NULL);

  for (uint32_t p = 0; p < self->num_power_levels; p++) {
    ring_init(&self->rings[p], self->pool + (uint64_t)p * ring_cap, ring_cap);
  }

  return memory;
}

void wnd_bit_count_apx_destruct(StateApx *self) { free(self->pool); }

void wnd_bit_count_apx_print(StateApx *self) {
  printf("DGIM State: timestamp=%u, r=%u, total_sum=%u, highest_power=%d\n",
         self->timestamp, self->r, self->total_sum, self->highest_power);
  for (uint32_t p = 0; p < self->num_power_levels; p++) {
    if (self->rings[p].count > 0) {
      printf("  Power %u (%u buckets): ", p, self->rings[p].count);
      Ring *ring = &self->rings[p];
      uint32_t idx = ring->tail;
      for (uint32_t i = 0; i < ring->count; i++) {
        printf("t=%u ", ring->timestamps[idx]);
        idx = (idx + 1) % ring->capacity;
      }
      printf("\n");
    }
  }
}

uint32_t wnd_bit_count_apx_next(StateApx *self, bool item) {
  self->timestamp++;

  // 1. Remove expired buckets.
  //    The oldest bucket is the tail of the highest occupied power level.
  while (self->highest_power >= 0) {
    Ring *ring = &self->rings[self->highest_power];
    if (ring->count == 0) {
      self->highest_power--;
      continue;
    }
    uint32_t age = self->timestamp - ring_peek_tail(ring);
    if (age >= self->wnd_size) {
      uint32_t sz = (1u << self->highest_power);
      self->total_sum -= sz;
      ring_pop_tail(ring);
      if (ring->count == 0) {
        self->highest_power--;
      }
    } else {
      break;
    }
  }

  // 2. If item is 1, insert new bucket at power 0 and cascade merges
  if (item) {
    ring_push(&self->rings[0], self->timestamp);
    self->total_sum += 1;
    if (self->highest_power < 0)
      self->highest_power = 0;

    // Merge cascade: while power p has r+2 entries, merge oldest two
    for (uint32_t p = 0; p < self->num_power_levels - 1; p++) {
      Ring *ring = &self->rings[p];
      if (ring->count <= self->r + 1) {
        break; // no merge needed
      }

      // Pop two oldest from power p, push one merged onto power p+1
      N_MERGES++;
      ring_pop_tail(ring);               // discard oldest
      uint32_t ts = ring_pop_tail(ring); // keep 2nd oldest's timestamp
      // total_sum unchanged: removed 2*2^p, will add 2^(p+1) = 2*2^p

      // Push merged bucket onto power p+1
      ring_push(&self->rings[p + 1], ts);
      if ((int32_t)(p + 1) > self->highest_power) {
        self->highest_power = p + 1;
      }
    }
  }

  // 3. Compute estimate
  if (self->highest_power < 0) {
    return 0;
  }

  // Find the oldest bucket (tail of highest occupied power level)
  // Update highest_power if needed
  while (self->highest_power >= 0 &&
         self->rings[self->highest_power].count == 0) {
    self->highest_power--;
  }
  if (self->highest_power < 0) {
    return 0;
  }

  uint32_t oldest_size = (1u << self->highest_power);
  uint32_t estimate = self->total_sum - oldest_size + 1;

  return estimate;
}

#endif // _WINDOW_BIT_COUNT_APX_

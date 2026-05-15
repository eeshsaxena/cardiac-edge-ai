/**
 * signal_buffer.h
 * ───────────────
 * Generic lock-free single-producer / single-consumer circular buffer.
 * Used for passing ECG windows from ISR to main loop without dynamic alloc.
 */
#pragma once
#include <Arduino.h>

template<typename T, int N>
struct RingBuf {
  volatile T   data[N];
  volatile int head = 0;   // write index (ISR)
  volatile int tail = 0;   // read  index (main)

  // Returns false if full (drops sample)
  bool push(T val) {
    int next = (head + 1) % N;
    if (next == tail) return false;  // full
    data[head] = val;
    head = next;
    return true;
  }

  bool pop(T& out) {
    if (tail == head) return false;  // empty
    out  = data[tail];
    tail = (tail + 1) % N;
    return true;
  }

  int available() const {
    return (head - tail + N) % N;
  }

  bool empty() const { return head == tail; }
  bool full()  const { return ((head + 1) % N) == tail; }
};

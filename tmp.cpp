#include "mpsc_ringbuf.hpp"
#include "spsc_ringbuf.hpp"

int main() {
    mpsc_ringbuf<void*, 16> q;
    q.create_ring_buf();
    q.destroy_ring_buf();
    spsc_ringbuf<void*, 16> q2;
    q2.create_ring_buf();
    q2.destroy_ring_buf();
    return 0;
}
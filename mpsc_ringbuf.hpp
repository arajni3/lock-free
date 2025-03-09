#pragma once

#include <atomic>
#include <type_traits>
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <type_traits>
#include <cstring>

#define MPSC_ALIGNMENT 64

typedef uint16_t size_type; // allows holding up to 64k simultaneous entries

// A lock-free (still wait-free for the consumer!) ring buffer, optimized for the consumer using one-sided caching and lazy updates.
template<typename T, size_type capacity>
struct mpsc_ringbuf {
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
    static_assert(!(capacity & (capacity - 1)), "capacity must be a power of 2");

    /* The underlying queue slots are not aligned to avoid false sharing to save memory and because the queue implementation is 
    optimized for burst produces and consumes; as long as those bursts are aligned, then the queue slots do not need to be aligned.
    */
    T* q = nullptr;

    /* bit array of packed words to minimize atomics and provide SIMD at the cost of sharing, where set bits 
    in the unconsumed bitset denote valid, unconsumed entries and the reserved bitset holds reserved entries amongst producers.
    */
    struct alignas(MPSC_ALIGNMENT) {
        std::atomic<uint8_t> __bits;
    } *unconsumed_bitset = nullptr, *reserved_bitset = nullptr;

    alignas(MPSC_ALIGNMENT) std::atomic<size_type> global_size{0}; // global published size
    alignas(MPSC_ALIGNMENT) std::atomic<size_type> head{0};

    struct alignas(MPSC_ALIGNMENT) {
        /* Optimization: Avoid atomics for the consumer via double caching and lazy updates.
        When size_cons > 0, we know there are >= size_cons elements in the queue and so does the producer, so they will not be mutated by 
        the producer, and we can avoid reloading the global size by not reloading until size_cons == 0. At that point, we reload the global size 
        (which was being incremented by the producer); this size is actually > the actual size, which is global size - size_cons_since_last_correction. 
        The latter variable is always set on each correction of the global size to the corrected value. So this avoids the extra 
        atomic when the queue is nonempty from the consumer's point of view.

        Thus, the consumer updates its state globally in two ways: primarily by updating the actual head, and secondarily 
        by correcting (reducing) the global size. By lazily doing the latter, the consumer significantly saves on atomic instructions. The 
        global size is needed only by the consumer since that is the only source of global producer updates, and the producer gets all the 
        updates from the state via the head updates. 
        */
        size_type size_cons = 0, size_cons_since_last_correction = 0;

        size_type head_cons = 0; // locally consumer-cached value of the head, modified only by the consumer hence always == atomic head
    };

    /* Unlike the single-producer case, producers cannot cache locally as they must share a common state.
    */
    alignas(MPSC_ALIGNMENT) std::atomic<size_type> size_prod{0};
    
    void create_ring_buf() {
        // the underlying array should be aligned at the beginning to avoid false sharing as well as to enable certain SIMD vectorizations
        q = (T*)std::aligned_alloc(MPSC_ALIGNMENT, sizeof(T) * capacity);
        if (q) {
            if constexpr (std::is_class_v<T>) {
                for (size_type i = 0; i < capacity; ++i) { new (&q[i]) T(); }
            }
        }
        unconsumed_bitset = (std::remove_reference_t<decltype(*unconsumed_bitset)>*)std::aligned_alloc(MPSC_ALIGNMENT, (8 + capacity - 1) >> 3);
        reserved_bitset = (std::remove_reference_t<decltype(*reserved_bitset)>*)std::aligned_alloc(MPSC_ALIGNMENT, (8 + capacity - 1) >> 3);
        for (size_type i = 0; i < (8 + capacity - 1) >> 3; ++i) {
            unconsumed_bitset[i].__bits.store(0, std::memory_order_relaxed);
            reserved_bitset[i].__bits.store(0, std::memory_order_relaxed);
        }
        std::atomic_thread_fence(std::memory_order_release);
    }

    void destroy_ring_buf() {
        if constexpr (std::is_class_v<T>) {
            for (size_type i = 0; i < capacity; ++i) { (&q[i])->~T(); }
        }
        std::free(q);
        std::free(unconsumed_bitset);
        std::free(reserved_bitset);
    }

    size_type produce(T& new_data) noexcept { return produce_burst(&new_data, 1); }
    bool consume(T& ret_data) noexcept { return consume_burst(&ret_data, 1); }

    size_type produce_burst(T* new_data, size_type size) noexcept {
        size_type num_produced = 0;
        do {
            size_type prod_size, head_val, start_tail;

            /* To keep the producers lock-free, each producer loads the producer size only once and goes across the ring buffer 
            themselves to find entries instead of reloading the producer size and relying on other producers to update the latter value.
            However, the producer size generally lags behind consumer updates, so update it safely in a separate CAS.
            */
            prod_size = size_prod.load(std::memory_order_relaxed);
            if (prod_size == capacity) { 
                size_type global_val, new_prod_size;
                do {
                    new_prod_size = size_prod.load(std::memory_order_relaxed);
                    global_val = global_size.load(std::memory_order_relaxed);
                } while (!size_prod.compare_exchange_weak(new_prod_size, global_val, std::memory_order_relaxed, std::memory_order_relaxed));
                prod_size = new_prod_size;
                if (prod_size == capacity) { break; } // could not find free space
            }

            uint8_t idx, bits, n_set, reserved_set;
            size_type entries_crossed = 0;
            // load head only once
            head_val = head.load(std::memory_order_relaxed);
            /* first reserve space in competition against other producers by CAS-ing unreserved bits to reserved wherever those 
            entries are also unconsumed
            */
            do {
                start_tail = (prod_size - head_val) & (capacity - 1);

                idx = start_tail >> 3;
                uint8_t off = start_tail & 7;
                size_type j1, j2;
                j1 = 7 - off;
                j2 = j1 - std::min(size_type(start_tail + size - num_produced), j1);
                // get the bits in [j2, j1]
                uint8_t mask = ((1 << j1) - (1 << j2)) | (1 << j1);
                bits = reserved_bitset[idx].load(std::memory_order_relaxed);

                /* count the leading unreserved (zero) bits in [j2, j1] by counting the leading zeroes in the 
                bitwise NOT of the NOT-ed mask application
                */
                uint8_t res = (bits & ~mask) << off, rev_mask;
                rev_mask = ~res;
                n_set = rev_mask ? __builtin_clz(rev_mask) : 8;

                /* In this case, there are no free entries in this range. Move to the right of it.
                */
                if (n_set == 0) { 
                    uint8_t n_move = j1 - j2 + 1;
                    entries_crossed += n_move;
                    prod_size += n_move;
                    continue;
                } 
                reserved_set = ((1 << j1) - (1 << (j1 - n_set - 1))) | (1 << j1);      

                // Check that these entries are consumed (unset).
                uint8_t uncons_bits = unconsumed_bitset[idx].load(std::memory_order_relaxed);
                /* If not all these entries are consumed, then, since this is a FIFO queue and the producers never lag behind the consumer 
                in unconsumed entries, it means that some of the leading entries were unconsumed (set).
                */
                uint8_t uncons_unresv_bits = uncons_bits & reserved_set;
                res = uncons_unresv_bits << off;
                rev_mask = ~res;
                // Avoid these leading entries.
                uint8_t n_set_avoid = rev_mask ? __builtin_clz(rev_mask) : 8;
                n_set -= n_set_avoid;

                /* In this case, there are no free entries in this range. Move to the right of it.
                */
                if (n_set == 0) { 
                    uint8_t n_move = j1 - j2 + 1;
                    entries_crossed += n_move;
                    prod_size += n_move;
                    continue;
                }

                prod_size += n_set_avoid;
                // Recompute the tail to avoid these leading entries in case of success below.
                start_tail = (start_tail + n_set_avoid) & (capacity - 1);
                mask = ((1 << j1) - (1 << (j1 - n_set_avoid - 1))) | (1 << j1);
                reserved_set &= ~mask;

            } while (entries_crossed < capacity && 
                !reserved_bitset[idx].compare_exchange_weak(bits, bits | reserved_set, std::memory_order_relaxed, std::memory_order_relaxed));
    
            if (n_set) {
                std::atomic_thread_fence(std::memory_order_acquire);
                size_prod.fetch_add(n_set, std::memory_order_relaxed);
                std::memcpy(q + start_tail, new_data + num_produced, sizeof(T) * n_set);
                num_produced += n_set;
                // Now mark these entries as unconsumed (equivalent to a bitwise OR since these bits were unset).
                unconsumed_bitset[idx].fetch_add(reserved_set, std::memory_order_release);
            }
        } while (num_produced < size);

        if (num_produced) { global_size.fetch_add(num_produced, std::memory_order_relaxed); } // store fences provided above

        return num_produced;
    }

    bool consume_burst(T* ret_data, size_type size) {
        size_type num_consumed = 0;
        if (size_cons) {
            num_consumed = std::min(size_cons, size);
            const size_type unwrapped_head = head_cons;
            const size_type old_num_end = std::min(num_consumed, size_type(capacity - unwrapped_head));
            /* Check the prefix of each contiguous segment in the ring buffer to see if the first entries are actually unconsumed because 
            producers may have produced out of order.
            */
            int i = unwrapped_head;
            bool started = false;
            while (i < unwrapped_head + old_num_end) {
                size_type idx = i >> 3, off = i & 7;
                if (started && off) { break; }
                started = true; 
                size_type j1, j2;
                j1 = 7 - off;
                j2 = j1 - std::min(size_type((unwrapped_head + old_num_end - 1) - i), j1); // j1 and j2 are inclusive, and j2 ends at the right

                // get the bits in [j2, j1]
                uint8_t mask = ((1 << j1) - (1 << j2)) | (1 << j1);
                uint8_t bits = unconsumed_bitset[i].__bits.load(std::memory_order_relaxed);
                /* count the leading set bits in [j2, j1] by counting the leading zeroes in the 
                bitwise NOT of the mask application
                */
                uint8_t res = (bits & mask) << off;
                uint8_t rev_mask = ~res;
                uint8_t n_set = rev_mask ? __builtin_clz(rev_mask) : 8;

                if (n_set == 0) { break; }
                i += n_set;

                // consume these entries before marking them as consumed
                std::atomic_thread_fence(std::memory_order_acquire);
                std::memcpy(ret_data + i - n_set, q + i - n_set, sizeof(T) * n_set);

                // mark all these unconsumed entries as consumed via a subtraction (equivalent to a NOT since they were all set)
                uint8_t consumed_bits = (((1 << j1) - (1 << (j1 - n_set - 1))) | (1 << j1)) << off;
                unconsumed_bitset[i].__bits.fetch_sub(consumed_bits, std::memory_order_release);
            }
            size_type suffix_consumed = i - unwrapped_head;
            if (suffix_consumed == old_num_end) // process prefix
            {
                i = 0;
                started = false;
                while (i < num_consumed - old_num_end) {
                    size_type idx = i >> 3, off = i & 7;
                    if (started && off) { break; }
                    started = true; 
                    size_type j1, j2;
                    j1 = 7 - off;
                    j2 = j1 - std::min(size_type((num_consumed - old_num_end - 1) - i), j1); // j1 and j2 are inclusive, and j2 ends at the right
    
                    // get the bits in [j2, j1]
                    uint8_t mask = ((1 << j1) - (1 << j2)) | (1 << j1);
                    uint8_t bits = unconsumed_bitset[i].__bits.load(std::memory_order_relaxed);
                    /* count the leading set bits in [j2, j1] by counting the leading zeroes in the 
                    bitwise NOT of the mask application
                    */
                    uint8_t res = (bits & mask) << off;
                    uint8_t rev_mask = ~res;
                    uint8_t n_set = rev_mask ? __builtin_clz(rev_mask) : 8;
    
                    if (n_set == 0) { break; }
                    i += n_set;

                    // consume these entries before marking them as consumed
                    std::atomic_thread_fence(std::memory_order_acquire);
                    std::memcpy(ret_data + + num_consumed + i - n_set, q + num_consumed + i - n_set, sizeof(T) * n_set);

                    // mark all these unconsumed entries as consumed via a subtraction (equivalent to a NOT since they were all set)
                    uint8_t consumed_bits = (((1 << j1) - (1 << (j1 - n_set - 1))) | (1 << j1)) << off;
                    unconsumed_bitset[i].__bits.fetch_sub(consumed_bits, std::memory_order_release);
                }
                uint8_t prefix_consumed = i;

                num_consumed = suffix_consumed + prefix_consumed;
            } else // the whole suffix was definitely not consumed, don't check the start of the ring buffer array
            {
                num_consumed = suffix_consumed;
                if (num_consumed == 0) { return false; }
            }

            size_cons -= num_consumed;

            head_cons = (head_cons + num_consumed) & (capacity - 1);
            head.store(head_cons, std::memory_order_release);
        } else {
            size_type correct_size = global_size.load(std::memory_order_relaxed);

            /* Even if the global size increased after the above load, the cached number by which to correct will be sufficiently correct 
            because it will be either 0 (no correction) or positive (that many elements to consume and hence by which to reduce the global 
            size).
            */

            // correct by reducing by how many elements were consumed since the last correction, but only if there were any consumed
            if (size_cons_since_last_correction) {
                // trivially synchronizes with the load above
                global_size.fetch_sub(size_cons_since_last_correction, std::memory_order_relaxed);
            }
            correct_size -= size_cons_since_last_correction;
            size_cons = size_cons_since_last_correction = correct_size;

            if (size_cons) // copy fast path logic to avoid redundant branch prediction
            { 
                num_consumed = std::min(size_cons, size);
                const size_type unwrapped_head = head_cons;
                const size_type old_num_end = std::min(num_consumed, size_type(capacity - unwrapped_head));
                /* Check the prefix of each contiguous segment in the ring buffer to see if the first entries are actually unconsumed because 
                producers may have produced out of order.
                */
                int i = unwrapped_head;
                bool started = false;
                while (i < unwrapped_head + old_num_end) {
                    size_type idx = i >> 3, off = i & 7;
                    if (started && off) { break; }
                    started = true; 
                    size_type j1, j2;
                    j1 = 7 - off;
                    j2 = j1 - std::min(size_type((unwrapped_head + old_num_end - 1) - i), j1); // j1 and j2 are inclusive, and j2 ends at the right
    
                    // get the bits in [j2, j1]
                    uint8_t mask = ((1 << j1) - (1 << j2)) | (1 << j1);
                    uint8_t bits = unconsumed_bitset[i].__bits.load(std::memory_order_relaxed);
                    /* count the leading set bits in [j2, j1] by counting the leading zeroes in the 
                    bitwise NOT of the mask application
                    */
                    uint8_t res = (bits & mask) << off;
                    uint8_t rev_mask = ~res;
                    uint8_t n_set = rev_mask ? __builtin_clz(rev_mask) : 8;
    
                    if (n_set == 0) { break; }
                    i += n_set;

                    // consume these entries before marking them as consumed
                    std::atomic_thread_fence(std::memory_order_acquire);
                    std::memcpy(ret_data + i - n_set, q + i - n_set, sizeof(T) * n_set);

                    // mark all these unconsumed entries as consumed via a subtraction (equivalent to a NOT since they were all set)
                    uint8_t consumed_bits = (((1 << j1) - (1 << (j1 - n_set - 1))) | (1 << j1)) << off;
                    unconsumed_bitset[i].__bits.fetch_sub(consumed_bits, std::memory_order_release);
                }
                size_type suffix_consumed = i - unwrapped_head;
                if (suffix_consumed == old_num_end) // process prefix
                {
                    i = 0;
                    started = false;
                    while (i < num_consumed - old_num_end) {
                        size_type idx = i >> 3, off = i & 7;
                        if (started && off) { break; }
                        started = true; 
                        size_type j1, j2;
                        j1 = 7 - off;
                        j2 = j1 - std::min(size_type((num_consumed - old_num_end - 1) - i), j1); // j1 and j2 are inclusive, and j2 ends at the right
        
                        // get the bits in [j2, j1]
                        uint8_t mask = ((1 << j1) - (1 << j2)) | (1 << j1);
                        uint8_t bits = unconsumed_bitset[i].__bits.load(std::memory_order_relaxed);
                        /* count the leading set bits in [j2, j1] by counting the leading zeroes in the 
                        bitwise NOT of the mask application
                        */
                        uint8_t res = (bits & mask) << off;
                        uint8_t rev_mask = ~res;
                        uint8_t n_set = rev_mask ? __builtin_clz(rev_mask) : 8;
        
                        if (n_set == 0) { break; }
                        i += n_set;

                        // consume these entries before marking them as consumed
                        std::atomic_thread_fence(std::memory_order_acquire);
                        std::memcpy(ret_data + + num_consumed + i - n_set, q + num_consumed + i - n_set, sizeof(T) * n_set);

                        // mark all these unconsumed entries as consumed via a subtraction (equivalent to a NOT since they were all set)
                        uint8_t consumed_bits = (((1 << j1) - (1 << (j1 - n_set - 1))) | (1 << j1)) << off;
                        unconsumed_bitset[i].__bits.fetch_sub(consumed_bits, std::memory_order_release);
                    }
                    uint8_t prefix_consumed = i;
    
                    num_consumed = suffix_consumed + prefix_consumed;
                } else // the whole suffix was definitely not consumed, don't check the start of the ring buffer array
                {
                    num_consumed = suffix_consumed;
                    if (num_consumed == 0) { return false; }
                }

                size_cons -= num_consumed;

                head_cons = (head_cons + num_consumed) & (capacity - 1);
                head.store(head_cons, std::memory_order_release);
             }
        }
        return num_consumed;
    }
};
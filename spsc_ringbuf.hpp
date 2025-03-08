#pragma once

#include <atomic>
#include <type_traits>
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <type_traits>
#include <cstring>

#define ALIGNMENT 64

typedef uint16_t size_type; // allows holding up to 64k simultaneous entries

// A wait-free ring buffer, optimized using local caching and lazy updates.
template<typename T, size_type capacity>
struct spsc_ringbuf {
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
    static_assert(!(capacity & (capacity - 1)), "capacity must be a power of 2");

    /* The underlying queue slots are not aligned to avoid false sharing to save memory and because the queue implementation is 
    optimized for burst produces and consumes; as long as those bursts are aligned, then the queue slots do not need to be aligned.
    */
    T* q = nullptr;

    void create_ring_buf() {
        // the underlying array should be aligned at the beginning to avoid false sharing as well as to enable certain SIMD vectorizations
        q = (T*)std::aligned_alloc(ALIGNMENT, sizeof(T) * capacity);
        if (q) {
            if constexpr (std::is_class_v<T>) {
                for (size_type i = 0; i < capacity; ++i) { new (&q[i]) T(); }
            }
        }
    }

    void destroy_ring_buf() {
        if constexpr (std::is_class_v<T>) {
            for (size_type i = 0; i < capacity; ++i) { (&q[i])->~T(); }
        }
        std::free(q);
    }

    alignas(ALIGNMENT) std::atomic<size_type> global_size{0}; // global published size
    alignas(ALIGNMENT) std::atomic<size_type> head{0};
    struct alignas(ALIGNMENT) {
        /* Optimization: Avoid atomics for the producer via double caching.
        When size_prod < capacity - 1, we know there are at least capacity - size_prod free slots in the queue, which will not be garbage-
        read by the consumer (the consumer at worst lags behind recent pushes, never ahead). The atomic head is then read, which will result 
        in a correction (reduction) of the number of unconsumed elements (size_prod) based on the difference between the actual head and 
        the previously producer-side cached head (head_prod).

        If size_prod is apparently >= capacity - 1, then we need to load the head anyway to take the chance to correct size_prod, and then 
        we check again if there are empty slots; if so, then we do the exact same enqueue action except we don't reread the head and re-correct 
        size_prod. This means that the producer only ever needs to read one atomic, the global head. It never needs to read the global size, 
        only update it upon success. Hence, the total number of atomic operations in one produce call is only two in case (acquire load and 
        release store) of success and only one (acquire load) in case of failure; in other words, there is no fast path vs. slow path for the 
        producer, both success paths are equally fast (save for one extra branch prediction).
        */
        size_type size_prod = 0, head_prod = 0;
    };

    struct alignas(ALIGNMENT) {
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

    bool produce(T& new_data) noexcept {
        bool success = false;
        if (size_prod < capacity - 1) {
            q[(head_prod + size_prod) & (capacity - 1)] = new_data;
            ++size_prod;
            /* We need to do an atomic (fetch-)add instead of a faster store because the consumer may have trailed behind the new elements 
            pushed and thus would have been triggered to perform a correction of the global size.
            */

            global_size.fetch_add(1, std::memory_order_release);
            success = true;
        } else {
            size_type head_val = head.load(std::memory_order_acquire);
            // Reduce the cached number of unconsumed elements by the change in head and then set the new producer-side cached head.
            size_prod -= (head_val - head_prod) & (capacity - 1);
            head_prod = head_val;

            if (size_prod < capacity - 1) {
                q[(head_val + size_prod) & (capacity - 1)] = new_data;
                ++size_prod;
                /* We need to do an atomic (fetch-)add instead of a faster store because the consumer may have trailed behind the new elements 
                pushed and thus would have been triggered to perform a correction of the global size.
                */

                global_size.fetch_add(1, std::memory_order_release);
                success = true;
            }
        }
        return success;
    }

    /* Attempts to produce up to size elements of the new data and returns how many were successfully produced.
    */
    size_type produce_burst(T* new_data, size_type size) noexcept {
        size_type num_produced = 0, num_producible = capacity - size_prod;
        if (num_producible) {
            // Recalculate the number of empty slots after correction and thereby calculate the actual number of elements to produce.
            num_producible = capacity - size_prod;
            num_produced = std::min(size, num_producible);

            // Exploit vectorization by splitting the end of the ring buffer and the beginning into two separate cases.
            const size_type unwrapped_tail = head_prod + size_prod;
            size_prod += num_produced;
            const size_type num_end = unwrapped_tail < capacity ? std::min(num_produced, size_type(capacity - unwrapped_tail)) : 0;
            std::memcpy(q + unwrapped_tail, new_data, sizeof(T) * num_end);
            std::memcpy(q, new_data + num_end, sizeof(T) * (num_produced - num_end));

            /* We need to do an atomic (fetch-)add instead of a faster store because the consumer may have trailed behind the new elements 
            pushed and thus would have been triggered to perform a correction of the global size.
            */

            global_size.fetch_add(num_produced, std::memory_order_release);
        } else {
            size_type head_val = head.load(std::memory_order_acquire);
            // Reduce the cached number of unconsumed elements by the change in head and then set the new producer-side cached head.
            size_prod -= (size_type)(head_val - head_prod) & (capacity - 1);
            head_prod = head_val;
            // Recalculate the number of empty slots after correction and thereby calculate the actual number of elements to produce.
            num_producible = capacity - size_prod;

            if (num_producible) {
                num_produced = std::min(size, num_producible);

                // Exploit vectorization by splitting the end of the ring buffer and the beginning into two separate cases.
                const size_type unwrapped_tail = head_val + size_prod;
                size_prod += num_produced;
                const size_type num_end = unwrapped_tail < capacity ? std::min(num_produced, size_type(capacity - unwrapped_tail)) : 0;
                std::memcpy(q + unwrapped_tail, new_data, sizeof(T) * num_end);
                std::memcpy(q, new_data + num_end, sizeof(T) * (num_produced - num_end));

                /* We need to do an atomic (fetch-)add instead of a faster store because the consumer may have trailed behind the new elements 
                pushed and thus would have been triggered to perform a correction of the global size.
                */

                global_size.fetch_add(num_produced, std::memory_order_release);
            }
        }
        return num_produced;
    }

    bool consume(T& ret_data) noexcept {
        bool success = false;
        if (size_cons) {
            --size_cons;
            ret_data = q[head_cons];
            head_cons = (head_cons + 1) & (capacity - 1);
            head.store(head_cons, std::memory_order_release);
            success = true;
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
                --size_cons;
                ret_data = q[head_cons];
                head_cons = (head_cons + 1) & (capacity - 1);
                head.store(head_cons, std::memory_order_release);
                success = true;
             }
        }
        return success;
    }

    /* Attempts to consume up to size elements from the queue and returns how many were successfully consumed.
    */
    size_type consume_burst(T* ret_data, size_type size) noexcept {
        size_type num_consumed = 0;
        if (size_cons) {
            num_consumed = std::min(size_cons, size);
            size_cons -= num_consumed;

            // Exploit vectorization by splitting the end of the ring buffer and the beginning into two separate cases.
            const size_type unwrapped_head = head_cons;
            head_cons = (head_cons + num_consumed) & (capacity - 1);
            const size_type num_end = std::min(num_consumed, size_type(capacity - unwrapped_head));
            std::memcpy(ret_data, q + unwrapped_head, sizeof(T) * num_end);
            std::memcpy(ret_data + num_end, q, sizeof(T) * (num_consumed - num_end));

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
                size_cons -= num_consumed;

                // Exploit vectorization by splitting the end of the ring buffer and the beginning into two separate cases.
                const size_type unwrapped_head = head_cons;
                head_cons = (head_cons + num_consumed) & (capacity - 1);
                const size_type num_end = std::min(num_consumed, size_type(capacity - unwrapped_head));
                std::memcpy(ret_data, q + unwrapped_head, sizeof(T) * num_end);
                std::memcpy(ret_data + num_end, q, sizeof(T) * (num_consumed - num_end));

                head.store(head_cons, std::memory_order_release);
             }
        }
        return num_consumed;
    }
};
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

API size_t decay_and_count(
    float *values,
    size_t count,
    float decay,
    float threshold
) {
    if (values == NULL) {
        return 0;
    }

    size_t active = 0;

    for (size_t i = 0; i < count; ++i) {
        const float value = values[i] * decay;
        values[i] = value;

        if (value > threshold) {
            ++active;
        }
    }

    return active;
}


API void edge_activity(
    const float *values,
    size_t value_count,
    const int32_t *edges,
    size_t edge_count,
    float *output
) {
    if (values == NULL || edges == NULL || output == NULL) {
        return;
    }

    for (size_t i = 0; i < edge_count; ++i) {
        const size_t edge_offset = i * 2;

        const int32_t source_index = edges[edge_offset];
        const int32_t destination_index = edges[edge_offset + 1];

        float activity = 0.0f;

        if (
            source_index >= 0 &&
            destination_index >= 0 &&
            (size_t)source_index < value_count &&
            (size_t)destination_index < value_count
        ) {
            const float source_value = values[source_index];
            const float destination_value = values[destination_index];

            activity = source_value > destination_value
                ? source_value
                : destination_value;
        }

        output[edge_offset] = activity;
        output[edge_offset + 1] = activity;
    }
}


API size_t region_activity(
    const float *values,
    const int16_t *regions,
    size_t count,
    size_t region_count,
    float threshold,
    float *sums
) {
    if (values == NULL || regions == NULL || sums == NULL) {
        return 0;
    }

    for (size_t region = 0; region < region_count; ++region) {
        sums[region] = 0.0f;
    }

    size_t active = 0;

    for (size_t index = 0; index < count; ++index) {
        const int16_t region = regions[index];
        const float value = values[index];

        if (region >= 0 && (size_t)region < region_count) {
            sums[(size_t)region] += value;
        }

        if (value > threshold) {
            ++active;
        }
    }

    return active;
}

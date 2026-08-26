#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

API size_t decay_and_count(float *values, size_t count, float decay, float threshold) {
    size_t active = 0;
    for (size_t i = 0; i < count; ++i) {
        float value = values[i] * decay;
        values[i] = value;
        active += value > threshold;
    }
    return active;
}

API void edge_activity(const float *values, const int32_t *edges, size_t edge_count, float *output) {
    for (size_t i = 0; i < edge_count; ++i) {
        float a = values[edges[i * 2]];
        float b = values[edges[i * 2 + 1]];
        float activity = a > b ? a : b;
        output[i * 2] = activity;
        output[i * 2 + 1] = activity;
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
    for (size_t region = 0; region < region_count; ++region) {
        sums[region] = 0.0f;
    }
    size_t active = 0;
    for (size_t index = 0; index < count; ++index) {
        const int16_t region = regions[index];
        const float value = values[index];
        if (region >= 0 && (size_t)region < region_count) {
            sums[region] += value;
        }
        active += value > threshold;
    }
    return active;
}

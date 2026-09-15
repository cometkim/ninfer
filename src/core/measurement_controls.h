#pragma once

#include <cstdlib>
#include <cstring>

namespace ninfer::measurement {

// Process-fixed controls for same-binary attribution. Set before Engine construction so
// graph capture and eager execution use the same route. Production defaults stay enabled.
inline bool pdl_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_BENCH_PDL");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

inline bool decode_fusions_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_BENCH_FUSIONS");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

} // namespace ninfer::measurement

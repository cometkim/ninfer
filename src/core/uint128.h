#pragma once

#include <cstdint>

#ifdef _WIN32
#include <intrin.h>
#endif

namespace ninfer {

#ifndef _WIN32

using Uint128 = unsigned __int128;

#else

// MSVC has no __int128. This is a minimal unsigned 128-bit integer covering
// exactly the saturating cost-model arithmetic the runtime contract and
// context-cost code need (little-endian 64-bit pair, wrap-around semantics on
// overflow identical to the GCC/Clang builtin type).
struct Uint128 {
    std::uint64_t lo = 0;
    std::uint64_t hi = 0;

    constexpr Uint128()                      = default;
    constexpr Uint128(std::uint64_t value) noexcept : lo(value), hi(0) {}

    constexpr explicit operator std::uint64_t() const noexcept { return lo; }
};

[[nodiscard]] inline Uint128 operator*(const Uint128& a, const Uint128& b) noexcept {
    Uint128 result;
    result.lo = _umul128(a.lo, b.lo, &result.hi);
    std::uint64_t ignored = 0;
    result.hi += _umul128(a.lo, b.hi, &ignored);
    result.hi += a.hi * b.lo;
    return result;
}

[[nodiscard]] inline Uint128 operator+(const Uint128& a, const Uint128& b) noexcept {
    Uint128 result;
    unsigned char carry = _addcarry_u64(0, a.lo, b.lo, &result.lo);
    _addcarry_u64(carry, a.hi, b.hi, &result.hi);
    return result;
}

[[nodiscard]] inline Uint128 operator-(const Uint128& a, const Uint128& b) noexcept {
    Uint128 result;
    unsigned char borrow = _subborrow_u64(0, a.lo, b.lo, &result.lo);
    _subborrow_u64(borrow, a.hi, b.hi, &result.hi);
    return result;
}

[[nodiscard]] inline Uint128 operator~(const Uint128& value) noexcept {
    Uint128 result;
    result.lo = ~value.lo;
    result.hi = ~value.hi;
    return result;
}

[[nodiscard]] inline Uint128 operator<<(const Uint128& value, unsigned shift) noexcept {
    if (shift == 0) { return value; }
    if (shift >= 128) { return Uint128{}; }
    Uint128 result;
    if (shift >= 64) {
        result.hi = value.lo << (shift - 64);
    } else {
        result.hi = (value.hi << shift) | (value.lo >> (64 - shift));
        result.lo = value.lo << shift;
    }
    return result;
}

[[nodiscard]] inline Uint128 operator>>(const Uint128& value, unsigned shift) noexcept {
    if (shift == 0) { return value; }
    if (shift >= 128) { return Uint128{}; }
    Uint128 result;
    if (shift >= 64) {
        result.lo = value.hi >> (shift - 64);
    } else {
        result.lo = (value.lo >> shift) | (value.hi << (64 - shift));
        result.hi = value.hi >> shift;
    }
    return result;
}

// Long division by a 64-bit divisor; the cost models only ever divide by
// small constants, so the bit-serial loop is fast enough.
[[nodiscard]] inline Uint128 operator/(const Uint128& value, std::uint64_t divisor) noexcept {
    Uint128 quotient{};
    std::uint64_t rem_hi = 0;
    std::uint64_t rem_lo = 0;
    for (int bit = 127; bit >= 0; --bit) {
        const std::uint64_t word       = bit >= 64 ? value.hi : value.lo;
        const unsigned bit_in_word     = static_cast<unsigned>(bit & 63U);
        const std::uint64_t bit_value  = (word >> bit_in_word) & 1ULL;
        const std::uint64_t shifted_lo = (rem_lo << 1) | bit_value;
        rem_hi = (rem_hi << 1) | (rem_lo >> 63);
        rem_lo = shifted_lo;
        // Loop invariant: the remainder stays below 2*divisor <= 2^65, so at
        // most one subtraction per step, and a high word of 1 always clears.
        if (rem_hi != 0 || rem_lo >= divisor) {
            rem_lo = rem_lo - divisor;  // modular: also correct when rem_hi == 1
            rem_hi = 0;
            if (bit >= 64) {
                quotient.hi |= (1ULL << (bit - 64));
            } else {
                quotient.lo |= (1ULL << bit);
            }
        }
    }
    return quotient;
}

[[nodiscard]] constexpr bool operator==(const Uint128& a, const Uint128& b) noexcept {
    return a.hi == b.hi && a.lo == b.lo;
}
[[nodiscard]] constexpr bool operator!=(const Uint128& a, const Uint128& b) noexcept {
    return !(a == b);
}
[[nodiscard]] constexpr bool operator<(const Uint128& a, const Uint128& b) noexcept {
    return a.hi != b.hi ? a.hi < b.hi : a.lo < b.lo;
}
[[nodiscard]] constexpr bool operator>(const Uint128& a, const Uint128& b) noexcept {
    return b < a;
}
[[nodiscard]] constexpr bool operator<=(const Uint128& a, const Uint128& b) noexcept {
    return !(b < a);
}
[[nodiscard]] constexpr bool operator>=(const Uint128& a, const Uint128& b) noexcept {
    return !(a < b);
}

inline Uint128& operator*=(Uint128& a, const Uint128& b) noexcept { return a = a * b; }
inline Uint128& operator+=(Uint128& a, const Uint128& b) noexcept { return a = a + b; }
inline Uint128& operator-=(Uint128& a, const Uint128& b) noexcept { return a = a - b; }

#endif

} // namespace ninfer

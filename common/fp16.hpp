// =============================================================================
//
// Copyright 2021-2022 Enflame. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#pragma once
#include <cstdint>
#include <cstring>
#include <utility>

// Define alignment macro based on compiler type (cannot assume C11 "_Alignas"
// is available)
#if __cplusplus >= 201103L
#define ___ALIGN__(n) alignas(n)  // C++11 kindly gives us a keyword for this
#else                             // !(__cplusplus >= 201103L)
#if defined(__GNUC__)
#define ___ALIGN__(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER)
#define ___ALIGN__(n) __declspec(align(n))
#else
#define ___ALIGN__(n)
#endif  // defined(__GNUC__)
#endif  // __cplusplus >= 201103L

// Macros to allow half & half2 to be used by inline assembly
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short*>(&(var)))
#define __HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short*>(&(var)))
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int*>(&(var)))
#define __HALF2_TO_CUI(var) *(reinterpret_cast<const unsigned int*>(&(var)))

/**
* Types which allow static initialization of "half" and "half2" until
* these become an actual builtin. Note this initialization is as a
* bitfield representation of "half", and not a conversion from short->half.
* Such a representation will be deprecated in a future version of CUDA.
* (Note these are visible to non-nvcc compilers, including C-only compilation)
*/
typedef struct ___ALIGN__(2) { unsigned short x; } __half_raw;

typedef struct ___ALIGN__(4) {
  unsigned short x;
  unsigned short y;
} __half2_raw;

/* Hide GCC member initialization list warnings because of host/device
 * in-function init requirement */
#if defined(__GNUC__)
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Weffc++"
#endif /* __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6) */
#endif /* defined(__GNUC__) */

/* class' : multiple assignment operators specified
   The class has multiple assignment operators of a single type. This warning is
   informational */
#if defined(_MSC_VER) && _MSC_VER >= 1500
#pragma warning(push)
#pragma warning(disable : 4522)
#endif /* defined(__GNUC__) */

class half;
static half __float2half(const float a);
static float __half2float(const half a);

class ___ALIGN__(2) half {
 public:
#if __cplusplus >= 201103L
  half() = default;
#else
  half() {}
#endif /* __cplusplus >= 201103L */

  /* Convert to/from __half_raw */
  half(const __half_raw& hr) : __x(hr.x) {}
  half& operator=(const __half_raw& hr) {
    __x = hr.x;
    return *this;
  }
  volatile half& operator=(const __half_raw& hr) volatile {
    __x = hr.x;
    return *this;
  }
  volatile half& operator=(const volatile __half_raw& hr) volatile {
    __x = hr.x;
    return *this;
  }
  operator __half_raw() const {
    __half_raw ret;
    ret.x = __x;
    return ret;
  }
  operator __half_raw() const volatile {
    __half_raw ret;
    ret.x = __x;
    return ret;
  }

  /* Construct from float/double */
  half(const float f) { __x = __float2half(f).__x; }
  half(const double f) { __x = __float2half(static_cast<float>(f)).__x; }
  half(const int f) { __x = __float2half(f).__x; }
  half(const int64_t f) { __x = __float2half(f).__x; }
  half(const uint32_t f) { __x = __float2half(static_cast<float>(f)).__x; }
  half(const uint64_t f) { __x = __float2half(static_cast<float>(f)).__x; }
  half(const bool f) { __x = __float2half(f).__x; }

  operator float() const { return __half2float(*this); }
  half& operator=(const float f) {
    __x = __float2half(f).__x;
    return *this;
  }

  /* We omit "cast to double" operator, so as to not be ambiguous about up-cast
   */
  half& operator=(const double f) {
    __x = __float2half(static_cast<float>(f)).__x;
    return *this;
  }

  half& operator=(const bool f) {
    __x = __float2half(static_cast<float>(f)).__x;
    return *this;
  }

  half& operator+=(const float f) {
    __x = __float2half(__half2float(*this) + f).__x;
    return *this;
  }
  half& operator-=(const float f) {
    __x = __float2half(__half2float(*this) - f).__x;
    return *this;
  }
  half& operator*=(const float f) {
    __x = __float2half(__half2float(*this) * f).__x;
    return *this;
  }
  half& operator/=(const float f) {
    __x = __float2half(__half2float(*this) / f).__x;
    return *this;
  }

 protected:
  unsigned short __x;
};

/* half2 is visible to non-nvcc host compilers */
class ___ALIGN__(4) half2 {
  half x;
  half y;

  // All construct/copy/assign/move
 public:
#if __cplusplus >= 201103L
  half2() = default;
  half2(half2&& src) { __HALF2_TO_UI(*this) = std::move(__HALF2_TO_CUI(src)); }
  half2& operator=(half2&& src) {
    __HALF2_TO_UI(*this) = std::move(__HALF2_TO_CUI(src));
    return *this;
  }
#else
  half2() {}
#endif /* __cplusplus >= 201103L */
  half2(const half& a, const half& b) : x(a), y(b) {}
  half2(const half2& src) { __HALF2_TO_UI(*this) = __HALF2_TO_CUI(src); }
  half2& operator=(const half2& src) {
    __HALF2_TO_UI(*this) = __HALF2_TO_CUI(src);
    return *this;
  }

  /* Convert to/from __half2_raw */
  half2(const __half2_raw& h2r) { __HALF2_TO_UI(*this) = __HALF2_TO_CUI(h2r); }
  half2& operator=(const __half2_raw& h2r) {
    __HALF2_TO_UI(*this) = __HALF2_TO_CUI(h2r);
    return *this;
  }
  operator __half2_raw() const {
    __half2_raw ret;
    __HALF2_TO_UI(ret) = __HALF2_TO_CUI(*this);
    return ret;
  }
};

/* Restore warning for multiple assignment operators */
#if defined(_MSC_VER) && _MSC_VER >= 1500
#pragma warning(pop)
#endif /* defined(_MSC_VER) && _MSC_VER >= 1500 */

/* Restore -Weffc++ warnings from here on */
#if defined(__GNUC__)
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic pop
#endif /* __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6) */
#endif /* defined(__GNUC__) */

#undef ___ALIGN__

static unsigned short __internal_float2half(const float f, unsigned int& sign,
                                            unsigned int& remainder) {
  unsigned int x;
  unsigned int u;
  unsigned int result = 0U;
  (void)std::memcpy(&x, &f, sizeof(f));
  u = (x & 0x7fffffffU);
  sign = ((x >> 16U) & 0x8000U);
  // NaN/+Inf/-Inf
  if (u >= 0x7f800000U) {
    remainder = 0U;
    result = ((u == 0x7f800000U) ? (sign | 0x7c00U) : 0x7fffU);
  } else if (u > 0x477fefffU) {  // Overflows
    remainder = 0x80000000U;
    result = (sign | 0x7bffU);
  } else if (u >= 0x38800000U) {  // Normal numbers
    remainder = u << 19U;
    u -= 0x38000000U;
    result = (sign | (u >> 13U));
  } else if (u < 0x33000001U) {  // +0/-0
    remainder = u;
    result = sign;
  } else {  // Denormal numbers
    const unsigned int exponent = u >> 23U;
    const unsigned int shift = 0x7eU - exponent;
    unsigned int mantissa = (u & 0x7fffffU);
    mantissa |= 0x800000U;
    remainder = mantissa << (32U - shift);
    result = (sign | (mantissa >> shift));
  }
  return static_cast<unsigned short>(result);
}

static half __float2half(const float a) {
  half val;
  __half_raw r;
  unsigned int sign;
  unsigned int remainder;
  r.x = __internal_float2half(a, sign, remainder);
  if ((remainder > 0x80000000U) ||
      ((remainder == 0x80000000U) && ((r.x & 0x1U) != 0U))) {
    r.x++;
  }
  val = r;
  return val;
}

static float __internal_half2float(const unsigned short h) {
  unsigned int sign = ((static_cast<unsigned int>(h) >> 15U) & 1U);
  unsigned int exponent = ((static_cast<unsigned int>(h) >> 10U) & 0x1fU);
  unsigned int mantissa = ((static_cast<unsigned int>(h) & 0x3ffU) << 13U);
  float f;
  if (exponent == 0x1fU) { /* NaN or Inf */
    sign = ((mantissa != 0U) ? 0U : sign);
    mantissa = ((mantissa != 0U) ? 0x7fffffU : 0U);
    exponent = 0xffU;
  } else if (exponent == 0U) { /* Denorm or Zero */
    if (mantissa != 0U) {
      unsigned int msb;
      exponent = 0x71U;
      do {
        msb = (mantissa & 0x400000U);
        mantissa <<= 1U; /* normalize */
        --exponent;
      } while (msb == 0U);
      mantissa &= 0x7fffffU; /* 1.mantissa is implicit */
    }
  } else {
    exponent += 0x70U;
  }
  unsigned int u = ((sign << 31U) | (exponent << 23U) | mantissa);
  (void)std::memcpy(&f, &u, sizeof(u));
  return f;
}

static float __half2float(const half a) {
  float val;
  val = __internal_half2float(static_cast<__half_raw>(a).x);
  return val;
}

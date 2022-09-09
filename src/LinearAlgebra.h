#pragma once

#ifdef USE_AVX
#include <immintrin.h>
#endif

#include <iostream>

template <typename value_t> constexpr value_t sq(const value_t a) { return a * a; }

template <typename value_t> struct vec3
{
    value_t x, y, z;

    vec3<value_t> operator-(const vec3<value_t> & o) const
    {
        return {x - o.x, y - o.y, z - o.z};
    }

    vec3<value_t> operator+(const vec3<value_t> & o) const
    {
        return {x + o.x, y + o.y, z + o.z};
    }

    vec3<value_t> operator*(const vec3<value_t> & o) const
    {
        return {x * o.x, y * o.y, z * o.z};
    }

    vec3<value_t> operator/(const value_t & o) const { return *this * (1 / o); }

    vec3<value_t> operator*(const value_t & o) const { return {x * o, y * o, z * o}; }

    value_t length() const { return std::sqrt(this->squared_length()); }

    value_t squared_length() const { return sq(x) + sq(y) + sq(z); }

    value_t dot(const vec3<value_t> & o) const { return x * o.x + y * o.y + z * o.z; }

    vec3<value_t> apply_friction(value_t friction) const
    {
        value_t len = length();
        if (len > friction)
        {
            value_t f = (len - friction) / len;
            return {x * f, y * f, z * f};
        }
        else
        {
            return {0, 0, 0};
        }
    }

    vec3<value_t> normalized() const { return *this / length(); }
};

#ifdef USE_AVX

template <>
struct vec3<__m512d> 
{
    __m512d x, y, z;

    vec3<__m512d> operator-(const vec3<__m512d> & o) const
    {

        return {_mm512_sub_pd(x, o.x), _mm512_sub_pd(y, o.y), _mm512_sub_pd(z, o.z)};
    }

    vec3<__m512d> operator+(const vec3<__m512d> & o) const
    {
        return {_mm512_add_pd(x, o.x), _mm512_add_pd(y, o.y), _mm512_add_pd(z, o.z)};
    }

    vec3<__m512d> operator*(const vec3<__m512d> & o) const
    {
        return {_mm512_mul_pd(x, o.x), _mm512_mul_pd(y, o.y), _mm512_mul_pd(z, o.z)};
    }

    vec3<__m512d> operator/(const __m512d & o) const { return *this * (1 / o); }

    vec3<__m512d> operator*(const __m512d & o) const
    {
        return {_mm512_mul_pd(x, o), _mm512_mul_pd(y, o), _mm512_mul_pd(z, o)};
    }

    __m512d dot(const vec3<__m512d> & o) const
    {
        return _mm512_fmadd_pd(x, o.x, _mm512_fmadd_pd(y, o.y, _mm512_mul_pd(z, o.z)));
    }

    __m512d squared_length() const { return dot(*this); }
    __m512d length() const { return _mm512_sqrt_pd(this->squared_length()); }

    vec3<__m512d> apply_friction(__m512d friction) const
    {
        __m512d len = this->length();
        __mmask8 mask{_mm512_cmp_pd_mask(len, friction, _CMP_LT_OQ)};
        auto f{*this * _mm512_div_pd(_mm512_sub_pd(len, friction), len)};

        return {_mm512_mask_blend_pd(mask, _mm512_set1_pd(0.), f.x),
                _mm512_mask_blend_pd(mask, _mm512_set1_pd(0.), f.y),
                _mm512_mask_blend_pd(mask, _mm512_set1_pd(0.), f.z)};
    }

    vec3<__m512d> normalized() const { return *this / this->length(); }
};

vec3<__m512d> blend(const __mmask8 & mask, const vec3<__m512d> & t,
                    const vec3<__m512d> & o)
{
    return {_mm512_mask_blend_pd(mask, t.x, o.x),  //
            _mm512_mask_blend_pd(mask, t.y, o.y),  //
            _mm512_mask_blend_pd(mask, t.z, o.z)};
}

vec3<__m512d> toAVX(const vec3<double> & v)
{
    return {_mm512_set1_pd(v.x), _mm512_set1_pd(v.y), _mm512_set1_pd(v.z)};
}

#endif

std::ostream & operator<<(std::ostream & os, const vec3<double> & v)
{
    return os << "vec3 x: " << v.x << " y: " << v.y << " z: " << v.z;
}

template <class value_t> struct points3d
{
    std::vector<value_t> x;
    std::vector<value_t> y;
    std::vector<value_t> z;

    points3d(size_t size) : x(size), y(size), z(size) {}

    inline vec3<value_t> load(const size_t i) const { return {x[i], y[i], z[i]}; }

    inline void store(const size_t i, const vec3<value_t> & v)
    {
        x[i] = v.x;
        y[i] = v.y;
        z[i] = v.z;
    }

#ifdef USE_AVX
    inline vec3<__m512d> loadavx(const size_t i) const
    {
        return {_mm512_loadu_pd(x.data() + i), //
                _mm512_loadu_pd(y.data() + i), //
                _mm512_loadu_pd(z.data() + i)};
    }

    inline void storeavx(const size_t i, const vec3<__m512d> & v)
    {
        _mm512_storeu_pd(x.data() + i, v.x);
        _mm512_storeu_pd(y.data() + i, v.y);
        _mm512_storeu_pd(z.data() + i, v.z);
    }
#endif

    inline void add_inplace(const size_t i, const vec3<value_t> & v)
    {
        x[i] += v.x;
        y[i] += v.y;
        z[i] += v.z;
    }

    inline void add_inplace(const size_t i, const value_t v)
    {
        x[i] += v;
        y[i] += v;
        z[i] += v;
    }

    inline void sub_inplace(const size_t i, const vec3<value_t> & v)
    {
        x[i] -= v.x;
        y[i] -= v.y;
        z[i] -= v.z;
    }

    inline void sub_inplace(const size_t i, const value_t v)
    {
        x[i] -= v;
        y[i] -= v;
        z[i] -= v;
    }

    inline void mul_inplace(const size_t i, const vec3<value_t> & v)
    {
        x[i] *= v.x;
        y[i] *= v.y;
        z[i] *= v.z;
    }

    inline void mul_inplace(const size_t i, const value_t v)
    {
        x[i] *= v;
        y[i] *= v;
        z[i] *= v;
    }

    inline void div_inplace(const size_t i, const vec3<value_t> & v)
    {
        x[i] /= v.x;
        y[i] /= v.y;
        z[i] /= v.z;
    }

    inline void div_inplace(const size_t i, const value_t v)
    {
        x[i] /= v;
        y[i] /= v;
        z[i] /= v;
    }
};

struct Sphere
{
    const vec3<double> position;
    const double radius;
    const double radius_squared;

    Sphere(const vec3<double> pos, double rad) : position{pos}, radius{rad}, radius_squared{sq(rad)}
    {}
};
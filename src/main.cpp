#include <algorithm>
#include <assert.h>
#include <cmath>
#include <fstream>
#ifdef USE_AVX
#include <immintrin.h>
#endif
#include <iostream>
#include <numbers>
#include <numeric>
#include <omp.h>
#include <vector>

#include "LinearAlgebra.h"
#include "Timer.h"

using Scene = std::vector<Sphere>;

/// ! CONFIGURATION
#define SCENE 4

#define DUMP

#define USE_EULER_FORWARD
// #define USE_TWO_STEP_ADAMS_BASHFORTH
// #define USE_FIVE_STEP_ADAMS_BASHFORTH

#define MULTITHREADING_CLOTH_SINGLE
#define MULTITHREADING_INTEGRATOR
#define MULTITHREADING_GRAVITY
#define MULTITHREADING_AIR_FRICTION
#define MULTITHREADING_COLLISION

#define ENABLE_HORIZONTAL_CONSTRAINTS
#define ENABLE_VERTICAL_CONSTRAINTS
#define ENABLE_M_DIAGONAL_CONSTRAINTS
#define ENABLE_A_DIAGONAL_CONSTRAINTS

#ifdef _WIN32
using s_size_t = long long;
#else
using s_size_t = long;
#endif

constexpr int THREADS{16};

constexpr size_t GRID_DIM{64};
constexpr double CLOTH_DIM{4.0};
constexpr size_t FLUIDITY{4};
constexpr double GROUND_HEIGHT{-1.5};

constexpr size_t STEPS{30000};
#ifdef DUMP
constexpr size_t HYPERSAMPLE_SIZE{200};
#endif

constexpr double DELTA_T{6e-3};
constexpr double SPRING_CONSTANT{32. / GRID_DIM};
constexpr double SPRING_CONSTANT_DIAGONAL{SPRING_CONSTANT / std::numbers::sqrt2};
constexpr double AIR_RESISTANCE{.7};
constexpr double FRICTION{.25};

constexpr double GRAVITY{-981e-2};

constexpr double HORIZONTAL_CONSTRAINT = CLOTH_DIM / (GRID_DIM - 1);
constexpr double DIAGONAL_CONSTRAINT = HORIZONTAL_CONSTRAINT * std::numbers::sqrt2;

/// ! CONSTRAINTS
static_assert(GRID_DIM > 2);
static_assert(STEPS > 2);

/// ! FUNCTIONS
constexpr auto flatten_idx(size_t x, size_t y) { return (x * GRID_DIM + y); }

void init_cloth(points3d<double> & coords)
{
    for (size_t i = 0; i < GRID_DIM; i++)
    {
        const double x = static_cast<double>(-CLOTH_DIM / 2.0) +
                         static_cast<double>(CLOTH_DIM * i) / (GRID_DIM - 1);
        for (size_t j = 0; j < GRID_DIM; j++)
        {
            const double y = static_cast<double>(-CLOTH_DIM / 2.0) +
                             static_cast<double>(CLOTH_DIM * j) / (GRID_DIM - 1);
            coords.x[flatten_idx(i, j)] = x;
            coords.y[flatten_idx(i, j)] = y;
            coords.z[flatten_idx(i, j)] = 1;
        }
    }
}

using Forces = std::vector<std::vector<double>>;

inline void calc_cloth_constraints_force_helper(points3d<double> & coords,     //
                                                const size_t l,                //
                                                Forces & forces_l,             //
                                                Forces & forces_r,             //
                                                const size_t i,                //
                                                const size_t j,                //
                                                const size_t si,               //
                                                const size_t sj,               //
                                                const double constraint,       //
                                                const double weightpi,         //
                                                const double feather_constant, //
                                                const size_t num_nodes)
{
    // load points
    auto pi{coords.load(i)}, pj{coords.load(j)};

    // compute connection and its length THEN normalize connection
    auto connection{pi - pj}; // pj -> pi
    auto connection_length{connection.length()};

    // scale connection with feather forces : \delta l * D
    auto force{(connection_length - constraint) * feather_constant};

    // calculate inverse weight
    auto weightpj{1 - weightpi};

    // apply forces
    for (size_t k = 0; k <= si; k++) forces_l[l][k] += weightpj * force;
    for (size_t k = sj; k < num_nodes; k++) forces_r[l][k] += weightpi * force;
}

inline void calc_cloth_constraints_velocity_helper(points3d<double> & coords,     //
                                                   points3d<double> & velocities, //
                                                   const size_t l,                //
                                                   Forces & forces_l,             //
                                                   Forces & forces_r,             //
                                                   const size_t i,                //
                                                   const size_t j,                //
                                                   const size_t si,               //
                                                   const size_t sj)
{
    // load points
    auto pi{coords.load(i)}, pj{coords.load(j)};

    // compute connection and its length THEN normalize connection
    auto connection{pi - pj}; // pj -> pi
    auto connection_length{connection.length()};
    connection = connection / connection_length;

    velocities.sub_inplace(i, connection * forces_l[l][si]);
    velocities.add_inplace(j, connection * forces_r[l][sj]);
}

inline void set_zero(std::vector<double> & forces) { std::fill(forces.begin(), forces.end(), 0); }

template <                                   //
    size_t (*num_springs_f)(size_t, size_t), //
    size_t (*start_f)(size_t, size_t),       //
    size_t LINE_COUNT,                       //
    size_t STRIDE>                           //
void calc_cloth_contraint_single(            //
    points3d<double> & coords,               //
    points3d<double> & velocities,           //
    Forces & forces_l,                       //
    Forces & forces_r,                       //
    const double CONSTRAINT_DISTANCE,        //
    const double LOCAL_SPRING_CONSTANT,      //
    const size_t & NEIGHBOUR_DISTANCE)
{
#ifdef MULTITHREADING_CLOTH_SINGLE
#pragma omp parallel
#endif
    {
// There are LINE_COUNT lines
#ifdef MULTITHREADING_CLOTH_SINGLE
#pragma omp for
#endif
        for (size_t line = 0; line < LINE_COUNT; ++line)
        {
            // Each line contains num_f(l) springs
            size_t num_springs{num_springs_f(line, NEIGHBOUR_DISTANCE)};

            // Get start and end values
            const size_t num_nodes{num_springs + NEIGHBOUR_DISTANCE};
            const size_t start{start_f(line, NEIGHBOUR_DISTANCE)};
            // const size_t end{start + STRIDE * (num_nodes - 1)};

            // Initialize first and second index
            size_t i{start};
            size_t j{start + NEIGHBOUR_DISTANCE * STRIDE};

            // Reset forces
            set_zero(forces_l[line]);
            set_zero(forces_r[line]);

            // Calculate forces
            for (size_t s = 0; s < num_springs; ++s, i += STRIDE, j += STRIDE)
            {
                const double weight_pi = (s + 1) / double(num_springs + 1);
                calc_cloth_constraints_force_helper(coords,                 //
                                                    line,                   //
                                                    forces_l,               //
                                                    forces_r,               //
                                                    i,                      // index in coords
                                                    j,                      //
                                                    s,                      // index in line
                                                    s + NEIGHBOUR_DISTANCE, //
                                                    CONSTRAINT_DISTANCE,    //
                                                    weight_pi,              //
                                                    LOCAL_SPRING_CONSTANT,  //
                                                    num_nodes);
            }
        }
#ifdef MULTITHREADING_CLOTH_SINGLE
#pragma omp for
#endif
        for (size_t l = 0; l < LINE_COUNT; ++l)
        {
            // Each line contains num_f(l) springs
            size_t num_springs{num_springs_f(l, NEIGHBOUR_DISTANCE)};

            // Initialize first and second index
            size_t i{start_f(l, NEIGHBOUR_DISTANCE)};
            size_t j{i + NEIGHBOUR_DISTANCE * STRIDE};

            // Calculate forces
            for (size_t s = 0; s < num_springs; ++s, i += STRIDE, j += STRIDE)
            {
                calc_cloth_constraints_velocity_helper(coords, velocities, l, forces_l, forces_r, i,
                                                       j, s, s + NEIGHBOUR_DISTANCE);
            }
        }
    }
}

void calc_cloth_constraint_horizontal(points3d<double> & coords, points3d<double> & velocities,
                                      Forces & forces_l, Forces & forces_r, size_t & n)
{
    const auto num_springs_f = [](size_t, size_t n) constexpr->size_t { return GRID_DIM - n; };
    const auto start_f = [](size_t l, size_t) constexpr { return flatten_idx(0, l); };

    calc_cloth_contraint_single<     //
        num_springs_f,               //
        start_f,                     //
        GRID_DIM,                    // LINE_COUNT
        flatten_idx(1, 0)            // STRIDE
        >(coords,                    //
          velocities,                //
          forces_l,                  //
          forces_r,                  //
          HORIZONTAL_CONSTRAINT * n, //
          SPRING_CONSTANT / n, n);
}

void calc_cloth_constraint_vertical(points3d<double> & coords, points3d<double> & velocities,
                                    Forces & forces_l, Forces & forces_r, size_t & n)
{
    const auto num_springs_f = [](size_t, size_t n) constexpr->size_t { return GRID_DIM - n; };
    const auto start_f = [](size_t l, size_t) constexpr { return flatten_idx(l, 0); };

    calc_cloth_contraint_single<     //
        num_springs_f,               //
        start_f,                     //
        GRID_DIM,                    // LINE_COUNT
        flatten_idx(0, 1)            // STRIDE
        >(coords,                    //
          velocities,                //
          forces_l,                  //
          forces_r,                  //
          HORIZONTAL_CONSTRAINT * n, //
          SPRING_CONSTANT / n, n);
}

void calc_cloth_constraint_main_diagonal(points3d<double> & coords, points3d<double> & velocities,
                                         Forces & forces_l, Forces & forces_r, size_t & n)
{
    auto num_springs_f = [](size_t l, size_t n) constexpr->size_t
    {
        s_size_t out = static_cast<s_size_t>(++l);
        if (l > GRID_DIM - 1)
            out = 2 * (GRID_DIM - 1) - out - (n - 1);
        else
            out -= (n - 1);
        if (out < 1)
            return 0;
        else
            return static_cast<size_t>(out);
    };
    auto start_f = [](size_t l, size_t) constexpr->size_t
    {
        // if true : left border else : upper border
        return (++l < GRID_DIM - 1) ? flatten_idx((GRID_DIM - 1) - l, 0)
                                    : flatten_idx(0, l - (GRID_DIM - 1));
    };

    calc_cloth_contraint_single<   //
        num_springs_f,             //
        start_f,                   //
        GRID_DIM * 2 - 3,          // LINE_COUNT
        flatten_idx(1, 1)          // STRIDE
        >(coords,                  //
          velocities,              //
          forces_l,                //
          forces_r,                //
          DIAGONAL_CONSTRAINT * n, //
          SPRING_CONSTANT_DIAGONAL / n, n);
}

void calc_cloth_constraint_anti_diagonal(points3d<double> & coords, points3d<double> & velocities,
                                         Forces & forces_l, Forces & forces_r, size_t & n)
{
    auto num_springs_f = [](size_t l, size_t n) constexpr->size_t
    {
        s_size_t out = static_cast<s_size_t>(++l);
        if (l > GRID_DIM - 1)
            out = 2 * (GRID_DIM - 1) - out - (n - 1);
        else
            out -= (n - 1);
        if (out < 1)
            return 0;
        else
            return static_cast<size_t>(out);
    };
    auto start_f = [](size_t l, size_t) constexpr->size_t
    {
        // if true : upper border else : right border
        return (++l < GRID_DIM - 1) ? flatten_idx(0, l)
                                    : flatten_idx(l - (GRID_DIM - 1), GRID_DIM - 1);
    };

    calc_cloth_contraint_single<   //
        num_springs_f,             //
        start_f,                   //
        GRID_DIM * 2 - 3,          // LINE_COUNT
        flatten_idx(1, -1)         // STRIDE
        >(coords,                  //
          velocities,              //
          forces_l,                //
          forces_r,                //
          DIAGONAL_CONSTRAINT * n, //
          SPRING_CONSTANT_DIAGONAL / n, n);
}

inline void calc_cloth_constraints(points3d<double> & coords, points3d<double> & velocities)
{
    Forces forces_l(GRID_DIM * 2 - 3);
    std::fill(forces_l.begin(), forces_l.end(), std::vector<double>(GRID_DIM));
    Forces forces_r(GRID_DIM * 2 - 3);
    std::fill(forces_r.begin(), forces_r.end(), std::vector<double>(GRID_DIM));

// ! LEFT <-> RIGHT
#ifdef ENABLE_HORIZONTAL_CONSTRAINTS
    for (size_t n = 1; n <= GRID_DIM / FLUIDITY; n *= 2)
    {
        calc_cloth_constraint_horizontal(coords,     //
                                         velocities, //
                                         forces_l,   //
                                         forces_r,   //
                                         n);
    }
#endif

// ! TOP <-> BOTTOM
#ifdef ENABLE_VERTICAL_CONSTRAINTS
    for (size_t n = 1; n <= GRID_DIM / FLUIDITY; n *= 2)
    {
        calc_cloth_constraint_vertical(coords,     //
                                       velocities, //
                                       forces_l,   //
                                       forces_r,   //
                                       n);
    }
#endif

// ! TOP LEFT <-> BOTTOM RIGHT
#ifdef ENABLE_M_DIAGONAL_CONSTRAINTS
    for (size_t n = 1; n <= GRID_DIM / (FLUIDITY * 2); n *= 2)
    {
        calc_cloth_constraint_main_diagonal(coords,     //
                                            velocities, //
                                            forces_l,   //
                                            forces_r,   //
                                            n);
    }
#endif

// ! BOTTOM LEFT <-> TOP RIGHT
#ifdef ENABLE_A_DIAGONAL_CONSTRAINTS
    for (size_t n = 1; n <= GRID_DIM / (FLUIDITY * 2); n *= 2)
    {
        calc_cloth_constraint_anti_diagonal(coords,     //
                                            velocities, //
                                            forces_l,   //
                                            forces_r,   //
                                            n);
    }
#endif
}

inline void calc_cloth_collision(points3d<double> & coords, points3d<double> & velocities,
                                 const Scene & scene)
{
#ifdef MULTITHREADING_COLLISION
#pragma omp parallel for
#endif
#ifdef USE_AVX
    for (size_t i = 0; i < GRID_DIM * GRID_DIM; i += sizeof(__m512d) / sizeof(double))
#else
    for (size_t i = 0; i < GRID_DIM * GRID_DIM; ++i)
#endif
    {
#ifdef USE_AVX
        auto pos{coords.loadavx(i)};
        auto v{velocities.loadavx(i)};
#else
        auto pos{coords.load(i)};
        auto v{velocities.load(i)};
#endif

        for (auto & sphere : scene)
        {
#ifdef USE_AVX
            // load position as set1 all x, y and z
            auto spherepos{toAVX(sphere.position)};
#else
            auto & spherepos{sphere.position};
#endif
            auto connection{pos - spherepos};
            auto length{connection.squared_length()};

#ifdef USE_AVX
            // cmp[i] = length[i] < sphere.radius_squared ? 1 : 0
            __mmask8 cmp{
                _mm512_cmp_pd_mask(length, _mm512_set1_pd(sphere.radius_squared), _CMP_LT_OQ)};

            // connection[i].xyz = connection[i].xyz * length[i]
            connection = connection * _mm512_rsqrt14_pd(length);
            pos = blend(cmp, pos, spherepos + connection * _mm512_set1_pd(sphere.radius * 1.0001));

            auto dot = connection.dot(v);
            connection = connection * dot;
            blend(cmp, v,
                  (v - connection * _mm512_set1_pd(1.2))
                      .apply_friction(
                          _mm512_mul_pd(_mm512_set1_pd((FRICTION / 2)), _mm512_abs_pd(dot))));

#else
            if (length < sphere.radius_squared)
            {
                // reset position to sphere surface
                connection = connection / std::sqrt(length);
                pos = spherepos + connection * (sphere.radius * 1.0001);

                // remove speed central to sphere (fully elastic)
                auto dot = connection.dot(v);
                connection = connection * dot;

                v = (v - connection * 1.2).apply_friction((FRICTION / 2) * std::abs(dot));
            }
#endif
        }

#ifdef USE_AVX
        const auto GROUND_HEIGHT_AVX{_mm512_set1_pd(GROUND_HEIGHT)};
        // pos.z[i] < GROUND_HEIGHT => 1
        __mmask8 cmp{_mm512_cmp_pd_mask(pos.z, GROUND_HEIGHT_AVX, _CMP_LT_OQ)};

        // pos.z[i] = cmp[i] ? GROUND_HEIGHT : pos.z[i]
        pos.z = _mm512_mask_blend_pd(cmp, pos.z, GROUND_HEIGHT_AVX);
        blend(cmp, v,
              v.apply_friction(_mm512_mul_pd(_mm512_set1_pd(FRICTION), _mm512_abs_pd(v.z))));
        v.z = _mm512_mask_mul_pd(v.z, cmp, v.z, _mm512_set1_pd(-1.));
#else
        if (pos.z < GROUND_HEIGHT)
        {
            pos.z = GROUND_HEIGHT;
            v = v.apply_friction(FRICTION * std::abs(v.z));
            v.z *= -1.0;
        }
#endif

#ifdef USE_AVX
        coords.storeavx(i, pos);
        velocities.storeavx(i, v);
#else
        coords.store(i, pos);
        velocities.store(i, v);
#endif
    }
}

void calc_cloth(points3d<double> & coords, std::vector<points3d<double>> & all_velocities,
                const Scene & scene)
{
#ifdef DUMP
    std::ofstream ofile("data.bin", std::ios::binary);

    // Write GRID_DIM into file
    double GRID_DIM_D = double(GRID_DIM);
    ofile.write(reinterpret_cast<char *>(&GRID_DIM_D), sizeof(double));
#endif

#ifdef USE_AVX
    const auto GRAVITY_EFFECT{_mm512_set1_pd(GRAVITY * DELTA_T / 2.0)};
    const auto AIR_RESISTANCE_AVX{_mm512_set1_pd(AIR_RESISTANCE)};
    const auto DELTA_T_AVX{_mm512_set1_pd(DELTA_T)};
#endif

    for (size_t s = 0; s < STEPS; ++s)
    {
        auto velocities_p0 = all_velocities[s % 5];

#if defined(USE_TWO_STEP_ADAMS_BASHFORTH) || defined(USE_FIVE_STEP_ADAMS_BASHFORTH)
        auto & velocities_p1 = all_velocities[(s + 4) % 5];
#endif
#ifdef USE_FIVE_STEP_ADAMS_BASHFORTH
        auto & velocities_p2 = all_velocities[(s + 3) % 5];
        auto & velocities_p3 = all_velocities[(s + 2) % 5];
        auto & velocities_p4 = all_velocities[(s + 1) % 5];
#endif

        /// ! Update Gravity Effects
#ifdef MULTITHREADING_GRAVITY
#pragma omp parallel for
#endif
#ifdef USE_AVX
        for (size_t i = 0; i < GRID_DIM * GRID_DIM; i += 8)
            _mm512_storeu_pd(
                velocities_p0.z.data() + i, //
                _mm512_add_pd(_mm512_loadu_pd(velocities_p0.z.data() + i), GRAVITY_EFFECT));
#else
        for (size_t i = 0; i < GRID_DIM * GRID_DIM; ++i)
            velocities_p0.z[i] += GRAVITY * DELTA_T / 2.0;
#endif

        /// ! Constraints
        calc_cloth_constraints(coords, velocities_p0);

        /// ! Air Resistance
#ifdef MULTITHREADING_AIR_FRICTION
#pragma omp parallel for
#endif
#ifdef USE_AVX
        for (size_t i = 0; i < GRID_DIM * GRID_DIM; i += 8)
        {
            _mm512_storeu_pd(
                velocities_p0.z.data() + i, //
                _mm512_mul_pd(_mm512_loadu_pd(velocities_p0.z.data() + i), AIR_RESISTANCE_AVX));
            _mm512_storeu_pd(
                velocities_p0.y.data() + i, //
                _mm512_mul_pd(_mm512_loadu_pd(velocities_p0.y.data() + i), AIR_RESISTANCE_AVX));
            _mm512_storeu_pd(
                velocities_p0.x.data() + i, //
                _mm512_mul_pd(_mm512_loadu_pd(velocities_p0.x.data() + i), AIR_RESISTANCE_AVX));
        }
#else
        for (size_t i = 0; i < GRID_DIM * GRID_DIM; ++i)
            velocities_p0.mul_inplace(i, AIR_RESISTANCE);
#endif

        /// ! Collision Test
        calc_cloth_collision(coords, velocities_p0, scene);

#ifdef USE_EULER_FORWARD
        /// ! Euler Forward
#ifdef MULTITHREADING_INTEGRATOR
#pragma omp parallel for
#endif
#ifdef USE_AVX
        for (size_t i = 0; i < GRID_DIM * GRID_DIM; i += 8)
        {
            _mm512_storeu_pd(coords.z.data() + i, //
                             _mm512_fmadd_pd(DELTA_T_AVX,
                                             _mm512_loadu_pd(velocities_p0.z.data() + i),
                                             _mm512_loadu_pd(coords.z.data() + i)));
            _mm512_storeu_pd(coords.y.data() + i, //
                             _mm512_fmadd_pd(DELTA_T_AVX,
                                             _mm512_loadu_pd(velocities_p0.y.data() + i),
                                             _mm512_loadu_pd(coords.y.data() + i)));
            _mm512_storeu_pd(coords.x.data() + i, //
                             _mm512_fmadd_pd(DELTA_T_AVX,
                                             _mm512_loadu_pd(velocities_p0.x.data() + i),
                                             _mm512_loadu_pd(coords.x.data() + i)));
        }
#else
        for (size_t i = 0; i < GRID_DIM * GRID_DIM; i += 1)
            coords.add_inplace(i, velocities_p0.load(i) * DELTA_T);
#endif
#endif

#ifdef USE_TWO_STEP_ADAMS_BASHFORTH
        /// ! Two Step Adams-Bashforth Forward
#ifdef MULTITHREADING_INTEGRATOR
#pragma omp parallel for
#endif
        for (size_t i = 0; i < GRID_DIM * GRID_DIM; ++i)
        {
            coords.add_inplace(
                i, //
                all_velocities[s % VELOCITY_HISTORY_SIZE].load(i) * DELTA_T * 1.5 -
                    all_velocities[(VELOCITY_HISTORY_SIZE + s - 1) % VELOCITY_HISTORY_SIZE].load(
                        i) *
                        DELTA_T * 0.5);
        }
#endif

#ifdef USE_FIVE_STEP_ADAMS_BASHFORTH
        /// ! Five Step Adams-Bashforth Forward
#ifdef MULTITHREADING_INTEGRATOR
#pragma omp parallel for
#endif
        for (size_t i = 0; i < GRID_DIM * GRID_DIM; ++i)
        {
            coords.add_inplace(i,                                        //
                               (velocities_p0.load(i) * (1901 / 720.0)   //
                                - velocities_p1.load(i) * (2774 / 720.0) //
                                + velocities_p2.load(i) * (2616 / 720.0) //
                                - velocities_p3.load(i) * (1274 / 720.0) //
                                + velocities_p4.load(i) * (251 / 720.0)) *
                                   DELTA_T);
        }
#endif

        all_velocities[(s + 1) % 5] = velocities_p0;

#ifdef DUMP
        if (s % HYPERSAMPLE_SIZE == 0)
        {
            ofile.write(reinterpret_cast<char *>(coords.x.data()),
                        sizeof(double) * coords.x.size());
            ofile.write(reinterpret_cast<char *>(coords.y.data()),
                        sizeof(double) * coords.y.size());
            ofile.write(reinterpret_cast<char *>(coords.z.data()),
                        sizeof(double) * coords.z.size());
        }
#endif
    }
}

int main()
{
    points3d<double> coordinates(GRID_DIM * GRID_DIM);

#if defined(MULTITHREADING_AIR_FRICTION) || defined(MULTITHREADING_CLOTH_SINGLE) ||                \
    defined(MULTITHREADING_COLLISION) || defined(MULTITHREADING_GRAVITY) ||                        \
    defined(MULTITHREADING_INTEGRATOR)
    omp_set_num_threads(THREADS);
    std::cout << "# Using " << omp_get_max_threads() << " threads.\n";
#else
    std::cout << "# Using " << 1 << " thread.\n";
#endif
    // Initialize velocities
    std::vector<points3d<double>> all_velocities;
    for (size_t s = 0; s < 5; ++s) all_velocities.emplace_back(GRID_DIM * GRID_DIM);

    Scene scene;
#if SCENE == 0
    scene.emplace_back(
        Sphere(vec3<double>{1, -4., 0}.normalized() * 1.2 + vec3<double>{0, 0, .5}, 0.5));
    scene.emplace_back(
        Sphere(vec3<double>{-4, 1, 0}.normalized() * 1.2 + vec3<double>{0, 0, 0}, 0.5));
    scene.emplace_back(
        Sphere(vec3<double>{1, 1., 0}.normalized() * 1.2 + vec3<double>{0, 0, -.5}, 0.5));
#elif SCENE == 1
    scene.emplace_back(Sphere(vec3<double>{0., -.5, 0}, 1));
#elif SCENE == 2
    scene.emplace_back(Sphere(vec3<double>{0., 0, 0}, 1));
#elif SCENE == 3
    scene.emplace_back(Sphere(vec3<double>{-1, -1, 0}, 0.5));
    scene.emplace_back(Sphere(vec3<double>{1, -1, 0}, 0.5));
    scene.emplace_back(Sphere(vec3<double>{-1, 1, 0}, 0.5));
    scene.emplace_back(Sphere(vec3<double>{1, 1, 0}, 0.5));
#elif SCENE == 4
    scene.emplace_back(Sphere(vec3<double>{0., 1, 0.3}, 0.5));
    scene.emplace_back(Sphere(vec3<double>{1, 0., 0.1}, 0.5));
    scene.emplace_back(Sphere(vec3<double>{0., -1, -0.1}, 0.5));
    scene.emplace_back(Sphere(vec3<double>{-1, 0., -0.3}, 0.5));
#endif

    {
        std::ofstream ofile("scene.bin", std::ios::binary);
        for (auto & s : scene)
        {
            ofile.write(reinterpret_cast<char *>(&s), sizeof(double) * 4);
        }
    }

    Timer init_timer("Initialization", std::cout);
    init_cloth(coordinates);
    init_timer.print();

    Timer compute_timer("Computation", std::cout);
    calc_cloth(coordinates, all_velocities, scene);
    compute_timer.print();

    return 0;
}

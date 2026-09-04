#pragma once

/**
 * Exact pairwise 2-opt local search (from RS4CO tsp_ops.hpp).
 * No Fast-2OPT / KNN candidate restriction.
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

namespace tsp {

inline void compute_dist_matrix(
    const float* points, float* dists, const int num_nodes
) {
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            const float dx = points[i * 2] - points[j * 2];
            const float dy = points[i * 2 + 1] - points[j * 2 + 1];
            dists[i * num_nodes + j] = std::sqrt(dx * dx + dy * dy);
        }
    }
}

inline std::tuple<int, int, float> find_best_two_opt(
    const int* tour,
    const float* dist_mat,
    const int num_nodes,
    const int type_2opt
) {
    float best_delta = std::numeric_limits<float>::lowest();
    int best_i = 0;
    int best_j = 0;

    for (int i = 0; i < num_nodes - 1; ++i) {
        for (int j = i + 2; j < num_nodes; ++j) {
            // Skip adjacent wrap-around edge (0, N-1)
            if (i == 0 && j == num_nodes - 1) continue;
            const float delta =
                dist_mat[tour[i] * num_nodes + tour[i + 1]] +
                dist_mat[tour[j] * num_nodes + tour[j + 1]] -
                dist_mat[tour[i] * num_nodes + tour[j]] -
                dist_mat[tour[i + 1] * num_nodes + tour[j + 1]];
            if (delta > best_delta) {
                if (type_2opt == 1 && delta > 0.f) {
                    return {i, j, delta};
                }
                best_delta = delta;
                best_i = i;
                best_j = j;
            }
        }
    }
    return {best_i, best_j, best_delta};
}

inline void apply_two_opt(int* tour, const int i, const int j) {
    std::reverse(tour + i + 1, tour + j + 1);
}

/**
 * Exact pairwise 2-opt.
 * ``type_2opt``: 1 = first improving move, 2 = best improving move.
 */
inline void two_opt(
    int* tour,
    const float* dist_mat,
    const int num_nodes,
    const int num_steps,
    const int type_2opt
) {
    if (num_steps <= 0) return;
    for (int step = 0; step < num_steps; ++step) {
        auto [i, j, delta] =
            find_best_two_opt(tour, dist_mat, num_nodes, type_2opt);
        if (delta < 1e-5f) break;
        apply_two_opt(tour, i, j);
    }
    // Keep closed
    tour[num_nodes] = tour[0];
}

}  // namespace tsp

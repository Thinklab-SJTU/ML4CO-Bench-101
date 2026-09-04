#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

/** Set to 1 to print debug lines to stderr for tracing double free / corruption */
#ifndef TSP_GREEDY_DEBUG
#define TSP_GREEDY_DEBUG 1
#endif
#define TSP_DBG(fmt, ...) \
    do { if (TSP_GREEDY_DEBUG) std::fprintf(stderr, "[tsp_greedy] " fmt "\n", ##__VA_ARGS__); } while (0)

/**
 * @brief Greedy insertion for TSP.
 * Input uses edge_indices (edge (i,j) as i*num_nodes+j); output tour has
 * length num_nodes+1 with tour[num_nodes]=tour[0].
 */
namespace tsp {

inline void greedy_insert(
    int* tour,
    const int* edge_indices,
    const int num_nodes,
    const int num_candidate_edges
) {
    // TSP_DBG("entry: num_nodes=%d num_candidate_edges=%d tour=%p edge_indices=%p",
    //         num_nodes, num_candidate_edges, (void*)tour, (void*)edge_indices);

    std::vector<int> subtour_id;
    subtour_id.resize(num_nodes);
    std::fill(subtour_id.begin(), subtour_id.end(), 0);

    auto update_id = [&subtour_id, &num_nodes](const int from, const int to) {
        for (int i = 0; i < num_nodes; ++i) {
            if (subtour_id[i] == from) {
                subtour_id[i] = to;
            }
        }
    };

    // TSP_DBG("after subtour_id init");

    std::vector<int> neighbors;
    neighbors.resize(2 * num_nodes);
    std::fill(neighbors.begin(), neighbors.end(), -1);
    // TSP_DBG("after neighbors init (size=%zu)", neighbors.size());

    auto set_neighbor = [&neighbors](const int& i, const int& j) {
        if (neighbors[2 * i] == -1) {
            neighbors[2 * i] = j;
        } else {
            assert(neighbors[2 * i + 1] == -1);
            neighbors[2 * i + 1] = j;
        }
        if (neighbors[2 * j] == -1) {
            neighbors[2 * j] = i;
        } else {
            assert(neighbors[2 * j + 1] == -1);
            neighbors[2 * j + 1] = i;
        }
    };

    int next_available_subtour = 1;
    int num_inserted_edges = 0;
    for (int edge_idx = 0; edge_idx < num_candidate_edges; ++edge_idx) {
        const int flat = edge_indices[edge_idx];
        const int i = flat / num_nodes;
        const int j = flat % num_nodes;

        if (neighbors[2 * i + 1] != -1 || neighbors[2 * j + 1] != -1) {
            continue;
        }
        if (i == j) {
            continue;
        }

        if (subtour_id[i] == 0) {
            ++num_inserted_edges;
            if (subtour_id[j] == 0) {
                subtour_id[i] = next_available_subtour;
                subtour_id[j] = next_available_subtour;
                ++next_available_subtour;
            } else {
                subtour_id[i] = subtour_id[j];
            }
            set_neighbor(i, j);
        } else {
            if (subtour_id[j] == 0) {
                ++num_inserted_edges;
                subtour_id[j] = subtour_id[i];
                set_neighbor(i, j);
            } else {
                if (subtour_id[i] == subtour_id[j]) {
                    continue;
                } else {
                    ++num_inserted_edges;
                    update_id(subtour_id[j], subtour_id[i]);
                    set_neighbor(i, j);
                }
            }
        }

        if (num_inserted_edges == num_nodes - 1) {
            break;
        }
    }

    // TSP_DBG("after edge loop: num_inserted_edges=%d (expect %d)", num_inserted_edges, num_nodes - 1);

    // Fallback: if candidates do not form a spanning tree, randomly add from remaining valid edges
    if (num_inserted_edges < num_nodes - 1) {
        std::mt19937 rng(42);
        while (num_inserted_edges < num_nodes - 1) {
            std::vector<std::pair<int, int>> addable;
            for (int i = 0; i < num_nodes; ++i) {
                if (neighbors[2 * i + 1] != -1) continue;
                for (int j = i + 1; j < num_nodes; ++j) {
                    if (neighbors[2 * j + 1] != -1) continue;
                    if (subtour_id[i] != 0 && subtour_id[j] != 0 && subtour_id[i] == subtour_id[j])
                        continue;
                    addable.push_back({i, j});
                }
            }
            if (addable.empty()) break;
            std::shuffle(addable.begin(), addable.end(), rng);
            const int i = addable[0].first;
            const int j = addable[0].second;
            if (subtour_id[i] == 0) {
                ++num_inserted_edges;
                if (subtour_id[j] == 0) {
                    subtour_id[i] = next_available_subtour;
                    subtour_id[j] = next_available_subtour;
                    ++next_available_subtour;
                } else {
                    subtour_id[i] = subtour_id[j];
                }
                set_neighbor(i, j);
            } else {
                if (subtour_id[j] == 0) {
                    ++num_inserted_edges;
                    subtour_id[j] = subtour_id[i];
                    set_neighbor(i, j);
                } else {
                    ++num_inserted_edges;
                    update_id(subtour_id[j], subtour_id[i]);
                    set_neighbor(i, j);
                }
            }
        }
        // TSP_DBG("after fallback: num_inserted_edges=%d", num_inserted_edges);
    }

    assert(num_inserted_edges == num_nodes - 1);

    // neighbors -> tour (find an endpoint: degree 1; with bounds check for safety)
    int start_node = 0;
    while (start_node < num_nodes &&
           (neighbors[2 * start_node] == -1 || neighbors[2 * start_node + 1] != -1)) {
        ++start_node;
    }
    // TSP_DBG("start_node=%d num_nodes=%d", start_node, num_nodes);
    assert(start_node < num_nodes);

    tour[0] = start_node;
    const int start_node_neighbor = neighbors[2 * start_node];
    // TSP_DBG("start_node_neighbor=%d", start_node_neighbor);
    if (neighbors[2 * start_node_neighbor] == start_node) {
        std::swap(neighbors[2 * start_node_neighbor], neighbors[2 * start_node_neighbor + 1]);
    }
    int current_node = start_node;

    std::vector<bool> has_visited(static_cast<size_t>(num_nodes), false);
    // TSP_DBG("before walk loop (num_nodes=%d)", num_nodes);

    for (int i = 1; i < num_nodes; ++i) {
        has_visited[current_node] = true;
        if (!has_visited[neighbors[2 * current_node]]) {
            tour[i] = neighbors[2 * current_node];
            current_node = neighbors[2 * current_node];
        } else {
            assert(!has_visited[neighbors[2 * current_node + 1]]);
            tour[i] = neighbors[2 * current_node + 1];
            current_node = neighbors[2 * current_node + 1];
        }
    }

    // Rotate so that node 0 is first, then close the tour with 0.
    int zero_pos = 0;
    while (zero_pos < num_nodes && tour[zero_pos] != 0) {
        ++zero_pos;
    }
    assert(zero_pos < num_nodes);
    if (zero_pos != 0) {
        std::rotate(tour, tour + zero_pos, tour + num_nodes);
    }
    tour[num_nodes] = 0;
    // TSP_DBG("return (greedy_insert done)");
}

}  // namespace tsp
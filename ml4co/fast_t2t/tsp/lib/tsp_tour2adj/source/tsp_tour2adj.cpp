#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cassert>
#include <algorithm>

#include "parallel.hpp"

namespace py = pybind11;

/**
 * @brief Convert TSP tours (B, N+1) to undirected adjacency (B, N, N).
 *
 * The logic follows tsp_dataset.py:
 *   adj[u, v] = 1 for each consecutive pair u=ref_tour[idx], v=ref_tour[idx+1]
 *   then make it symmetric via adj = adj + adj.T
 *
 * For a Hamiltonian cycle, this is equivalent to setting both adj[u, v] and adj[v, u]
 * for each consecutive pair.
 */
inline py::array_t<float> tsp_tour2adj(
    const py::array_t<int>& tours,
    const int num_workers
) {
    assert(tours.flags() & py::array::c_style);
    assert(tours.ndim() == 2);

    const int batch_size = static_cast<int>(tours.shape()[0]);
    const int num_nodes = static_cast<int>(tours.shape()[1]) - 1; // tours length = N+1
    assert(batch_size > 0);
    assert(num_nodes > 0);

    const auto tours_ptr = static_cast<const int*>(tours.data());

    // Output: (B, N, N)
    py::array_t<float> adj({batch_size, num_nodes, num_nodes});
    auto adj_ptr = static_cast<float*>(adj.request().ptr);

    {
        auto task_fn = [&](const int task_id) {
            float* adj_b = adj_ptr + static_cast<size_t>(task_id) * num_nodes * num_nodes;
            std::fill(adj_b, adj_b + static_cast<size_t>(num_nodes) * num_nodes, 0.0f);

            const int* tour_b = tours_ptr + static_cast<size_t>(task_id) * (num_nodes + 1);
            // Fill edges for idx in [0, N-1] (N edges), last element at position N should equal tour[0].
            for (int idx = 0; idx < num_nodes; ++idx) {
                const int u = tour_b[idx];
                const int v = tour_b[idx + 1];
                // Defensive: assume tours are valid permutations [0, N-1]
                assert(0 <= u && u < num_nodes);
                assert(0 <= v && v < num_nodes);
                adj_b[u * num_nodes + v] = 1.0f;
                adj_b[v * num_nodes + u] = 1.0f;
            }
        };

        parallelize(task_fn, batch_size, num_workers);
    }

    return adj;
}

PYBIND11_MODULE(tsp_tour2adj_impl, m) {
    m.doc() = "Convert TSP tours to (undirected) adjacency matrices.";

    m.def(
        "tsp_tour2adj",
        &tsp_tour2adj,
        py::arg("tours"),
        py::arg("num_workers") = 1,
        "Convert tours (B, N+1) into adjacency (B, N, N).\n"
    );
}


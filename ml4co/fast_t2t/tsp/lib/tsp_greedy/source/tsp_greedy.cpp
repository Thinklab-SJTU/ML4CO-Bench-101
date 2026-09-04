#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cassert>
#include <cstring>
#include <algorithm>
#include "tsp_greedy.hpp"
#include "parallel.hpp"

namespace py = pybind11;

/**
 * @brief Greedy insertion for TSP.
 *
 * Constructs TSP tour(s) from candidate edges using greedy insertion.
 * Supports both single instance and batched processing with parallel execution.
 * 
 * @param edge_indices Edge array(s):
 *                        - Single instance: 1D array of shape (E,)
 *                          Each element = row_idx * nodes_num + col_idx
 *                        - Batched: 2D array of shape (batch_size, E)
 * @param num_nodes Number of nodes in the TSP instance
 * @param num_workers Number of worker threads for parallel execution (for batched mode)
 * 
 * @return Tour(s):
 *         - Single instance: 1D array of shape (num_nodes + 1,)
 *         - Batched: 2D array of shape (batch_size, num_nodes + 1)
 */
inline py::array_t<int> tsp_greedy_insert(
    const py::array_t<int>& edge_indices,
    const int num_nodes,
    const int num_workers
) {
    assert(edge_indices.flags() & py::array::c_style);
    assert(num_nodes > 0);

    const bool batched = edge_indices.ndim() == 2;

    if (!batched) {
        assert(edge_indices.ndim() == 1);
        const auto indices_ptr = edge_indices.data();
        const int num_candidate_edges = static_cast<int>(edge_indices.shape()[0]);

        const size_t tour_len = static_cast<size_t>(num_nodes + 1);
        std::vector<int> tour_buf(tour_len);
        
        {
            pybind11::gil_scoped_release release;
            tsp::greedy_insert(tour_buf.data(), indices_ptr, num_nodes, num_candidate_edges);
        }

        py::array_t<int> result(tour_len);
        std::memcpy(result.request().ptr, tour_buf.data(), tour_len * sizeof(int));
        return result;
    } else {
        const int batch_size = static_cast<int>(edge_indices.shape()[0]);
        const int num_candidate_edges = static_cast<int>(edge_indices.shape()[1]);
        const auto indices_ptr = edge_indices.data();

        const size_t tour_len = static_cast<size_t>(num_nodes + 1);
        py::array_t<int> results({batch_size, (int)tour_len});
        auto results_ptr = static_cast<int*>(results.request().ptr);

        {
            pybind11::gil_scoped_release release;
            auto task_fn = [&](const int task_id) {
                tsp::greedy_insert(
                    results_ptr + task_id * tour_len,
                    indices_ptr + task_id * num_candidate_edges,
                    num_nodes,
                    num_candidate_edges
                );
            };
            parallelize(task_fn, batch_size, num_workers);
        }

        return results;
    }
}

PYBIND11_MODULE(tsp_greedy_impl, m) {
    m.doc() = "Greedy insertion for TSP from edge index sequence.";

    m.def(
        "tsp_greedy_insert",
        &tsp_greedy_insert,
        py::arg("edge_indices"),
        py::arg("num_nodes"),
        py::arg("num_workers") = 1,
        "Build TSP tour by greedy insertion from edge sequence.\n"
        "\n"
        "Parameters:\n"
        "  edge_indices: 1D array (E,) or 2D array (B, E) of edge indices; edge (i, j) as i * num_nodes + j\n"
        "  num_nodes: Number of nodes\n"
        "  num_workers: Number of worker threads for parallel execution\n"
        "\n"
        "Returns:\n"
        "  Tour(s) as 1D array (N+1,) or 2D array (B, N+1), last element equals first."
    );
}
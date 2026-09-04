#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <stdexcept>
#include <vector>

#include "parallel.hpp"
#include "tsp_2opt.hpp"

namespace py = pybind11;

/**
 * Exact pairwise 2-opt local search for Euclidean TSP (2D points).
 *
 * @param points     (B, N, 2) float32, C-contiguous
 * @param tours      (B, N+1) int32 closed tours, C-contiguous
 * @param max_iters  max 2-opt iterations per instance
 * @param type_2opt  1 = first improving, 2 = best improving
 * @param num_workers parallel workers over the batch
 */
inline py::array_t<int> tsp_2opt(
    const py::array_t<float>& points,
    const py::array_t<int>& tours,
    const int max_iters,
    const int type_2opt,
    const int num_workers
) {
    if (!(points.flags() & py::array::c_style)) {
        throw py::value_error("points must be C-contiguous");
    }
    if (!(tours.flags() & py::array::c_style)) {
        throw py::value_error("tours must be C-contiguous");
    }
    if (points.ndim() != 3 || static_cast<int>(points.shape(2)) != 2) {
        throw py::value_error("points must have shape (batch_size, N, 2)");
    }
    if (tours.ndim() != 2) {
        throw py::value_error("tours must have shape (batch_size, N+1)");
    }
    if (type_2opt != 1 && type_2opt != 2) {
        throw py::value_error("type_2opt must be 1 (first) or 2 (best)");
    }

    const int batch_size = static_cast<int>(points.shape(0));
    const int num_nodes = static_cast<int>(points.shape(1));

    if (batch_size < 1 || num_nodes < 1) {
        throw py::value_error("invalid points dimensions");
    }
    if (static_cast<int>(tours.shape(0)) != batch_size) {
        throw py::value_error("tours batch_size must match points");
    }
    if (static_cast<int>(tours.shape(1)) != num_nodes + 1) {
        throw py::value_error("tours second dimension must be N+1");
    }

    const int workers = num_workers < 1 ? 1 : num_workers;
    const size_t tour_len = static_cast<size_t>(num_nodes + 1);
    const size_t dist_per = static_cast<size_t>(num_nodes) * num_nodes;
    const size_t pts_stride = static_cast<size_t>(num_nodes) * 2;

    py::array_t<int> out({batch_size, num_nodes + 1});
    int* out_ptr = static_cast<int*>(out.request().ptr);
    std::memcpy(
        out_ptr,
        tours.data(),
        static_cast<size_t>(batch_size) * tour_len * sizeof(int)
    );

    const float* pts_ptr = points.data();
    std::vector<float> dists(static_cast<size_t>(batch_size) * dist_per);

    {
        py::gil_scoped_release release;
        auto dist_fn = [&](const int b) {
            tsp::compute_dist_matrix(
                pts_ptr + static_cast<size_t>(b) * pts_stride,
                dists.data() + static_cast<size_t>(b) * dist_per,
                num_nodes
            );
        };
        parallelize(dist_fn, batch_size, workers);

        auto opt_fn = [&](const int b) {
            tsp::two_opt(
                out_ptr + static_cast<size_t>(b) * tour_len,
                dists.data() + static_cast<size_t>(b) * dist_per,
                num_nodes,
                max_iters,
                type_2opt
            );
        };
        parallelize(opt_fn, batch_size, workers);
    }

    return out;
}

PYBIND11_MODULE(tsp_2opt_impl, m) {
    m.doc() = "Exact pairwise 2-opt local search for Euclidean TSP (CPU).";

    m.def(
        "tsp_2opt",
        &tsp_2opt,
        py::arg("points"),
        py::arg("tours"),
        py::arg("max_iters") = 5000,
        py::arg("type_2opt") = 2,
        py::arg("num_workers") = 1,
        "Run exact pairwise 2-opt on closed tours.\n"
        "\n"
        "Parameters:\n"
        "  points: (B, N, 2) float32 coordinates\n"
        "  tours: (B, N+1) int32 closed tours\n"
        "  max_iters: max 2-opt iterations per instance\n"
        "  type_2opt: 1 = first improving, 2 = best improving\n"
        "  num_workers: parallel workers over the batch\n"
        "\n"
        "Returns:\n"
        "  Optimized tours (B, N+1) int32 (input unchanged)."
    );
}

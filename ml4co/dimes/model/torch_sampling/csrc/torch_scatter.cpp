/** adopted from torch_scatter library **/
/** @link https://github.com/rusty1s/pytorch_scatter **/

#include <tuple>

#include <torch/script.h>

#include "torch_scatter.h"

namespace torch_scatter {

const auto OPTIONAL_TENSOR = torch::optional<torch::Tensor>();

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
segment_csr_fw(torch::Tensor src, torch::Tensor indptr,
               torch::optional<torch::Tensor> optional_out,
               std::string reduce) {
  if (src.device().is_cuda()) {
#ifdef WITH_CUDA
    return ::segment_csr_cuda(src, indptr, optional_out, reduce);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return ::segment_csr_cpu(src, indptr, optional_out, reduce);
  }
}

torch::Tensor segment_sum_csr(torch::Tensor src, torch::Tensor indptr) {
    return std::get<0>(segment_csr_fw(src, indptr, OPTIONAL_TENSOR, "sum"));
}

std::tuple<torch::Tensor, torch::Tensor> segment_max_csr(torch::Tensor src, torch::Tensor indptr) {
    auto result = segment_csr_fw(src, indptr, OPTIONAL_TENSOR, "max");
    return std::make_tuple(std::get<0>(result), std::get<1>(result).value());
}

}

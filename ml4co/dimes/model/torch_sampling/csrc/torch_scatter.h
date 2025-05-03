/** adopted from torch_scatter library **/
/** @link https://github.com/rusty1s/pytorch_scatter **/

#include <torch/extension.h>

extern std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
    segment_csr_cpu(
        torch::Tensor src, torch::Tensor indptr,
        torch::optional<torch::Tensor> optional_out,
        std::string reduce);

#ifdef WITH_CUDA
extern std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
    segment_csr_cuda(
        torch::Tensor src, torch::Tensor indptr,
        torch::optional<torch::Tensor> optional_out,
        std::string reduce);
#endif


namespace torch_scatter {

torch::Tensor segment_sum_csr(torch::Tensor src, torch::Tensor indptr);

std::tuple<torch::Tensor, torch::Tensor> segment_max_csr(torch::Tensor src, torch::Tensor indptr);

}

#include <bits/stdc++.h>

#include <torch/extension.h>

#include "torch_scatter.h"

using torch_scatter::segment_sum_csr;
using torch_scatter::segment_max_csr;

const float FLOAT_NEG_INF = -std::numeric_limits<float>::infinity(), FLOAT_NEG_INF_THRESH = -1e16, FLOAT_EPS = 1e-6;

const auto SLICE_NO_FIRST = torch::indexing::Slice(1, torch::indexing::None), SLICE_NO_LAST = torch::indexing::Slice(torch::indexing::None, -1);

class TspBaseSampler {
public:
    int n_nodes, n_edges, sample_size, n_cands;
    torch::Tensor x, deg, edge_v;
    torch::TensorOptions optionsF, optionsI;
    torch::Device device;
    torch::Tensor mask, cands, s_grp, e_ofs, par_ptr;
    TspBaseSampler(const torch::Tensor & x,
        const torch::Tensor & deg, const torch::Tensor & edge_v, // assuming edges sorted
        int sample_size, torch::TensorOptions optionsF):
        n_nodes(int(x.size(0))), n_edges(int(edge_v.size(-1))), sample_size(sample_size), n_cands(n_nodes),
        x(x), deg(deg), edge_v(edge_v),
        optionsF(optionsF), optionsI(optionsF.dtype(torch::kInt64)), device(optionsF.device()),
        mask(torch::ones({sample_size, n_nodes}, optionsF.dtype(torch::kBool))),
        cands(torch::arange(n_nodes, torch::dtype(torch::kInt64)).repeat({sample_size, 1}).to(device)),
        s_grp(torch::arange(sample_size, optionsI)), e_ofs(torch::arange(n_nodes * sample_size, optionsI)),
        par_ptr(torch::cat({torch::zeros({1}, optionsI), deg.index({SLICE_NO_LAST})}, 0).cumsum(0)) {}
    virtual ~TspBaseSampler() {}
    virtual torch::Tensor init() = 0;
    virtual void start(const torch::Tensor & v0) {
        this->mask.scatter_(-1, v0.unsqueeze(-1), 0);
    }
    virtual void sample_cands(int i, const torch::Tensor & u, torch::Tensor & v, const torch::Tensor & s_idx) {
        int s_size = s_idx.size(0);
        if (s_size > 0) {
            int n_next = this->n_nodes - i;
            auto us = u.index({s_idx}).unsqueeze(1).expand({s_size, n_next}).flatten();
            auto cs = this->cands.index({s_idx});
            auto vs = cs.masked_select(this->mask.index({s_idx}).gather(1, cs));
            auto dist = at::pairwise_distance(x.index({us}), x.index({vs})).reshape({s_size, n_next});
            auto vs_idx = dist.argmin(1, true);
            v.index_put_({s_idx}, vs.reshape({s_size, n_next}).gather(1, vs_idx).squeeze(1));
        }
    }
    virtual torch::Tensor compute_scores(const torch::Tensor & e_msk, const torch::Tensor & e_idx) = 0;
    virtual void update(
        const torch::Tensor & v_idx, const torch::Tensor & s_idx,
        const torch::Tensor & e_idx, const torch::Tensor & e_msk,
        const torch::Tensor & e_grp, const torch::Tensor & e_ptr) {}
    virtual void transit(int i, const torch::Tensor & u, const torch::Tensor & v) {
        this->mask.scatter_(-1, v.unsqueeze(-1), 0);
        if ((this->n_nodes - i - 1) * 2 < this->n_cands && i < this->n_nodes - 1) {
            this->n_cands = this->n_nodes - i - 1;
            this->cands = this->cands.masked_select(this->mask.gather(-1, this->cands)).view({this->sample_size, this->n_cands});
        }
    }
    virtual void finalize(const torch::Tensor & u, const torch::Tensor & v0) {}
    virtual std::vector<torch::Tensor> result() const { return {}; }
    virtual std::vector<torch::Tensor> sample() {
        auto v0 = this->init(), u = v0, v = torch::empty_like(u);
        this->start(v0);
        auto e_wid = torch::empty({this->sample_size + 1}, this->optionsI);
        e_wid.index_put_({0}, 0);
        torch::Tensor e_ptr, e_idx, e_deg, e_grp, e_msk, scores, v_scr, v_idx, s_msk, s_idx;
        for (int i = 1; i < this->n_nodes; ++i) {
            e_deg = this->deg.index({u});
            e_grp = this->s_grp.repeat_interleave(e_deg);
            e_wid.index_put_({SLICE_NO_FIRST}, e_deg);
            e_ptr = e_wid.cumsum(0);
            e_idx = (this->par_ptr.index({u}) - e_ptr.index({SLICE_NO_LAST})).repeat_interleave(e_deg) + this->e_ofs.index({torch::indexing::Slice(0, e_grp.size(0))});
            e_msk = this->mask.index({e_grp, this->edge_v.index({e_idx})});
            scores = this->compute_scores(e_msk, e_idx);
            std::tie(v_scr, v_idx) = segment_max_csr(scores, e_ptr);
            s_msk = (v_scr > FLOAT_NEG_INF_THRESH).logical_and(v_idx != e_idx.size(0));
            s_idx = s_msk.nonzero();
            this->update(v_idx.masked_select(s_msk), s_idx, e_idx, e_msk, e_grp, e_ptr);
            v.index_put_({s_idx}, this->edge_v.index(e_idx.index(v_idx.index(s_idx))));
            this->sample_cands(i, u, v, s_msk.logical_not().nonzero().squeeze(-1));
            this->transit(i, u, v);
            u = v.clone();
        }
        this->finalize(u, v0);
        return this->result();
    }
};

class TspSampler: public TspBaseSampler {
public:
    using Super = TspBaseSampler;
    torch::Tensor y, tours;
    TspSampler(const torch::Tensor & x,
        const torch::Tensor & deg, const torch::Tensor & edge_v,
        int sample_size, torch::TensorOptions optionsF):
        Super(x, deg, edge_v, sample_size, optionsF),
        y(torch::zeros(sample_size, optionsF)),
        tours(torch::empty({n_nodes, sample_size}, optionsI)) {}
    virtual ~TspSampler() {}
    virtual void start(const torch::Tensor & v0) {
        this->Super::start(v0);
        this->tours.index_put_({0}, v0);
    }
    virtual void transit(int i, const torch::Tensor & u, const torch::Tensor & v) {
        this->Super::transit(i, u, v);
        this->y.add_(at::pairwise_distance(this->x.index({u}), this->x.index({v})));
        this->tours.index_put_({i}, v);
    }
    virtual void finalize(const torch::Tensor & u, const torch::Tensor & v0) {
        this->Super::finalize(u, v0);
        this->y.add_(at::pairwise_distance(this->x.index({u}), this->x.index({v0})));
    }
    virtual std::vector<torch::Tensor> result() const {
        auto result = this->Super::result();
        result.push_back(this->y);
        result.push_back(this->tours.t());
        return result;
    }
};

class TspGreedySampler: public TspSampler {
public:
    using Super = TspSampler;
    torch::Tensor par, neg_inf;
    TspGreedySampler(const torch::Tensor & x,
        const torch::Tensor & deg, const torch::Tensor & edge_v,
        const torch::Tensor & par, int sample_size):
        Super(x, deg, edge_v, sample_size, par.options()),
        par((par - par.mean()).clone()),
        neg_inf(torch::full({1}, FLOAT_NEG_INF, par.options())) {}
    virtual ~TspGreedySampler() {}
    virtual torch::Tensor init() {
        assert(this->n_nodes >= this->sample_size);
        return torch::randperm(this->n_nodes, this->optionsI)
            .index({torch::indexing::Slice(0, this->sample_size)});
    }
    virtual torch::Tensor compute_scores(const torch::Tensor & e_msk, const torch::Tensor & e_idx) {
        return this->par.index({e_idx}).where(e_msk, this->neg_inf);
    }
};

class TspSoftmaxSampler: public TspGreedySampler {
public:
    using Super = TspGreedySampler;
    float y_bl;
    torch::Tensor par_e;
    TspSoftmaxSampler(const torch::Tensor & x,
        const torch::Tensor & deg, const torch::Tensor & edge_v,
        const torch::Tensor & par, int sample_size, float y_bl):
        Super(x, deg, edge_v, par, sample_size), y_bl(y_bl) {}
    virtual ~TspSoftmaxSampler() {}
    virtual torch::Tensor init() {
        return torch::randint(this->n_nodes, {this->sample_size}, this->optionsI);
    }
    virtual torch::Tensor compute_scores(const torch::Tensor & e_msk, const torch::Tensor & e_idx) {
        this->par_e = this->Super::compute_scores(e_msk, e_idx);
        return this->par_e - torch::empty_like(this->par_e).exponential_().log();
    }
};

class TspSoftmaxGradSampler: public TspSoftmaxSampler {
public:
    using Super = TspSoftmaxSampler;
    std::vector<torch::Tensor> gi_i_idx, gi_s_idx, gi_p_idx, gi_p_grp, gi_probs;
    TspSoftmaxGradSampler(const torch::Tensor & x,
        const torch::Tensor & deg, const torch::Tensor & edge_v,
        const torch::Tensor & par, int sample_size, float y_bl):
        Super(x, deg, edge_v, par, sample_size, y_bl) {
        gi_i_idx.reserve(this->n_nodes - 1);
        gi_s_idx.reserve(this->n_nodes - 1);
        gi_p_idx.reserve(this->n_nodes - 1);
        gi_p_grp.reserve(this->n_nodes - 1);
        gi_probs.reserve(this->n_nodes - 1);
    }
    virtual ~TspSoftmaxGradSampler() {}
    virtual void update(
        const torch::Tensor & v_idx_, const torch::Tensor & s_idx_,
        const torch::Tensor & e_idx_, const torch::Tensor & e_msk_,
        const torch::Tensor & e_grp_, const torch::Tensor & e_ptr_) {
        auto i_idx_ = e_idx_.index({v_idx_}), e_msk_idx_ = e_msk_.nonzero();
        auto p_idx_ = e_idx_.index({e_msk_idx_}), p_grp_ = e_grp_.index({e_msk_idx_});
        auto logits = this->par_e - std::get<0>(segment_max_csr(this->par_e, e_ptr_)).index({e_grp_});
        auto par_exp_ = logits.exp(), p_denom_ = segment_sum_csr(par_exp_, e_ptr_);
        auto probs_ = par_exp_.index({e_msk_idx_}) / p_denom_.index({p_grp_});
        this->gi_i_idx.push_back(i_idx_);
        this->gi_s_idx.push_back(s_idx_);
        this->gi_p_idx.push_back(p_idx_);
        this->gi_p_grp.push_back(p_grp_);
        this->gi_probs.push_back(probs_);
    }
    virtual std::vector<torch::Tensor> result() const {
        auto result = this->Super::result();
        result.pop_back();
        auto grad = torch::zeros(this->par.sizes(), this->optionsF);
        auto coefs = this->y;
        if (std::isnan(this->y_bl))
            coefs = coefs - this->y.mean();
        else
            coefs = coefs - this->y_bl;
        auto i_idx = torch::cat(this->gi_i_idx, 0);
        auto s_idx = torch::cat(this->gi_s_idx, 0).squeeze(-1);
        auto p_idx = torch::cat(this->gi_p_idx, 0).squeeze(-1);
        auto p_grp = torch::cat(this->gi_p_grp, 0).squeeze(-1);
        auto probs = torch::cat(this->gi_probs, 0).squeeze(-1);
        grad.scatter_add_(0, i_idx, coefs.index({s_idx}));
        grad.scatter_add_(0, p_idx, coefs.index({p_grp}) * -probs);
        grad.div_(this->sample_size);
        result.push_back(grad);
        return result;
    }
};

std::vector<torch::Tensor> tsp_greedy(const torch::Tensor & x,
    const torch::Tensor & deg, const torch::Tensor & edge_v,
    const torch::Tensor & par, int sample_size) {
    return TspGreedySampler(x, deg, edge_v, par, sample_size).sample(); // ys, tours
}

std::vector<torch::Tensor> tsp_softmax(const torch::Tensor & x,
    const torch::Tensor & deg, const torch::Tensor & edge_v,
    const torch::Tensor & par, int sample_size, float y_bl) {
    return TspSoftmaxSampler(x, deg, edge_v, par, sample_size, y_bl).sample(); // ys, tours
}

std::vector<torch::Tensor> tsp_softmax_grad(const torch::Tensor & x,
    const torch::Tensor & deg, const torch::Tensor & edge_v,
    const torch::Tensor & par, int sample_size, float y_bl) {
    return TspSoftmaxGradSampler(x, deg, edge_v, par, sample_size, y_bl).sample(); // ys, grad
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tsp_greedy",       &tsp_greedy,       "TSP greedy sampling");
    m.def("tsp_softmax",      &tsp_softmax,      "TSP softmax sampling");
    m.def("tsp_softmax_grad", &tsp_softmax_grad, "TSP softmax samping and gradient");
}

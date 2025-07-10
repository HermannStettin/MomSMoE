# Integrating Momentum Into SMoEIntegrating Momentum Into SMoE

We implemented code from [MomentumSMoE: Integrating Momentum into Sparse Mixture of Experts](https://arxiv.org/abs/2410.14574) paper and extended it on Mars and Ademamix optimizers.

### Prerequisites

- pytorch
- [fastmoe](https://github.com/laekov/fastmoe)
- wandb
- Wikitext-103 dataset

See directory scripts/ for examples.

### Tested Models
- [x] SMoE without Momentum
- [x] Heavy-ball Momentum SMoE (from original paper)
- [x] Adam SMoE (from original paper)
- [x] Mars SMoE
- [x] Ademamix SMoE

For the results see this wandb report: https://api.wandb.ai/links/bigesod536-team/m6j5jd8f

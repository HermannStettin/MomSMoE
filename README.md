# MomSMoE

Code is based upon [MomentumSMoE: Integrating Momentum into Sparse Mixture of Experts](https://arxiv.org/abs/2410.14574) paper.

### Prerequisites

- pytorch
- [fastmoe](https://github.com/laekov/fastmoe)
- wandb
- Wikitext-103 dataset

See directory scripts/ for examples.

### Tested Models
- [x] SMoE without Momentum
- [x] Heavy-ball Momentum SMoE (from orignal paper)
- [x] Adam SMoE (from orignal paper)
- [x] Mars SMoE
- [x] Ademamix SMoE

For the results see this wandb report: https://api.wandb.ai/links/bigesod536-team/m6j5jd8f

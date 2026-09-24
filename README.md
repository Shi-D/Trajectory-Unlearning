# Trajectory Unlearning on LLM-based Agents



## Repository Structure

```
.
├── README.md                    # this file
├── LICENSE
├── Notice.txt
├── requirements.txt
├── pyproject.toml
├── setup.py
├── .gitignore
├── .env.example                  # template for API keys / model & data roots — copy to .env
├── figs/                         # images used in the READMEs
├── unlearn/                         # unlearning method implementations
│                                  #   ga_unlearn.py / npo_unlearn.py / dpo_unlearn.py (offline SFT-style baselines)
│                                  #   unlearn_inject.py (GiRPO's forget-trajectory injection + reward shaping)
│                                  #   unlearn_offpolicy.py (NPO+GRPO's off-policy step-pair sampling)
│                                  #   unlearn.py / unlearn_reward.py / unlearn_actionseq.py / analysis.py / prompting.py / skill_gen.py
├── gigpo/                        # GiGPO / Advantage estimator (core_gigpo.py)
├── agent_system/
│   ├── environments/
│   │   ├── env_package/
│   │   │   ├── alfworld/         # ALFWorld env wrapper
│   │   │   ├── webshop/          # WebShop env wrapper (vendored WebShop repo + setup.sh)
│   │   │   ├── sokoban/
│   │   │   ├── search/
│   │   │   ├── sciworld/
│   │   │   ├── appworld/
│   │   │   └── gym_cards/
│   │   ├── env_manager.py        # per-benchmark EnvironmentManager classes (obs formatting, task-key plumbing)
│   │   └── base.py
│   ├── multi_turn_rollout/       # rollout loop, sample preprocessing
│   ├── reward_manager/
│   └── memory/
├── verl/                         # verl PPO/GRPO training framework
│   ├── trainer/                  # main_ppo*.py entrypoints, ray_trainer.py, core_algos.py, config/ppo_trainer.yaml
│   ├── workers/                  # FSDP actor/rollout/critic workers
│   ├── single_controller/
│   └── utils/
├── scripts/                      # checkpoint tooling (model_merger.py: FSDP-sharded -> HF format; merge.sh)
├── examples/
│   └── traj_trainer/             # every training entrypoint used by this project
│       ├── _common/              # alfworld.sh / webshop.sh — shared hydra-CLI trainer wrapper
│       └── ...                   # one run_*.sh per baseline x benchmark (see Baselines below)
├── collection/                   # ALFWorld: data-prep, eval, LLM-judge, analysis scripts + result reports (*.md)
├── collection_webshop/           # WebShop: data-prep, eval, LLM-judge, analysis scripts + result reports (*.md)
└── ablation_study/
    ├── data/                     # small CSV summaries backing report.md / report_v2.md
    ├── report.md
    └── report_v2.md              # delta/lambda reward-formula ablation writeup
```



## 🛠️ Installation

### 1. Core veRL

```bash
conda create -n traj python==3.12 -y
conda activate traj

pip3 install vllm==0.11.0
pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
```

### 2. ALFWorld benchmark

```bash
conda activate traj
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip3 install alfworld
alfworld-download -f    # downloads PDDL/game files + MaskRCNN detector to ~/.cache/alfworld/
```

### 3. WebShop benchmark

```bash
conda create -n traj-webshop python==3.10 -y
conda activate traj-webshop

cd agent_system/environments/env_package/webshop/webshop
./setup.sh -d all
cd -    # back to repo root

pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.2
```

If it fails, consider using [AgentGym's WebShop environment](https://github.com/WooooDyy/AgentGym/tree/main/agentenv-webshop) as a fallback.


## Baselines

All ALFWorld scripts are invoked directly with `bash` (no `#SBATCH`
header of their own); all WebShop scripts are `sbatch`-ready. Every
training script accepts overrides via `sbatch --export=ALL,VAR=value ...`
(WebShop) or as plain environment variables before the `bash` call
(ALFWorld).

| Method | What it does | ALFWorld train | ALFWorld postprocess | WebShop train |
|---|---|---|---|---|
| **GA** | Gradient ascent on the forget set (`-CE(forget) + gamma*CE(retain)`) | `run_ga_unlearn_mixed100.sh` | `run_ga_unlearn_mixed100_postprocess.sh` | `run_ga_unlearn_webshop.sh` |
| **NPO** | Bounded preference loss vs. a frozen reference on the forget set + retain CE | `run_npo_unlearn_mixed100.sh` | `run_npo_unlearn_mixed100_postprocess.sh` | `run_npo_unlearn_webshop.sh` |
| **DPO** | Direct preference loss on forget trajectories (no retain term) | `run_dpo_unlearn_mixed100.sh` | `run_dpo_unlearn_mixed100_postprocess.sh` | `run_dpo_unlearn_webshop.sh` |
| **GRPO-control** | Vanilla on-policy GRPO trained only on the retain pool (forget set never sampled) | `run_alfworld_grpo_control_mixed100_retain.sh` | `run_grpo_control_mixed100_retain_postprocess.sh` | `run_grpo_control_webshop_retain.sh` |
| **NPO+GRPO** (`v8`) | On-policy GRPO on the full pool + an off-policy NPO auxiliary loss (forget: NPO loss vs. frozen ref; retain: CE), cycled batch-by-batch over the recorded step pairs | `run_unlearn_mixed100_v8.sh` | `run_unlearn_mixed100_postprocess_v8.sh` | `run_unlearn_webshop_v8.sh` |
| **GiRPO** (`v22`) | On-policy GRPO; whenever a rollout's game matches the forget set, an extra synthetic trajectory (recorded prompt/response pairs) is injected with reward forced to `min(real rollout rewards) - delta`, excluded from the group's own advantage baseline. Optional *guaranteed coverage*: a fixed number of task-slots per step are dedicated to the forget set on a deterministic cycle, so every forget task is visited within a bounded number of epochs instead of relying on random luck | `run_unlearn_mixed100_v22.sh` | `run_unlearn_mixed100_postprocess_v22.sh` | `run_unlearn_webshop_girpo.sh` |



## Citation

If you find this project useful, welcome to cite us.


# Trajectory Unlearning on LLM-based Agents



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

All scripts live in `examples/traj_trainer/` and are run from anywhere with
`bash` (wrap them with `sbatch` / your scheduler as needed). Set `MODELS_ROOT`
(and optionally the other variables in `.env.example`) in `.env` or the
environment first. Every script accepts overrides as environment variables,
e.g. `GAMMA=10.0 bash examples/traj_trainer/run_alfworld_ga.sh`; extra
arguments to the RL scripts are forwarded to Hydra.

| Method | ALFWorld train                                            | WebShop train |
|---|-----------------------------------------------------------|---|
| **GA** | `run_alfworld_ga.sh` | `run_webshop_ga.sh` |
| **NPO** | `run_alfworld_npo.sh`                                     | `run_webshop_npo.sh` |
| **DPO** | `run_alfworld_dpo.sh`                                     | `run_webshop_dpo.sh` |
| **GRPO-control** | `run_alfworld_grpo_control.sh`                            | `run_webshop_grpo_control.sh` |
| **NPO+GRPO** | `run_alfworld_npo_grpo.sh`                                | `run_webshop_npo_grpo.sh` |
| **GiRPO** | `run_alfworld_girpo.sh`                                   | `run_webshop_girpo.sh` |



## Citation

If you find this project useful, welcome to cite us.


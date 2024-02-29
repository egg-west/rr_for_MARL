# Higher Replay Ratio for MARL

## 1. Configuration

```
conda create -n marl python=3.8.1
pip install -r ./requirements.txt
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install POT wandb==0.12.21 absl-py sacred einops torch_scatter gym
pip install info-nce-pytorch tqdm

./install_sc2.sh
```

## 2. To run
Please first edit the configuration in `src/config/default.yaml`, `src/config/envs/sc2.yaml`, and `src/config/algs/*.yaml`. `*` includes `vdn, qmix, qplex`

```
python3 src/mtrl_main.py --config=vdn --env-config=sc2 with env_args.map_name=3s5z seed=123
```

# License
All the source code that has been taken from the PyMARL repository was licensed (and remains so) under the Apache License v2.0 (included in `LICENSE` file).
Any new code is also licensed under the Apache License v2.0

# How to train

```shell
    git checkout easyaction
    cd my_submission/entry
    python training.py --use_gpu_for_driver=True --num_gpus=1 --num_cpus=32 --num_workers_per_device=2 --algorithm=PPO
```
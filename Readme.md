# PPO-Pytorch
This is a well-written and easy-to-follow implementation of Proximal Policy Optimization algortihm, please refer to original paper
from UCB: https://arxiv.org/abs/1707.06347

# HowTo

Step 1. set OMP_NUM_THREADS=1, otherwise it will block multiprocessing threads.
```
export OMP_NUM_THREADS=1
```

Step 2. run
```
python train.py
```



# Hopper
![Hopper](res/Hopper.gif)

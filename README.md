# RL system for Kube Scheduler

We use DQN, TD3, SAC for DRL + GNN method for custom kubernetes scheduler.

## How to run

1. Setup your kubernetes cluster. you can refer [k8s-SFC-deployment/k8s](https://github.com/k8s-SFC-deployment/k8s).
2. You must change your kubernetes scheduler to use our custom scheduler-plugin. You must edit below file for your cluster.
   ```bash
   $ sudo cp scheduler-config/scheduler-config-rl.yaml /etc/kubernetes
   # for backup default setup
   $ sudo cp /etc/kubernetes/manifest/kube-scheduler.yaml scheduler-config
   # if you change this file, kubernetes automatically restart kube-scheduler with this file.
   $ sudo cp scheduler-config/kube-scheduler-rl.yaml /etc/kubernetes/manifest/kube-scheduler.yaml
   ```
3. start each model's train file.
   ```bash
   $ python -m src.dqn.train
   $ python -m src.sac.train
   $ python -m src.td3.train
   ```

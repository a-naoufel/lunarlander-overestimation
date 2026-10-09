import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.td3.policies import TD3Policy


def make_env(cfg):
    kwargs = {"continuous": True} if cfg.env.startswith("LunarLander") else {}
    return gym.make(cfg.env, **kwargs)


def add_layer_norm(root):
    mlps = [m for m in root.modules()
            if isinstance(m, nn.Sequential) and any(isinstance(c, nn.Linear) for c in m)]
    added = 0
    for mlp in mlps:
        linears = [i for i, c in enumerate(mlp) if isinstance(c, nn.Linear)]
        for i in linears[:-1]:
            lin = mlp[i]
            mlp[i] = nn.Sequential(lin, nn.LayerNorm(lin.out_features, device=lin.weight.device))
            added += 1
    return added


def make_policy_class(actor_ln, critic_ln):
    class LayerNormPolicy(TD3Policy):
        def make_actor(self, features_extractor=None):
            actor = super().make_actor(features_extractor)
            if actor_ln:
                add_layer_norm(actor)
            return actor

        def make_critic(self, features_extractor=None):
            critic = super().make_critic(features_extractor)
            if critic_ln:
                add_layer_norm(critic)
            return critic

    return LayerNormPolicy


def check_architecture(model, cfg):
    def n_ln(net):
        return sum(isinstance(m, nn.LayerNorm) for m in net.modules())

    want_actor = cfg.n_layers if cfg.actor_ln else 0
    want_critic = cfg.n_layers * cfg.n_critics if cfg.critic_ln else 0
    for name, net, want in (("actor", model.actor, want_actor), ("actor_target", model.actor_target, want_actor),
                            ("critic", model.critic, want_critic), ("critic_target", model.critic_target, want_critic)):
        assert n_ln(net) == want, f"{name}: expected {want} LayerNorm layers, found {n_ln(net)}"
    for name, net in (("actor", model.actor), ("critic", model.critic)):
        in_optimizer = {id(p) for g in net.optimizer.param_groups for p in g["params"]}
        assert all(id(p) in in_optimizer for p in net.parameters()), f"{name}: parameter missing from its optimizer"


def build_model(cfg, env):
    arch = [cfg.hidden] * cfg.n_layers
    n_actions = env.action_space.shape[0]
    common = dict(
        policy=make_policy_class(cfg.actor_ln, cfg.critic_ln),
        env=env,
        learning_rate=cfg.lr,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        batch_size=cfg.batch_size,
        tau=cfg.tau,
        gamma=cfg.gamma,
        train_freq=(1, "step"),
        gradient_steps=cfg.utd,
        action_noise=NormalActionNoise(np.zeros(n_actions), cfg.expl_noise * np.ones(n_actions)),
        seed=cfg.seed,
        device="cpu",
        verbose=0,
    )
    policy_kwargs = dict(net_arch=dict(pi=arch, qf=arch), activation_fn=nn.ReLU)
    if cfg.algo == "td3":
        policy_kwargs["n_critics"] = cfg.n_critics
        model = TD3(**common, policy_kwargs=policy_kwargs, policy_delay=cfg.policy_delay,
                    target_policy_noise=cfg.target_noise, target_noise_clip=cfg.noise_clip)
    else:
        model = DDPG(**common, policy_kwargs=policy_kwargs)
    check_architecture(model, cfg)
    return model


class SB3Agent:
    def __init__(self, model):
        self.model = model

    def act(self, obs):
        return self.model.predict(obs, deterministic=True)[0]

    def q_values(self, obs, actions):
        policy = self.model.policy
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=policy.device)
            actions = torch.as_tensor(policy.scale_action(actions), dtype=torch.float32, device=policy.device)
            return torch.cat(self.model.critic(obs, actions), dim=1).cpu().numpy()


def self_check():
    import stable_baselines3
    from mp_sb3.config import Config
    print(f"stable-baselines3 {stable_baselines3.__version__} | torch {torch.__version__} | gymnasium {gym.__version__}")
    for algo in ("ddpg", "td3"):
        for ln in (False, True):
            cfg = Config(algo=algo, critic_ln=ln, actor_ln=ln, learning_starts=100, total_steps=300,
                         buffer_size=1_000, tensorboard=False)
            model = build_model(cfg, make_env(cfg))
            model.learn(300)
            obs_dim, act_dim = model.observation_space.shape[0], model.action_space.shape[0]
            q = SB3Agent(model).q_values(np.zeros((5, obs_dim), np.float32), np.zeros((5, act_dim), np.float32))
            n_params = sum(p.numel() for p in model.critic.parameters())
            print(f"{algo:5s} LayerNorm={ln!s:5s} critics={q.shape[1]} critic params={n_params:,} "
                  f"policy_delay={getattr(model, 'policy_delay', None)} Q(0,0)={q[0].round(3)}")
    print("OK")


if __name__ == "__main__":
    self_check()
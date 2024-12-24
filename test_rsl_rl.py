from rsl_rl.env import MMVecEnv
from rsl_rl.runners import OnPolicyRunnerMM
import torch

class mycustomenv(MMVecEnv):
    def __init__(self,
                 num_envs,
                 num_obs,
                 num_ref_obs,
                 num_actions,
                 max_episode_length,
                 device,
                 num_privileged_obs = None,
                 num_privileged_ref_obs = None):
        self.num_actions = num_actions
        self.max_episode_length = max_episode_length
        self.device = device
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.num_ref_obs = num_ref_obs
        self.num_privileged_obs = num_privileged_obs if num_privileged_obs is not None else 0
        self.num_privileged_ref_obs = num_privileged_ref_obs if num_privileged_ref_obs is not None else 0
        self.obs_buf = torch.zeros((num_envs, num_obs), device=device)
        self.ref_obs_buf = torch.zeros((num_envs, num_ref_obs), device=device)
        self.privileged_obs_buf = torch.zeros((num_envs, self.num_privileged_obs), device=device)
        self.privileged_ref_obs_buf = torch.zeros((num_envs, self.num_privileged_ref_obs), device=device)
        self.ref_obs_mask_buf = torch.zeros((num_envs, num_ref_obs), device=device)
        self.privileged_ref_obs_mask_buf = torch.zeros((num_envs, self.num_privileged_ref_obs), device=device)
        self.extras = {}
        self.reset()

    def get_observations(self):
        return self.obs_buf, {}
    
    def reset(self):
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.ref_obs_buf = torch.zeros((self.num_envs, self.num_ref_obs), device=self.device)
        self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device)
        self.privileged_ref_obs_buf = torch.zeros((self.num_envs, self.num_privileged_ref_obs), device=self.device)
        self.ref_obs_mask_buf = torch.zeros((self.num_envs, self.num_ref_obs), device=self.device)
        self.privileged_ref_obs_mask_buf = torch.zeros((self.num_envs, self.num_privileged_ref_obs), device=self.device)
        obs_buf, obs_dict = self.get_observations()
        ref_obs, ref_obs_dict = self.get_reference_observations()
        joined_dict = {**obs_dict, **ref_obs_dict}
        return obs_buf, ref_obs, joined_dict
    
    def get_reference_observations(self):
        return (self.ref_obs_buf, self.ref_obs_mask_buf), {}
    
    def step(self, actions):
        # test, so just add random values to the observations
        self.obs_buf += torch.rand_like(self.obs_buf)
        self.ref_obs_buf += torch.rand_like(self.ref_obs_buf)
        self.privileged_obs_buf += torch.rand_like(self.privileged_obs_buf)
        self.privileged_ref_obs_buf += torch.rand_like(self.privileged_ref_obs_buf)
        self.ref_obs_mask_buf = torch.rand_like(self.ref_obs_mask_buf)
        self.privileged_ref_obs_mask_buf = torch.rand_like(self.privileged_ref_obs_mask_buf)
        rewards = torch.rand(self.num_envs, device=self.device)
        dones = torch.zeros(self.num_envs, device=self.device)
        infos = {}
        return self.obs_buf, (self.ref_obs_buf, self.ref_obs_mask_buf), rewards, dones, infos
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf, {}
    
    def get_privileged_reference_observations(self):
        return (self.privileged_ref_obs_buf, self.privileged_ref_obs_mask_buf), {}
    
train_cfg={
    "algorithm":{
        "class_name":"MMPPO",
        "value_loss_coef":1.0,
        "clip_param":0.2,
        "use_clipped_value_loss":True,
        "desired_kl":0.01,
        "entropy_coef":0.01,
        "gamma":0.99,
        "lam":0.95,
        "max_grad_norm":1.0,
        "learning_rate":0.001,
        "num_mini_batches":4,
        "schedule":"adaptive",
    },
    "policy":{
        "class_name":"ActorCriticPolicyMM",
        "max_len":4,
        "dim_model":128,
        "num_layers":4,
        "num_heads":8,
        "init_noise_std":0.1,
    },
    "runner":{
        "num_steps_per_env":24,
        "max_iterations":100,
        "empirical_normalization":False,
        "save_interval":50,
        "experiment_name":"test_mmppo",
        "run_name":"test_mmppo",
        "logger": "tensorboard",
        "resume": False,
        "load_run":-1,
        "resume_path": None,
        "checkpoint":-1,
    },
    "runner_class_name":"OnPolicyRunnerMM",
    "seed":0,
}

num_envs = 2048
num_obs = 64
num_ref_obs = 40
num_actions = 29
max_episode_length = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
env = mycustomenv(num_envs, num_obs, num_ref_obs, num_actions, max_episode_length, device)

runner = OnPolicyRunnerMM(
    env=env,
    train_cfg=train_cfg,
    device=device,

)

import numpy as np
import pickle
import torch

from mjrl.policies.gaussian_mlp import MLP

class BC():
    def __init__(self, policy):
        self.policy = policy
        print(f'Obs dim {self.policy.observation_dim} Act dim {self.policy.action_dim}')
        print(self.policy.in_shift, self.policy.in_scale)
        print(self.policy.out_shift, self.policy.out_scale)

    # For 1-batch query only!
    def predict(self, sample):
        with torch.no_grad():
            at = self.policy.forward(sample)
            if False: # NO RANDOM
                at = at + torch.randn(at.shape).to(policy.device) * torch.exp(policy.log_std)
            # clamp states and actions to avoid blowup
            return at.to('cpu').detach().numpy()

def init_agent_from_config(policy_pickle, device):

    policy = pickle.load(open(policy_pickle, 'rb'))
    policy.set_param_values(policy.get_param_values())
    policy.to(device)

    return BC(policy)

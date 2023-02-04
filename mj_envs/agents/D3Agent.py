import numpy as np
import pickle
import torch
import d3rlpy

class D3Agent():
    def __init__(self, policy, device):
        self.policy = policy
        self.device = device
    # For 1-batch query only!
    def predict(self, sample):
        with torch.no_grad():
            input = torch.from_numpy(sample).float().unsqueeze(0).to(self.device)
            at = self.policy(input)[0].to('cpu').detach().numpy()
        # print("action: ", at)
        return at

def init_agent_from_config(policy_pt, device):
    policy = torch.jit.load(policy_pt)
    policy.to(device)
    # import pdb; pdb.set_trace()
    # agent_class_name = f'd3rlpy.algos.{config.agent.cls}'
    # policy = eval(agent_class_name)()
    # policy.load_state_dict(torch.load(config.agent.policy_pt))
    # policy.eval()
    # policy = torch.load(config.agent.policy_pt)

    return D3Agent(policy, device)


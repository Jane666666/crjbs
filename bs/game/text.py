import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                device, numOfEpisodes, env):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.device = device
        self.env = env
        self.numOfEpisodes = numOfEpisodes

    # 根据动作概率分布随机采样
    def takeAction(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        action_probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']
        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor(np.array([state_list[i]]), dtype=torch.float).to(self.device)
            action = torch.tensor(np.array([action_list[i]]), dtype=torch.int64).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降

    def REINFORCERun(self):
        returnList = []
        for i in range(10):
            with tqdm(total=int(self.numOfEpisodes / 10), desc='Iteration %d' % i) as pbar:
                for episode in range(int(self.numOfEpisodes / 10)):
                    # initialize state
                    state, info = self.env.reset()
                    terminated = False
                    truncated = False
                    episodeReward = 0
                    transition_dict = {
                        'states': [],
                        'actions': [],
                        'next_states': [],
                        'rewards': [],
                        'terminateds': [],
                        'truncateds':[]
                    }
                    # Loop for each step of episode:
                    while (not terminated) or (not truncated):
                        action = self.takeAction(state)
                        next_state, reward, terminated, truncated, info = self.env.step(action)
                        if terminated or truncated:
                            break
                        transition_dict['states'].append(state)
                        transition_dict['actions'].append(action)
                        transition_dict['next_states'].append(next_state)
                        transition_dict['rewards'].append(reward)
                        transition_dict['terminateds'].append(terminated)
                        transition_dict['truncateds'].append(truncated)
                        state = next_state
                        episodeReward += reward
                    self.update(transition_dict)
                    returnList.append(episodeReward)
                    if (episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                        pbar.set_postfix({
                            'episode':
                                '%d' % (self.numOfEpisodes / 10 * i + episode + 1),
                            'return':
                                '%.3f' % np.mean(returnList[-10:])
                        })
                    pbar.update(1)
        return returnList

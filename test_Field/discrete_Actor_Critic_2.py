import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

gamma = 0.998
env = gym.make('CartPole-v1')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
board = SummaryWriter()
mini_batch_size = 100

class Actor(nn.Module):
    def __init__(self,
                 state_size=4,
                 h1_size=200,
                 h2_size=100,
                 action_size=2):
        super(Actor,self).__init__()
#        self.net = nn.Sequential(
#                nn.Linear(state_size,h1_size),
#                nn.ReLU(),
#                nn.Linear(h1_size,action_size),
#                nn.Softmax()
#                )
        self.fc_sh1 = nn.Linear(state_size,h1_size)
        nn.init.xavier_uniform_(self.fc_sh1.weight)
        self.fc_sh1.bias.data.fill_(0.01)
        
        self.fc_h1h2 = nn.Linear(h1_size,h2_size)
        nn.init.xavier_uniform_(self.fc_h1h2.weight)
        self.fc_h1h2.bias.data.fill_(0.01)
        
        self.fc_h2a = nn.Linear(h2_size,action_size)
        nn.init.xavier_uniform_(self.fc_h2a.weight)
        self.fc_h2a.bias.data.fill_(0.01)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
    def forward(self,state):
#        action_score = self.net(state)
        h1 = self.relu(self.fc_sh1(state))
        h2 = self.relu(self.fc_h1h2(h1))
        action_score = self.fc_h2a(h2)
        action_score = self.softmax(action_score)
        return action_score

class Critic(nn.Module):
    def __init__(self):
        super(Critic,self).__init__()
#        self.net = nn.Sequential(
#                nn.Linear(4,200),
#                nn.ReLU(),
#                nn.Linear(200,1)
#                )
        self.fc_sah1 = nn.Linear(4,100)
        nn.init.xavier_uniform_(self.fc_sah1.weight)
        self.fc_sah1.bias.data.fill_(0.01)
        
        self.fc_h1h2 = nn.Linear(100,50)
        nn.init.xavier_uniform_(self.fc_h1h2.weight)
        self.fc_h1h2.bias.data.fill_(0.01)
        
        self.fc_h2q = nn.Linear(50,1)
        nn.init.xavier_uniform_(self.fc_h2q.weight)
        self.fc_h2q.bias.data.fill_(0.01)
        
        self.relu = nn.ReLU()
        
    def forward(self,s):
#        q = self.net(s)
        h1 = self.relu(self.fc_sah1(s))
        h2 = self.relu(self.fc_h1h2(h1))
        q = self.fc_h2q(h2)
        return q
        
pi_target = Actor().to(device)
q_target = Critic().to(device)

optim_pi_target = optim.RMSprop(pi_target.parameters(),lr=1e-4)
optim_q_target = optim.RMSprop(q_target.parameters(),lr=2e-4)

Huber_loss = nn.SmoothL1Loss()
Mseloss = nn.MSELoss()

def sim():
    batch = []
    q_batch = []
    state_batch = []
    state_next_batch = []
    a_batch = []
    done_batch = []
    reward_batch = []
    state = env.reset()
    done = False  
    while not done:
        env.render()
        action_score = pi_target(torch.from_numpy(state).float().to(device)).data
        p = Categorical(action_score)
        action = p.sample().cpu().numpy()
        state_next, reward, done, info = env.step(action)
        a_batch.append(action)
        if done ==True:
            q_next = 0.
        else:    
            q_next = q_target(torch.from_numpy(state).float().to(device)).data
        q_batch.append(reward+gamma*q_next)
        state_batch.append(state)
        state_next_batch.append(state_next)
        done_batch.append(done)
        reward_batch.append(reward)
        state = state_next
    q_mean = np.mean(q_batch)
    s_batch = np.array(state_batch)
    s_next_batch = np.array(state_next_batch)
    a_batch = np.array(a_batch)
    d_batch = np.array(done_batch)
    r_batch = np.array(reward_batch)
    batch = s_batch,s_next_batch,a_batch,d_batch,r_batch
    return batch,q_mean



duration = []

for i in range(3000):
    counter = 1
    batch,q_mean = sim()
    s_batch,s_next_batch,a_batch,d_batch,r_batch = batch
    if s_batch.shape[0]<mini_batch_size:
        s_mini_batch = s_batch
        s_next_mini_batch = s_next_batch
        a_mini_batch = a_batch
        d_mini_batch = d_batch
        r_mini_batch = r_batch
        
    else:
        mask = np.random.choice(np.arange(s_batch.shape[0]),mini_batch_size)
        s_mini_batch = s_batch[mask]
        s_next_mini_batch = s_next_batch[mask]
        a_mini_batch = a_batch[mask]
        d_mini_batch = d_batch[mask]
        r_mini_batch = r_batch[mask]
        
    action_score = pi_target(torch.from_numpy(s_mini_batch).float().to(device))
    a_mini_batch = a_mini_batch.reshape((a_mini_batch.shape[0],1))
    p_action = action_score.gather(1,torch.LongTensor(torch.from_numpy(a_mini_batch).data).to(device))
    r_mini_batch = torch.from_numpy(r_mini_batch).float().to(device)

    '''
        update Critic
    '''
    optim_q_target.zero_grad()
    q = q_target(torch.from_numpy(s_mini_batch).float().to(device))
    q_next = q_target(torch.from_numpy(s_next_mini_batch).float().to(device)).detach()
    for j in range(q.size()[0]):
        if d_mini_batch[j] == True:
            q_next[j] = torch.Tensor([0.]).float().to(device)
    q_delta = q-gamma*q_next-r_mini_batch
    q_loss = q_delta.pow(2).sum()
    q_loss.backward()
#    for param in q_target.parameters():
#        param.grad.data.clamp_(-1,1)
    optim_q_target.step()
    
    '''
    updata Actor
    '''
    optim_pi_target.zero_grad()
    
    q_sum = (-(gamma*q_next+r_mini_batch.data-q.detach())*torch.log(p_action)).sum()
#    q_sum = (-(gamma*q_next+r_mini_batch.data-q_mean)*torch.log(p_action)).sum()
    q_sum.backward()
#    for param in pi_target.parameters():
#        param.grad.data.clamp_(-1,1)
    optim_pi_target.step()

    counter = s_batch.shape[0]
    print('duration:',counter)
    if i%50==0:
        board.add_scalar('duration_mean',np.mean(duration[i-50:i-1]),i)
        board.add_scalar('duration_var',np.var(duration[i-50:i-1]),i)
#        board.add_histogram('actor net 1 weight',pi_target.fc_sh1.state_dict()['weight'].cpu().numpy(),i)
#        board.add_histogram('actor net 2 weight',pi_target.fc_h1h2.state_dict()['weight'].cpu().numpy(),i)
#        board.add_histogram('actor net 3 weight',pi_target.fc_h2h3.state_dict()['weight'].cpu().numpy(),i)
#        board.add_histogram('actor net 3 weight',pi_target.fc_h2a.state_dict()['weight'].cpu().numpy(),i)
    duration.append(counter)
    print(i,' th update')

torch.save(pi_target,'pi_target.pkl')
torch.save(q_target,'q_target.pkl')

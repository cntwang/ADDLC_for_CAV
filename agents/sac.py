
from collections import deque
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def network(actions=81):
    return nn.Sequential(nn.Linear(38,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,actions))


class ThreeStepReplay:
    def __init__(self,capacity=100000,seed=60):
        self.capacity=capacity
        self.obs=np.zeros((capacity,38),np.float32)
        self.next_obs=np.zeros_like(self.obs)
        self.actions=np.zeros(capacity,np.int64)
        self.rewards=np.zeros((capacity,3),np.float32)
        self.discount=np.zeros(capacity,np.float32)
        self.done=np.zeros(capacity,np.float32)
        self.pending=deque()
        self.pos=self.size=0
        self.rng=np.random.default_rng(seed)

    def add(self,obs,action,reward,next_obs,done):
        self.pending.append((obs.copy(),action,reward,next_obs.copy(),done))
        if len(self.pending)==3:
            self.commit()
        if done:
            while self.pending:
                self.commit()

    def commit(self):
        row=self.pos
        first=self.pending[0]
        final=self.pending[-1]
        self.obs[row]=first[0]
        self.actions[row]=first[1]
        self.rewards[row]=0.
        for i,transition in enumerate(self.pending):
            self.rewards[row,i]=transition[2]
        self.discount[row]=.99**len(self.pending)
        self.next_obs[row]=final[3]
        self.done[row]=float(final[4])
        self.pending.popleft()
        self.pos=(self.pos+1)%self.capacity
        self.size=min(self.size+1,self.capacity)

    def sample(self,vec,batch=1024):
        ids=self.rng.integers(0,self.size,batch)
        
        
        normalized=vec.normalize_reward(self.rewards[ids])
        rewards=(normalized*np.array([1.,.99,.99**2],np.float32)).sum(axis=1)
        return [torch.as_tensor(x) for x in (self.obs[ids],self.actions[ids],rewards,
            self.next_obs[ids],self.done[ids],self.discount[ids])]


class DiscreteSAC:
    def __init__(self,seed=60,actions=81):
        torch.manual_seed(seed)
        self.actor=network(actions)
        self.q1,self.q2=network(actions),network(actions)
        self.target1,self.target2=network(actions),network(actions)
        self.target1.load_state_dict(self.q1.state_dict())
        self.target2.load_state_dict(self.q2.state_dict())
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=3e-4)
        self.critic_opt=torch.optim.Adam([*self.q1.parameters(),*self.q2.parameters()],lr=3e-4)
        self.log_alpha=torch.zeros((),requires_grad=True)
        self.alpha_opt=torch.optim.Adam([self.log_alpha],lr=1e-4)
        self.target_entropy=.5*np.log(actions)

    @torch.no_grad()
    def predict(self,obs,deterministic=False):
        logits=self.actor(torch.as_tensor(obs,dtype=torch.float32))
        if deterministic:
            return int(logits.argmax(dim=-1).item())
        return int(torch.distributions.Categorical(logits=logits).sample().item())

    def update(self,data):
        obs,actions,rewards,next_obs,done,discount=data
        alpha=self.log_alpha.exp().detach()
        with torch.no_grad():
            lp=F.log_softmax(self.actor(next_obs),dim=1)
            prob=lp.exp()
            next_q=torch.minimum(self.target1(next_obs),self.target2(next_obs))
            value=(prob*(next_q-alpha*lp)).sum(dim=1)
            target=rewards+discount*(1-done)*value
        q1=self.q1(obs).gather(1,actions[:,None]).squeeze(1)
        q2=self.q2(obs).gather(1,actions[:,None]).squeeze(1)
        critic_loss=.5*(F.mse_loss(q1,target)+F.mse_loss(q2,target))
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        lp=F.log_softmax(self.actor(obs),dim=1)
        prob=lp.exp()
        with torch.no_grad():
            q=torch.minimum(self.q1(obs),self.q2(obs))
        actor_loss=(prob*(alpha*lp-q)).sum(dim=1).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        entropy=-(prob*lp).sum(dim=1).detach().mean()
        alpha_loss=self.log_alpha*(entropy-self.target_entropy)
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            for target_net,source_net in ((self.target1,self.q1),(self.target2,self.q2)):
                for target_param,source_param in zip(target_net.parameters(),source_net.parameters()):
                    target_param.mul_(1-.001).add_(source_param,alpha=.001)
        return {'actor_loss':actor_loss.item(),'critic_loss':critic_loss.item(),
                'alpha':self.log_alpha.exp().item(),'entropy':entropy.item()}

    def save(self,path,steps):
        torch.save({'actor':self.actor.state_dict(),'q1':self.q1.state_dict(),'q2':self.q2.state_dict(),
            'target1':self.target1.state_dict(),'target2':self.target2.state_dict(),
            'log_alpha':self.log_alpha.detach(),'steps':steps,
            'actor_optimizer':self.actor_opt.state_dict(),'critic_optimizer':self.critic_opt.state_dict(),
            'alpha_optimizer':self.alpha_opt.state_dict()},path)

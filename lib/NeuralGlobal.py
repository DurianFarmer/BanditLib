import numpy as np
import torch
import random
import math
from .BaseAlg import BaseAlg

class NeuralGlobal(torch.nn.Module):
    def __init__(self, dim, hidden_dim, device):
        super(NeuralGlobal, self).__init__()
        stddev = 1 / math.sqrt(float(hidden_dim))
        self.linear_article = torch.nn.Linear(dim, hidden_dim)
        self.linear_user = torch.nn.Linear(5, hidden_dim)
        self.linear_end = torch.nn.Linear(hidden_dim, 1, bias=False)
        self.relu = torch.nn.ReLU()
        torch.nn.init.uniform_(self.linear_end.weight, a=0.0, b=stddev)
        self.total_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.to(device)

    def forward(self, user_vec, feature_vec):
        user_embed = self.linear_user(user_vec)
        article_embed = self.linear_article(feature_vec)
        return self.linear_end(self.relu(user_embed + article_embed))

class NeuralGlobalAlgorithm(BaseAlg):
    def __init__(self, arg_dict):
        BaseAlg.__init__(self, arg_dict)
        self.learner = NeuralGlobal(self.dimension, self.hidden_dim, self.device)
        self.learner.eval()
        self.U = self.lamdba * torch.ones(self.learner.total_param, device=self.device, dtype=torch.float)
        self.article_history = torch.empty(0, self.dimension, device=self.device, dtype=torch.float)
        self.user_history = torch.empty(0, 5, device=self.device, dtype=torch.float)
        self.click_history = torch.empty(0, device=self.device, dtype=torch.float)
        self.len = 0
        self.loss_func = torch.nn.MSELoss()

        self.path = './Dataset/Yahoo/YahooKMeansModel/10kmeans_model160.dat'
        self.user_feature = torch.from_numpy(np.genfromtxt(self.path, delimiter=' ')).to(device=self.device, dtype=torch.float)


    def decide(self, pool_articles, userID, k=1):
        select_article = None
        select_score = None
        user_vec = self.user_feature[userID].view(1, -1)
        for x in pool_articles:
            self.learner.zero_grad()
            score = self.learner(user_vec, torch.from_numpy(x.contextFeatureVector[:self.dimension]).view(1, -1).to(self.device, torch.float32))
            score.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() if p.requires_grad else torch.tensor([], device=torch.device('cuda'))for p in self.learner.parameters()])
            score_UCB = self.alpha * torch.sum(g * g / self.U) + score.detach()
            if select_article is None or select_score < score_UCB:
                select_article = x
                select_score = score_UCB
        return [select_article]

    def updateParameters(self, articlePicked, click, userID):
        # first update covariance matrix
        x = torch.from_numpy(articlePicked.contextFeatureVector[:self.dimension]).view(1, -1).to(self.device, torch.float)
        user_vec = self.user_feature[userID].view(1, -1)
        click_tensor = torch.tensor([click], dtype=torch.float, device=self.device)
        self.learner.zero_grad()
        self.learner(user_vec, x).backward()
        g = torch.cat([p.grad.flatten().detach() if p.requires_grad else torch.tensor([], device=torch.device('cuda'))for p in self.learner.parameters()])
        self.U += g * g
        # update the buffer
        if click == 1 or random.random() < 0.05:
            if self.len < self.sz:
                self.click_history = torch.cat((self.click_history, click_tensor))
                self.user_history = torch.cat((self.user_history, user_vec))
                self.article_history = torch.cat((self.article_history, x))
                self.len += 1
            else:
                self.click_history = torch.cat((self.click_history[1:], click_tensor))
                self.user_history = torch.cat((self.user_history[1:], user_vec))
                self.article_history = torch.cat((self.article_history[1:], x))

            # update the network
            optim = torch.optim.SGD(self.learner.parameters(), lr=self.lr, weight_decay=self.lamdba / self.len)
            self.learner.train()
            for _ in range(self.iter):
                optim.zero_grad()
                pred = self.learner(self.user_history, self.article_history).view(-1)
                loss = self.loss_func(pred, self.click_history)
                loss.backward()
                optim.step()
            print(loss.item(), optim.param_groups[0]['weight_decay'], torch.mean(self.click_history).item())
            self.learner.eval()

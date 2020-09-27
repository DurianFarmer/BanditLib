import numpy as np
import torch
import random
import math

from util_functions import vectorize
from Recommendation import Recommendation
from BaseAlg import BaseAlg

class NeuralUCB(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, thres, lr, decay, iter):
        super(NeuralUCB, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, 1)
        # self.linear = torch.nn.Linear(input_dim, 1, bias=False)
        self.relu = torch.nn.ReLU()
        self.loss_func = torch.nn.MSELoss()
        self.thres = thres
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=lr, weight_decay=decay)
        self.decay = decay
        self.iter = iter
        self.total_param = sum(p.numel()
                               for p in self.parameters() if p.requires_grad)

    def forward(self, feature_vec):
        # return self.linear(feature_vec)
        return self.linear2(self.relu(self.linear1(feature_vec)))

    def forward_calc_g(self, feature_vec):
        # Forward and calculate the gradient vector, only 1 size could be fed in
        self.zero_grad()
        score = self.forward(feature_vec.view(1, -1))
        score.backward(retain_graph=True)
        g = torch.cat([
            p.grad.flatten().detach() if p.requires_grad else torch.tensor(
                [], device=torch.device('cuda'))
            for p in self.parameters()
        ])
        return score.view(-1).detach(), g

    def update_model(self, feature_vec, clicks):
        # print self.optimizer.param_groups[0]['weight_decay'], clicks.shape[0]
        self.optimizer.param_groups[0]['weight_decay'] = self.decay / clicks.shape[0]
        self.train()
        prev_loss = float('inf')
        early_stopping = 0
        i = 0
        while i < self.iter:
            i += 1
            self.optimizer.zero_grad()
            pred = self.forward(feature_vec).view(-1)
            loss = self.loss_func(pred, clicks)
            loss.backward()
            self.optimizer.step()
            if (loss - prev_loss).abs() < self.thres:
                early_stopping += 1
            else:
                early_stopping = 0
            if early_stopping >= 5:
                break
            prev_loss = loss.item()
        print feature_vec.shape, clicks.shape, pred.shape
        self.eval()
        return loss.item()


class NeuralUCBUserStruct:
    def __init__(self, input_dim, hidden_dim, thres, device, lr, decay, iter, sz, lamdba, nu):
        self.device = device
        self.learner = NeuralUCB(input_dim, hidden_dim, thres, lr, decay, iter)
        self.learner.to(device=self.device)
        self.learner.eval()
        self.feature_history = torch.empty(0, input_dim, device=self.device)
        self.click_history = torch.empty(0, device=self.device)
        self.buffer_size = sz
        self.current_size = 0
        self.U = lamdba * \
            torch.ones(self.learner.total_param, device=self.device)
        self.lamdba = lamdba
        self.nu = nu

    def updateParameters(self, feature_vec, click):
        # update the sampling buffer
        # print self.feature_history.shape, self.click_history.shape
        if self.current_size < self.buffer_size:
            self.click_history = torch.cat((self.click_history, click))
            self.feature_history = torch.cat(
                (self.feature_history, feature_vec.view(1, -1)))
            self.current_size += 1
        else:
            self.click_history = torch.cat((self.click_history[1:], click))
            self.feature_history = torch.cat((
                self.feature_history[1:], feature_vec.view(1, -1)))

        return self.learner.update_model(self.feature_history, self.click_history)

    def decide(self, feature_pool):
        score_g = torch.cat([torch.cat(self.learner.forward_calc_g(
            x.view(-1))).view(1, -1) for x in feature_pool])
        UCB = torch.sqrt(torch.sum(score_g[:, 1:] * score_g[:, 1:] / self.U, dim=1))
        # print UCB, score_g[:, 0]
        # print '-' * 5
        arm = torch.argmax(self.nu * UCB + score_g[:, 0]).item()
        self.U += score_g[arm, 1:] * score_g[arm, 1:]
        return arm


class NeuralUCBAlgorithm(BaseAlg):
    def __init__(self, arg_dict):
        BaseAlg.__init__(self, arg_dict)
        print arg_dict
        torch.set_num_threads(8)
        self.users = [
            NeuralUCBUserStruct(self.dimension, self.hidden_layer_dimension, self.thres, self.device, self.lr, self.decay, self.iter, self.sz, self.lamdba, self.nu) for _ in range(self.n_users)
        ]

    def decide(self, pool_articles, userID, k=1):
        X = torch.cat([torch.from_numpy(x.contextFeatureVector[:self.dimension]).view(1, -1).to(self.device, torch.float32)
                       for x in pool_articles])
        # print X.shape, self.users[userID].click_history.shape
        return [pool_articles[self.users[userID].decide(X)]]

    def updateParameters(self, articlePicked, click, userID):
        return self.users[userID].updateParameters(
            torch.from_numpy(articlePicked.contextFeatureVector[:self.dimension]).to(
                self.device, torch.float32),
            torch.tensor([click], device=self.device, dtype=torch.float32)
        )
class NeuralUCB1Algorithm(BaseAlg):
    def __init__(self, arg_dict):
        BaseAlg.__init__(self, arg_dict)
        print arg_dict
        torch.set_num_threads(8)
        self.users = [
            NeuralUCBUserStruct(self.dimension, self.hidden_layer_dimension, self.thres, self.device, self.lr, self.decay, self.iter, self.sz, self.lamdba, self.nu) for _ in range(self.n_users)
        ]

    def decide(self, pool_articles, userID, k=1):
        X = torch.cat([torch.from_numpy(x.contextFeatureVector[:self.dimension]).view(1, -1).to(self.device, torch.float32)
                       for x in pool_articles])
        # print X.shape, self.users[userID].click_history.shape
        return [pool_articles[self.users[userID].decide(X)]]

    def updateParameters(self, articlePicked, click, userID):
        return self.users[userID].updateParameters(
            torch.from_numpy(articlePicked.contextFeatureVector[:self.dimension]).to(
                self.device, torch.float32),
            torch.tensor([click], device=self.device, dtype=torch.float32)
        )
class NeuralUCB2Algorithm(BaseAlg):
    def __init__(self, arg_dict):
        BaseAlg.__init__(self, arg_dict)
        print arg_dict
        torch.set_num_threads(8)
        self.users = [
            NeuralUCBUserStruct(self.dimension, self.hidden_layer_dimension, self.thres, self.device, self.lr, self.decay, self.iter, self.sz, self.lamdba, self.nu) for _ in range(self.n_users)
        ]

    def decide(self, pool_articles, userID, k=1):
        X = torch.cat([torch.from_numpy(x.contextFeatureVector[:self.dimension]).view(1, -1).to(self.device, torch.float32)
                       for x in pool_articles])
        # print X.shape, self.users[userID].click_history.shape
        return [pool_articles[self.users[userID].decide(X)]]

    def updateParameters(self, articlePicked, click, userID):
        return self.users[userID].updateParameters(
            torch.from_numpy(articlePicked.contextFeatureVector[:self.dimension]).to(
                self.device, torch.float32),
            torch.tensor([click], device=self.device, dtype=torch.float32)
        )
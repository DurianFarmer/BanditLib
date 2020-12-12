import torch
import numpy as np
import random
import math
from .BaseAlg import BaseAlg
import time
from backpack import backpack, extend
from backpack.extensions import BatchGrad
import io

class NeuMF(torch.nn.Module):
    def __init__(self, user_dim, item_dim, mf_dim, mlp_dim, lr):
        super(NeuMF, self).__init__()
        self.user_dim, self.item_dim = user_dim, item_dim
        self.mf_dim, self.mlp_dim = mf_dim, mlp_dim

        self.MF_Embedding_User_hidden = torch.nn.Linear(user_dim, mf_dim)
        self.MF_Embedding_Item_hidden = torch.nn.Linear(item_dim, mf_dim)
        # self.MF_Embedding_User = torch.nn.Linear(2 * mf_dim, mf_dim)
        # self.MF_Embedding_Item = torch.nn.Linear(2 * mf_dim, mf_dim)


        self.MLP_Embedding_User_hidden = torch.nn.Linear(user_dim, mlp_dim[0] // 2)
        self.MLP_Embedding_Item_hidden = torch.nn.Linear(item_dim, mlp_dim[0] // 2)
        # self.MLP_Embedding_User = torch.nn.Linear(mlp_dim[0], mlp_dim[0] // 2)
        # self.MLP_Embedding_Item = torch.nn.Linear(mlp_dim[0], mlp_dim[0] // 2)

        layers = []
        for idx in range(len(mlp_dim) - 1):
            layers.append(torch.nn.Linear(mlp_dim[idx], mlp_dim[idx + 1]))
            layers.append(torch.nn.ReLU())

        self.MLP_layers = torch.nn.Sequential(*layers)
        self.prediction = torch.nn.Linear(mf_dim + mlp_dim[-1], 1)

        self.lr = lr
        self.optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

        self.total_param = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, user_context, item_context):
        mf_user_latent = self.MF_Embedding_User_hidden(user_context)
        # mf_user_latent = self.relu(mf_user_latent)
        # mf_user_latent = self.MF_Embedding_User(mf_user_latent)

        mf_item_latent = self.MF_Embedding_Item_hidden(item_context)
        # mf_item_latent = self.relu(mf_item_latent)
        # mf_item_latent = self.MF_Embedding_Item(mf_item_latent)

        mf_vector = mf_user_latent * mf_item_latent

        mlp_user_latent = self.MLP_Embedding_User_hidden(user_context)
        # mlp_user_latent = self.relu(mlp_user_latent)
        # mlp_user_latent = self.MLP_Embedding_User(mlp_user_latent)

        mlp_item_latent = self.MLP_Embedding_Item_hidden(item_context)
        # mlp_item_latent = self.relu(mlp_item_latent)
        # mlp_item_latent = self.MLP_Embedding_Item(mlp_item_latent)

        mlp_vector = torch.cat((mlp_user_latent, mlp_item_latent), 1)
        mlp_vector = self.MLP_layers(mlp_vector)

        prediction_vector = torch.cat((mf_vector, mlp_vector), 1)
        return self.sigmoid(self.prediction(prediction_vector))


class DataLoader(torch.utils.data.Dataset):
    def __init__(self):
        self.user_history = []
        self.article_history = []
        self.click_history = []
        self.size = 65536 * 16
        self.grid = 64

    def push(self, user, article, click):
        self.user_history.append(user)
        self.article_history.append(article)
        self.click_history.append(click)
        if len(self.user_history) >= self.size:
            self.user_history = self.user_history[self.grid:]
            self.article_history = self.article_history[self.grid:]
            self.click_history = self.click_history[self.grid:]
            print(len(self.user_history))

    def __len__(self):
        return len(self.user_history)

    def __getitem__(self, idx):
        return {
            'user': self.user_history[idx],
            'article': torch.from_numpy(self.article_history[idx]).to(torch.float),
            'click': torch.tensor(self.click_history[idx], dtype=torch.float)
            }


class NeuMFYahooAlgorithm(BaseAlg):
    def __init__(self, arg_dict):
        BaseAlg.__init__(self, arg_dict)
        self.learner = extend(NeuMF(user_dim=5, item_dim=5, mf_dim=16, mlp_dim = [32, 16, 8], lr=1e-3).cuda())
        self.lossfunc = extend(torch.nn.BCELoss())

        self.path = './Dataset/Yahoo/YahooKMeansModel/10kmeans_model160.dat'
        self.user_feature = torch.from_numpy(np.genfromtxt(self.path, delimiter=' ')).to(dtype=torch.float).cuda()

        self.data = DataLoader()
        self.cnt = 0
        self.batch = 1024

        torch.set_num_threads(8)
        torch.set_num_interop_threads(8)

        self.lamdba = 1
        self.nu = 0.1
        self.U = self.lamdba * torch.ones((self.learner.total_param), dtype=torch.float).cuda()
        self.U1 = torch.zeros((self.learner.total_param), dtype=torch.float).cuda()
        self.g = None
        self.reg = None
        self.t1 = time.time()


    def decide(self, pool_articles, userID, k=1):
        self.a = len(pool_articles)
        
        user_vec = torch.cat(self.a*[self.user_feature[userID].view(1, -1)])
        article_vec = torch.cat([
            torch.from_numpy(x.contextFeatureVector[:self.dimension]).view(1, -1).to(torch.float32)
            for x in pool_articles])
        score = self.learner(user_vec, article_vec.cuda()).view(-1)
        sum_score = torch.sum(score)
        with backpack(BatchGrad()):
            sum_score.backward()
        
        grad = torch.cat([p.grad_batch.view(self.a, -1) for p in self.learner.parameters()], dim=1)
        sigma = torch.sqrt(torch.sum(grad * grad / self.U, dim=1))
        self.reg = self.nu * torch.mean(sigma).item()
        arm = torch.argmax(score + self.nu * sigma).item()
        self.g = grad[arm]
        return [pool_articles[arm]]

    def updateParameters(self, articlePicked, click, userID):
        if click == 1 or random.random() < 0.05:
            user_vec = self.user_feature[userID]
            article_vec = articlePicked.contextFeatureVector[:self.dimension]
            assert self.g is not None
            self.U1 += self.g * self.g

            self.data.push(user_vec, article_vec, click)
            self.cnt = (self.cnt + 1) % self.batch
            if self.cnt % self.batch == 0:
                self.learner.optim.param_groups[0]['weight_decay'] = self.lamdba / len(self.data)
                t2 = time.time() - self.t1
                t1 = time.time()
                dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch, shuffle=True, num_workers=0)
                # self.learner.train().cuda()

                loss_list = []
                early_cnt = 0

                for i in range(1000):
                    tot_loss = 0
                    for j, batch in enumerate(dataloader):
                        self.learner.optim.zero_grad()
                        u = batch['user'].cuda()
                        a = batch['article'].cuda()
                        c = batch['click'].cuda()
                        pred = self.learner(u, a).view(-1)
                        loss = self.lossfunc(pred, c)
                        tot_loss += loss.item()
                        loss.backward()
                        self.learner.optim.step()
                    # early stopping
                    if i != 0 and loss_list[-1] < tot_loss / (j + 1):
                        early_cnt += 1
                    else:
                        early_cnt = 0

                    if early_cnt >= 5:
                        break
                    loss_list.append(tot_loss / (j + 1))

                # self.learner.eval().cpu()
                self.U += self.U1
                self.U1 *= 0
                print('[{:.2f}, {:.2f} s]: loss: {:.3f}, data: {}, iterations: {}, Covar: {:.2e}, +: [{:.2e}, {:.2e}], -: [{:.2e}, {:.2e}]'.format(
                    t2, time.time() - t1, loss_list[-1], len(self.data), i + 1, self.reg, 
                    torch.mean(pred.detach()[c==1]).item(), torch.std(pred.detach()[c==1]).item(),
                    torch.mean(pred.detach()[c==0]).item(), torch.std(pred.detach()[c==0]).item()))
                self.t1 = time.time()

class NeuMFLastFMAlgorithm(BaseAlg):
    def __init__(self, arg_dict):
        BaseAlg.__init__(self, arg_dict)
        self.learner = extend(NeuMF(user_dim=25, item_dim=25, mf_dim=32, mlp_dim = [32, 16, 8], lr=1e-3).cuda())
        self.lossfunc = extend(torch.nn.BCELoss())

        self.path = './Dataset/LastFM.dat'
        self.user_feature = torch.from_numpy(np.genfromtxt(self.path, delimiter=' ')).to(dtype=torch.float).cuda()

        self.data = DataLoader()
        self.cnt = 0
        self.batch = 100

        torch.set_num_threads(8)
        torch.set_num_interop_threads(8)

        self.lamdba = 1
        self.nu = 1
        self.U = self.lamdba * torch.ones((self.learner.total_param), dtype=torch.float).cuda()
        self.U1 = torch.zeros((self.learner.total_param), dtype=torch.float).cuda()
        self.g = None
        self.reg = None
        self.t1 = time.time()


    def decide(self, pool_articles, userID, k=1):
        self.a = len(pool_articles)
        
        user_vec = torch.cat(self.a*[self.user_feature[userID - 1].view(1, -1)])
        article_vec = torch.cat([
            torch.from_numpy(x.contextFeatureVector[:self.dimension]).view(1, -1).to(torch.float32)
            for x in pool_articles])
        score = self.learner(user_vec, article_vec.cuda()).view(-1)
        sum_score = torch.sum(score)
        with backpack(BatchGrad()):
            sum_score.backward()
        
        grad = torch.cat([p.grad_batch.view(self.a, -1) for p in self.learner.parameters()], dim=1)
        sigma = torch.sqrt(torch.sum(grad * grad / self.U, dim=1))
        self.reg = self.nu * torch.mean(sigma).item()
        arm = torch.argmax(score + self.nu * sigma).item()
        self.g = grad[arm]
        return [pool_articles[arm]]

    def updateParameters(self, articlePicked, click, userID):
        if click == 1 or random.random() < 1:
            user_vec = self.user_feature[userID - 1]
            article_vec = articlePicked.contextFeatureVector[:self.dimension]
            assert self.g is not None
            self.U1 += self.g * self.g

            self.data.push(user_vec, article_vec, click)
            self.cnt = (self.cnt + 1) % self.batch
            if self.cnt % self.batch == 0:
                self.learner.optim.param_groups[0]['weight_decay'] = self.lamdba / len(self.data)
                t2 = time.time() - self.t1
                t1 = time.time()
                dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch, shuffle=True, num_workers=0)
                # self.learner.train().cuda()

                loss_list = []
                early_cnt = 0

                for i in range(1000):
                    tot_loss = 0
                    for j, batch in enumerate(dataloader):
                        self.learner.optim.zero_grad()
                        u = batch['user'].cuda()
                        a = batch['article'].cuda()
                        c = batch['click'].cuda()
                        pred = self.learner(u, a).view(-1)
                        loss = self.lossfunc(pred, c)
                        tot_loss += loss.item()
                        loss.backward()
                        self.learner.optim.step()
                    # early stopping
                    if i != 0 and loss_list[-1] < tot_loss / (j + 1):
                        early_cnt += 1
                    else:
                        early_cnt = 0

                    if early_cnt >= 5:
                        break
                    loss_list.append(tot_loss / (j + 1))

                # self.learner.eval().cpu()
                self.U += self.U1
                self.U1 *= 0
                print('[{:.2f}, {:.2f} s]: loss: {:.3f}, data: {}, iterations: {}, Covar: {:.2e}, +: [{:.2e}, {:.2e}], -: [{:.2e}, {:.2e}]'.format(
                    t2, time.time() - t1, loss_list[-1], len(self.data), i + 1, self.reg, 
                    torch.mean(pred.detach()[c==1]).item(), torch.std(pred.detach()[c==1]).item(),
                    torch.mean(pred.detach()[c==0]).item(), torch.std(pred.detach()[c==0]).item()))
                self.t1 = time.time()

class NeuMFDeliciousAlgorithm(BaseAlg):
    def __init__(self, arg_dict):
        BaseAlg.__init__(self, arg_dict)
        self.learner = extend(NeuMF(user_dim=25, item_dim=25, mf_dim=32, mlp_dim = [32, 16, 8], lr=1e-3).cuda())
        self.lossfunc = extend(torch.nn.BCELoss())

        self.path = './Dataset/delicious.dat'
        self.user_feature = []
        with open(self.path, 'r') as f:
            for line in f:
                if line.strip():
                    self.user_feature.append(torch.from_numpy(np.genfromtxt(io.StringIO(line), delimiter=" ")).to(dtype=torch.float).cuda())
                else:
                    self.user_feature.append(None)
        print(len(self.user_feature))
        self.data = DataLoader()
        self.cnt = 0
        self.batch = 100

        torch.set_num_threads(8)
        torch.set_num_interop_threads(8)

        self.lamdba = 1
        self.nu = 1
        self.U = self.lamdba * torch.ones((self.learner.total_param), dtype=torch.float).cuda()
        self.U1 = torch.zeros((self.learner.total_param), dtype=torch.float).cuda()
        self.g = None
        self.reg = None
        self.t1 = time.time()


    def decide(self, pool_articles, userID, k=1):
        if self.user_feature[userID - 1] is None:
            print("isolated user, random choice")
            return [pool_articles[np.random.choice(len(pool_articles), size=1, replace=False)[0]]]
            
        self.a = len(pool_articles)
        
        user_vec = torch.cat(self.a*[self.user_feature[userID - 1].view(1, -1)])
        article_vec = torch.cat([
            torch.from_numpy(x.contextFeatureVector[:self.dimension]).view(1, -1).to(torch.float32)
            for x in pool_articles])
        score = self.learner(user_vec, article_vec.cuda()).view(-1)
        sum_score = torch.sum(score)
        with backpack(BatchGrad()):
            sum_score.backward()
        
        grad = torch.cat([p.grad_batch.view(self.a, -1) for p in self.learner.parameters()], dim=1)
        sigma = torch.sqrt(torch.sum(grad * grad / self.U, dim=1))
        self.reg = self.nu * torch.mean(sigma).item()
        arm = torch.argmax(score + self.nu * sigma).item()
        self.g = grad[arm]
        return [pool_articles[arm]]

    def updateParameters(self, articlePicked, click, userID):
        if self.user_feature[userID - 1] is None:
            return
        if click == 1 or random.random() < 1:
            user_vec = self.user_feature[userID - 1]
            article_vec = articlePicked.contextFeatureVector[:self.dimension]
            assert self.g is not None
            self.U1 += self.g * self.g

            self.data.push(user_vec, article_vec, click)
            self.cnt = (self.cnt + 1) % self.batch
            if self.cnt % self.batch == 0:
                self.learner.optim.param_groups[0]['weight_decay'] = self.lamdba / len(self.data)
                t2 = time.time() - self.t1
                t1 = time.time()
                dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch, shuffle=True, num_workers=0)
                # self.learner.train().cuda()

                loss_list = []
                early_cnt = 0

                for i in range(1000):
                    tot_loss = 0
                    for j, batch in enumerate(dataloader):
                        self.learner.optim.zero_grad()
                        u = batch['user'].cuda()
                        a = batch['article'].cuda()
                        c = batch['click'].cuda()
                        pred = self.learner(u, a).view(-1)
                        loss = self.lossfunc(pred, c)
                        tot_loss += loss.item()
                        loss.backward()
                        self.learner.optim.step()
                    # early stopping
                    if i != 0 and loss_list[-1] < tot_loss / (j + 1):
                        early_cnt += 1
                    else:
                        early_cnt = 0

                    if early_cnt >= 5:
                        break
                    loss_list.append(tot_loss / (j + 1))

                # self.learner.eval().cpu()
                self.U += self.U1
                self.U1 *= 0
                print('[{:.2f}, {:.2f} s]: loss: {:.3f}, data: {}, iterations: {}, Covar: {:.2e}, +: [{:.2e}, {:.2e}], -: [{:.2e}, {:.2e}]'.format(
                    t2, time.time() - t1, loss_list[-1], len(self.data), i + 1, self.reg, 
                    torch.mean(pred.detach()[c==1]).item(), torch.std(pred.detach()[c==1]).item(),
                    torch.mean(pred.detach()[c==0]).item(), torch.std(pred.detach()[c==0]).item()))
                self.t1 = time.time()

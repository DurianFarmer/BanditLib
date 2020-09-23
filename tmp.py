class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, threshold, learning_rate, n=1):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim, bias=True)
        self.linear2 = torch.nn.Linear(hidden_dim, 1, bias=True)
        self.relu = torch.nn.ReLU(inplace=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.linear2.weight.data.fill_(1)
        self.linear1.weight.data.fill_(1)
        self.linear1.bias.data.fill_(1)
        self.linear2.bias.data.fill_(1)
        self.n = n
        self.loss_function = torch.nn.MSELoss()
        self.threshold = threshold
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=learning_rate, weight_decay=1e-3)

    def forward(self, article_FeatureVector):
        output_layer_one = self.linear1(article_FeatureVector)
        output_layer_one = self.relu(output_layer_one)
        output_layer_two = self.linear2(output_layer_one)
        pred = self.relu(output_layer_two)
        return pred

    def update_model(self, article_FeatureVectors, clicks, perturb_scale=0):
        self.train()
        # UPDATE: multiple updates till converge
        prev_loss = float('inf')
        i = 0
        while i < 100:
            i += 1
            self.optimizer.zero_grad()
            pred = self.forward(article_FeatureVectors)
            loss = self.loss_function(pred, clicks)
            loss.backward()  # computes gradient
            self.optimizer.step()  # updates weights
            # end while
            if (loss - prev_loss).abs() < self.threshold:  # please set an appropriate threshold
                break
            prev_loss = loss
        self.eval()
        return prev_loss

class MLPUserStruct:
    def __init__(self, input_dim, hidden_dim, threshold, device, learning_rate, perturb_type='normal', n=1):
        self.device = device
        self.mlp = MLP(input_dim, hidden_dim, threshold, learning_rate, n=n)
        self.mlp.to(device=self.device)
        self.mlp.eval()
        self.article_FeatureVectors = torch.empty(
            0, input_dim).to(device=self.device)
        self.articles = []
        self.clicks_list = []
        self.history = []
        self.clicks = torch.empty(0, 1).to(device=self.device)

    def updateParameters(self, article_FeatureVector, click):
        article_FeatureVector = torch.tensor(
            [article_FeatureVector]).float().to(device=self.device)
        click = torch.tensor([[click]]).float().to(device=self.device)
        return self.mlp.update_model(article_FeatureVector, click)

    def getProb(self, article_FeatureVector):
        return self.mlp(torch.from_numpy(article_FeatureVector).float().to(device=self.device))


# class NeuralUCBAlgorithm(BaseAlg):
#     def __init__(self, arg_dict):  # n is number of users
#         BaseAlg.__init__(self, arg_dict)
#         device = torch.device('cpu')
#         self.users = []
#         for i in range(self.n_users):
#             self.users.append(MLPUserStruct(
#                 self.dimension, self.hidden_layer_dimension, self.threshold, device, self.learning_rate))

#     def decide(self, pool_articles, userID, k=1):
#         maxPTA = float('-inf')
#         articlePicked = None

#         for x in pool_articles:
#             x_pta = self.users[userID].getProb(
#                 x.contextFeatureVector[:self.dimension])
#             if maxPTA < x_pta:
#                 articlePicked = x
#                 maxPTA = x_pta

#         return [articlePicked]

#     def updateParameters(self, articlePicked, click, userID):
#         return self.users[userID].updateParameters(articlePicked.contextFeatureVector[:self.dimension], click)

    
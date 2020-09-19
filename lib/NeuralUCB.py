import numpy as np
from util_functions import vectorize
from Recommendation import Recommendation
from BaseAlg import BaseAlg
from LinUCB import LinUCBUserStruct


class NeuralUCBAlgorithm(BaseAlg):
    def __init__(self, arg_dict, init="zero"):  # n is number of users
        # hyper parameters are set in basealg, see LN 7-8 on BaseAlg.py
        BaseAlg.__init__(self, arg_dict)
        self.users = []
        # algorithm have n users, each user has a user structure
        for i in range(arg_dict['n_users']):
            self.users.append(LinUCBUserStruct(
                arg_dict['dimension'], arg_dict['lambda_'], init))

    def decide(self, pool_articles, userID, k=1):
        # MEAN
        art_features = np.empty([len(pool_articles), len(
            pool_articles[0].contextFeatureVector[:self.dimension])])
        for i in range(len(pool_articles)):
            art_features[i, :] = pool_articles[i].contextFeatureVector[:self.dimension]
        user_features = self.users[userID].UserTheta
        mean_matrix = np.dot(art_features, user_features)

        # VARIANCE
        var_matrix = np.sqrt(
            np.dot(np.dot(art_features, self.users[userID].AInv), art_features.T).clip(0))
        pta_matrix = mean_matrix + self.alpha*np.diag(var_matrix)

        pool_positions = np.argsort(pta_matrix)[(k*-1):]
        articles = []
        for i in range(k):
            articles.append(pool_articles[pool_positions[i]])
        return articles

    # def getProb(self, pool_articles, userID):
    #     means = []
    #     vars = []
    #     for x in pool_articles:
    #         x_pta, mean, var = self.users[userID].getProb_plot(
    #             self.alpha, x.contextFeatureVector[:self.dimension])
    #         means.append(mean)
    #         vars.append(var)
    #     print 'This is USED'
    #     return means, vars

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(
            articlePicked.contextFeatureVector[:self.dimension], click)

    def getTheta(self, userID):
        return self.users[userID].UserTheta

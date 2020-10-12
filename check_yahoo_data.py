import numpy as np


def parseLine_ID(line):
    line = line.split("|")

    tim, articleID, click = line[0].strip().split(" ")
    tim, articleID, click = int(tim), int(articleID), int(click)

    userID = int(line[1].strip())

    pool_articles = [l.strip().split(" ") for l in line[2:]]
    pool_articles = np.array([[int(l[0])] + [float(x.split(':')[1]) for x in l[1:]] for l in pool_articles])
    return tim, articleID, click, userID, pool_articles

userInfo = {}
for day in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']:
    filename = "ydata-fp-td-clicks-v1_0.200905{}.160.userID".format(day)
    print("Day {}".format(day))
    with open(filename, 'r') as f:
        # reading file line ie observations running one at a time
        for i, line in enumerate(f, 1):
            tim, article_chosen, click, currentUserID, pool_articles = parseLine_ID(line)

            if currentUserID not in userInfo.keys():
                userInfo[currentUserID] = set()

            for article in pool_articles:
                article_featureVector = np.asarray(article[1:6])
                if len(article_featureVector) == 5:
                    article_id = int(article[0])
                    userInfo[currentUserID].add(article_id)

            if i % 10000 == 0:
                print("Line: {}".format(i))


for userid in userInfo.keys():
    print(userid, len(userInfo[userid]))
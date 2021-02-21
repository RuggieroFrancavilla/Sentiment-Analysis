from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words
import csv 
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, confusion_matrix, plot_confusion_matrix
from imblearn.pipeline import Pipeline,make_pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt

dataset = [[],[]]
with open("development.csv", encoding = 'utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in csv.reader(f):
        dataset[0].append(row[0])
        dataset[1].append(row[1])

review_train, review_test, label_train, label_test = train_test_split(dataset[0], dataset[1] , test_size=0.20, random_state = 0)

dataset_len = len(dataset[0])
reviews_mv = sum([1 for d in dataset[0] if d == ''])
labels_mv = sum([1 for d in dataset[1] if d == ''])

labels_dict = Counter(label_test)
pos_percentage = "{0:.2f}".format(100*(list(labels_dict.values())[0]/sum(list(labels_dict.values()))))
neg_percentage = "{0:.2f}".format(100*(list(labels_dict.values())[1]/sum(list(labels_dict.values()))))

print(f"The length of the dataset is {dataset_len}")
print(f"The dataset presents {reviews_mv} missing values in the reviews and {labels_mv} missing values in the labels")
print(f"The dataset presents: {labels_dict},so a ratio of positive to negative class of {pos_percentage} / {neg_percentage}")

plt.pie([float(v) for v in labels_dict.values()], labels=[k for k in labels_dict], autopct='%1.1f%%', startangle=90)

def preprocessor(document):
    special = string.punctuation+'‘’“”•…$£´ı¨€☺¹¿½²°🤷‍♀🎉👍🏻♥️😀😙😆👌🤗😘😁😂º🥺😱😖⭐🤮😤🤔😄😬😳🤭😉😢🐳😎✌🏼😕🙄🍦💪🍍🍊▶😅�😭😦😰😔😡🙅 ♂😃❤️👋🔝🚶🤐😃🙋🙈😊👎👠😃😨👜👏😞😲🍰😜😑😍'+'樓下朝食付きデ泊しまラックスルームに的接待會在位置눉上抽煙，導致整棟也都是味因為通風房間非連行李箱小常不好無法打開但離車站近所以唯或奉要入住這家討決定厭決た利便性を考慮て選んだのくのバポレト乗り場は工事で閉鎖され遠くから歩たタフはアジア系のハウキパ外愛想で笑顔がありせんフテは夏外ロズですジャグジは壊れて修理ヶ月かかるそうです水漏れで変わった部屋のジャグジからは黒いカビが噴出てた層階の部屋の工事でパイプを破って汚水が部屋落ちてたのサモンは美かったですがロワサンは最悪良い点コヒの料サビ観光地い総評宿た日はタフの対応とマナよって全く落ち着くことが出来せん特アメリカの有名な俳似た夜のタフはアジア人を完全見ていた彼ら欠けている物価格表を見ると定勸一優點空否則夠痛苦氣很糟。'
    num = [0,1,2,3,4,5,6,7,8,9]
    for i in range(len(document)):
        for punct in special:
            document[i] = document[i].replace(punct, " ")
        for n in num:
            document[i] = document[i].replace(str(n), " ")            
        document[i] = ' '.join(document[i].split())
        document[i] = ' '.join( [w for w in document[i].split() if len(w)>1] )
    return document

review_train = preprocessor(review_train)

tfidfvectorizer = TfidfVectorizer(stop_words=get_stop_words('it'),ngram_range=(1,2))
random = RandomOverSampler(random_state= 0)
selectpercentile = SelectPercentile()
multinomialnb = MultinomialNB()

pipeline = make_pipeline(tfidfvectorizer,
                         random,
                         selectpercentile,
                         multinomialnb)
parameters = {
    'tfidfvectorizer__min_df': (3,4,5),    
    'tfidfvectorizer__max_df': (0.03, 0.04, 0.05, 0.06, 0.1, 0.3),
    'tfidfvectorizer__norm':('l1','l2'),
    'selectpercentile__score_func': (f_classif,chi2),
    'selectpercentile__percentile': (20,23,25),
    'multinomialnb__alpha': (0.5,0.6,0.7,0.8,0.9,1)
}

     
CV = GridSearchCV(pipeline, parameters, scoring = 'f1_weighted', verbose = 10, return_train_score = True)

CV.fit(review_train,label_train)
    
print('Best score and parameter combination = ')

print(CV.best_score_)    
print(CV.best_params_)
print('')
print(CV.cv_results_)


review_test = preprocessor(review_test)
predict = CV.predict(review_test)
f1 = f1_score(label_test,predict, average ="weighted")
print(f"The f1_score for the test set is:{f1}")

# Word Cloud
feat = np.asarray(list(sorted(tfidfvectorizer.vocabulary_.keys())))
mask = np.asarray(selectpercentile.get_support())
features = feat[mask]
score = selectpercentile.scores_

d1 = {}
for f,s in zip(features,score):
    d1[f] = s
ordered = list(sorted(d1.items(),key= lambda x:x[1],reverse = True))
best = ordered[0:1000]
dictionary = dict((x, y) for x, y in best)

wc=WordCloud(width=1200, height=800, background_color="white")
wordcloud=wc.generate_from_frequencies(dictionary)
fig, ax=plt.subplots(figsize=(10,6), dpi=100)
fig.suptitle('The first 1000 features used', fontsize=16)
ax.axis("off")
plt.imshow(wordcloud, interpolation='bilinear')

# Confusion Matrix
disp = plot_confusion_matrix(pipeline, review_test, label_test,
                             labels = ['pos','neg'],
                             display_labels=['pos','neg'],
                             cmap=plt.cm.Blues,
                             values_format='.0f')
plt.title('Confusion matrix')
plt.show()

evaluation = []
with open("evaluation.csv", encoding = 'utf8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in csv.reader(f):
        evaluation.append(row[0])
        
evaluation = preprocessor(evaluation)
predictions = CV.predict(evaluation)
with open('submission_file17.csv', mode='w') as submission_file:
    submission_writer = csv.writer(submission_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    submission_writer.writerow(['Id','Predicted'])
    [submission_writer.writerow([i,prediction]) for i,prediction in enumerate(predictions)]
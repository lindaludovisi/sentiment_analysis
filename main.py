"""
Created on Mon Jan 13 13:50:00 2020

@author: lindaludovisi
                          
    TODO: gestisci "all' - l'" in generale apostrofi --> FATTO
    TODO: gestisci "non + aggettivo" , trattali come unica feature --> FATTO
    modificato label "class"->"labels"
    TODO: gestisci 'pulito'
    TODO: gridsearch sui parametri di selectkbeast, linearsvc
             
"""


import pandas as pd

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import snowball 
from nltk.corpus import stopwords as sw
import re
from nltk.tokenize import word_tokenize

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


raw_df = pd.read_csv("dataset_winter_2020/development.csv") #raw_df is a dataframe

"""
    DATA EXPLORATION
    The different classes are balanced ?
  
"""
print(f"The dataset contains {len(raw_df)} TripAdvisor reviews")
print("Test Set:"% raw_df.columns, raw_df.shape, len(raw_df))

raw_df.head() #show the first records of the raw dataframe
raw_df.info() #there aren't missing values

#here we store infos about the review and the sentiment in two different data structures
docu = raw_df["text"]
labels = raw_df["class"]

#here we change the name of the column "class" in "label" 
labels = pd.Series(labels)
labels = pd.DataFrame({'label' : labels})

#Percentage of Positive/Negative reviews
print("Positive: ", labels.label.value_counts()['pos']/len(raw_df)*100,"%")
print("Negative: ", labels.label.value_counts()['neg']/len(raw_df)*100,"%")

#Plot the % of positive-negative reviews
positive = [labels.label.value_counts()['pos']/len(raw_df)*100]
negative = [labels.label.value_counts()['neg']/len(raw_df)*100]
y = [0]
plt.barh(y, positive, color='pink', edgecolor='white')
plt.barh(y, negative, left=[100-i for i in negative], color='lightblue', edgecolor='white')
plt.show() 

#Plot the number of positive-negative reviews
sns.countplot(x= 'class',data = raw_df)

labels = raw_df["class"]


"""
    DATA PREPARATION
    1. tokenization : we split documents into tokens (e.g. words)
        - text boundaries
        - separate words in sentences
        - language dependent
    2. case normalization : convert each token completely upper-case or lower-case 
                        
    3. stemming : disregard singular/plural , coniugating verbs
                       
    4. stopword elimination
    5. creation of bigrams
"""

   
class LemmaBigramsTokenizer(object):
     def __init__(self):
        self.ita_stemmer = snowball.ItalianStemmer()
      
        
     def __call__(self, document):
        lemmas = []
        cond = 0
        
        cleaned_text = re.sub('[^a-zA-Z]',' ', document) 
        
        #create unigrams
       
        for t in word_tokenize(cleaned_text):
            
            t = t.strip() #strips whitespace from the beginning and end of the token
                          
            lemma = self.ita_stemmer.stem(t) #stemmer in ITALIAN
            
            if (lemma == 'non'):
                lemma_non = lemma
                cond = 1
            else :
                #if before 'lemma' there was the word 'non'
                if (cond == 1):
                    lemma = lemma_non + ' ' + lemma
                    cond = 0
                         
                # remove tokens 
                if ( len(lemma) > 1 and len(lemma) < 16) : 
                    lemmas.append(lemma)
     
            
        #create bigrams
        bigrams = list(nltk.bigrams(lemmas))
        
        for j in map(' '.join, bigrams) :
            lemmas.append(j)          
            
        return lemmas
    
   
# Italian stop-words
stopwords = sw.words('italian') + ['abbi', 'abbiam', 'avemm', 'avend', 'avess',
                    'avesser', 'avessim', 'avest', 'avet', 'avev', 'avevam',
                    'avra', 'avrann', 'avre', 'avrebb', 'avrebber', 'avrem',
                    'avremm', 'avrest', 'avret', 'avut', 'contr', 'ebber', 
                    'eran', 'erav', 'eravam', 'essend', 'facc', 'facess', 
                    'facessim', 'facest', 'fann', 'fara', 'farann', 'fare',
                    'farebb', 'farebber', 'farem', 'farest', 'fecer', 'foss', 
                    'fosser', 'fossim', 'fost', 'fumm', 'hann', 'nostr', 'perc',
                    'qual', 'quant', 'quell', 'quest', 'sara', 'sarann', 'sare',
                    'sarebb', 'sarebber', 'sarem', 'sarest', 'siam', 'sian',
                    'siat', 'siet', 'stand', 'stann', 'star', 'stara', 'starann', 
                    'stare', 'starebb', 'starebber', 'starem', 'starest', 'stav', 
                    'stavam', 'stemm', 'stess', 'stesser', 'stessim', 'stest', 
                    'stett', 'stetter', 'stiam', 'tutt', 'vostr', 'com', 'fac', 
                    'far', 'fec', 'fur', 'lor', 'sar', 'son', 'sti', 'avr'] 

stopwords_cleaned = [word for word in stopwords if (word != 'non' and word != 'pi' and word != 'contr'
                                                    and word != 'anche' and word != 'ma')]

tokenizer = LemmaBigramsTokenizer()
vectorize =  TfidfVectorizer(tokenizer=tokenizer,lowercase = True,  
                             min_df = 5 , stop_words=stopwords_cleaned )
#vectorize.vocabulary_
X_tfidf = vectorize.fit_transform(docu)

"""
   PLOT the points in 2d (scatterplot)
"""

#plot the distribution of pos/neg reviews
svd = TruncatedSVD(2) #n_dimensions=2 to plot the points in 2 dimensions
X_svd = svd.fit_transform(X_tfidf)

color= ['plum' if l == 'pos' else 'pink' for l in labels]
plt.scatter(X_svd[:,0],X_svd[:,1], color=color)
plt.show()


"""
    ALGORITHM CHOICE
    uncomment this part to see how I chose the algorithm.
    The choice was based on the highest f1-score
    
"""


"""
kb = SelectKBest(chi2,30000)
X_kb = kb.fit_transform(X_tfidf, labels)

X_train, X_test, y_train, y_test = train_test_split(X_kb, labels, test_size=0.2, random_state = 21)

#knearestneighbours
#clf = KNeighborsClassifier(n_neighbors=5)
#decisionTree
#clf = DecisionTreeClassifier(random_state=0)
#linearSVC
clf = svm.LinearSVC(dual=False, C= 1)
#SVC(kernel='linear')
#clf = svm.SVC(kernel='linear', C=1)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
print(f1)
"""


"""
    HYPERPARAMETERS TUNING
    use of GridSearchCV 
    
"""

X_train, X_test, y_train, y_test = train_test_split(docu, labels, test_size=0.2, random_state = 21)

#make pipeline 
pipeline = Pipeline([('vectorizer',TfidfVectorizer(tokenizer=tokenizer,lowercase = True, 
                                        min_df = 5 , stop_words=stopwords_cleaned )),
                     ('selectkbest', SelectKBest(chi2)),
                     ('classifier', svm.LinearSVC(dual=False))
                     ]) 

# Set the parameters 
grid_svc_kbest = {'classifier__C': [ 0.75, 1, 10],
                  'selectkbest__k': [15000, 30000, 50000 ]}
                  
gsCV = GridSearchCV(pipeline, param_grid = grid_svc_kbest, scoring='f1_weighted', 
                    verbose = 3, cv=5, n_jobs=-1)
gsCV.fit(X_train, y_train)

print()
print(f"Best parameters set found on development set: {gsCV.best_params_}")

print("Detailed classification report:")
y_pred = gsCV.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))
print()

#F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
print(f1)

#confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

#now we apply the models to the entire dataset
kb = SelectKBest(chi2, gsCV.best_params_['selectkbest__k'])
svc = svm.LinearSVC(dual=False, C= gsCV.best_params_['classifier__C'])

X_kb = kb.fit_transform(X_tfidf, labels)
X_svc = svc.fit(X_kb, labels)

"""
    EVALUATION:
    Let's try to evaluate the results in "evaluation.csv"
"""

#load evaluation.csv
raw_eval_df = pd.read_csv('dataset_winter_2020/evaluation.csv')

eval_reviews = raw_eval_df["text"]

#preprocessing + apply my model
X_eval_tok = vectorize.transform(eval_reviews)
X_eval_kb = kb.transform(X_eval_tok)
y_eval_pred = svc.predict(X_eval_kb)


"""
    RESULTS to file
"""

#load sample_submission.csv 
raw_results_df = pd.read_csv('dataset_winter_2020/sample_submission.csv')
columns= ['Predicted']
raw_results_df= raw_results_df.drop(columns, axis=1) 

#load results of classification 
y_series_res = pd.Series(y_eval_pred)
y_df_res = pd.DataFrame({'Predicted' : y_series_res})

#concatenate the two dataframes
df_concat = pd.concat([raw_results_df, y_df_res], axis=1)

#fast check
df_concat.describe()
df_concat.head(4)

export_csv = df_concat.to_csv(r'dataset_winter_2020/my_submission.csv', index = None,
                              header=True)



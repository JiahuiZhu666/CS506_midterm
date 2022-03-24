import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load files into DataFrames
X_train = pd.read_csv("./data/X_train.csv")
X_submission = pd.read_csv("./data/X_test.csv")

# Split training set into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(
    X_train.drop(['Score'], axis=1),
    X_train['Score'],
    test_size=1 / 4.0,
    random_state=0
)

# This is where you can do more feature selection
X_train_processed = X_train.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary'])
X_test_processed = X_test.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary'])
X_submission_processed = X_submission.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'Score'])
expr_pattern = re.compile(pattern=r"[^a-zA-Z0-9'\s]", flags=0)
X_train['Text'] = [expr_pattern.sub('', word) for word in X_train['Text'].tolist()]
X_train['Text'] = X_train['Text'].str.lower().str.split()
stopwords_list = ['i', "i've", "ive", 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "youre",
                  "you've", "youve", "you'll", "you'd", "youll", "youd", 'your', 'yours', 'yourself',
                  'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", "shes", 'her', 'hers', 'herself', 'it',
                  "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                  'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", "thatll", 'these', 'those',
                  'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
                  'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                  'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
                  'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
                  'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                  'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                  'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                  'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
                  'can', 'will', 'just', 'don', "don't", "dont", 'should', "should've", "shouldve",
                  'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", "arent", 'couldn', "couldn't",
                  "couldnt", 'didn', "didn't", "didnt", 'doesn', "doesn't", "doesnt", 'hadn', "hadn't", "hadnt", 'hasn',
                  "hasn't", "hasnt", 'haven', "haven't", "havent", 'isn', "isn't", "isnt", 'ma', 'mightn', "mightn't",
                  "mightnt", 'mustn', "mustn't", "mustnt", 'needn', "needn't", "neednt", 'shan', "shan't", "shant",
                  'shouldn', "shouldn't", "shouldnt",
                  'wasn', "wasn't", "wasnt", 'weren', "weren't", "werent", 'won', "won't", "wont", 'wouldn', "wouldn't",
                  "wouldnt"]
X_train['Text'] = X_train['Text'].apply(lambda review: [word for word in review if word not in stopwords_list])
X_train['Text'] = X_train['Text'].apply(', '.join)
expr_pattern1 = re.compile(pattern = r"[^a-zA-Z0-9'\s]", flags = 0)
X_train['Text'] = [expr_pattern1.sub('', x) for x in X_train['Text'].tolist()]
X_train['Text'] = X_train['Text'].str.lower()
# Learn the model
pca = PCA(n_components=4)
# knn = KNeighborsClassifier(n_neighbors=1)
# clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
# clf = LogisticRegression(penalty='l1',C=1, class_weight='balanced', solver='saga',
#                          multi_class='auto', n_jobs=-1, random_state=40)
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
clf = RandomForestClassifier(n_estimators=300, oob_score=True,
                               n_jobs=-1, random_state=50,
                               max_features="auto", min_samples_leaf=200)
model = make_pipeline(pca,clf)
model = model.fit(X_train_processed, Y_train)

# Predict the score using the model
Y_test_predictions = model.predict(X_test_processed)
X_submission['Score'] = model.predict(X_submission_processed)

# Evaluate your model on the testing set
print("Accuracy on testing set = ", accuracy_score(Y_test, Y_test_predictions))

# Create the submission file
submission = X_submission[['Id', 'Score']]
submission.to_csv("./data/submission_random_modify_1.csv", index=False)
submission['Score'].value_counts().plot(kind='bar', legend=True, alpha=.5)
plt.show()

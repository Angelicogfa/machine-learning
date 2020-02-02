from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from html import unescape
import re
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import email.policy
import email
import os
import tarfile
import urllib

# Pasta onde os arquivos serao carregados
arquivos = os.path.join(os.path.abspath('..\..\\'), "dados")

DOWNLOAD_ROOT = 'http://spamassassin.apache.org/old/publiccorpus/'
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join(arquivos, "spam")

# funcao para baixar os dados


def fetch_spam_data(spam_url=SPAM_URL, spam_path=SPAM_PATH):
    if not os.path.isdir(spam_path):
        os.makedirs(spam_path)
    for filename, url in (("ham.tar.bz2", HAM_URL), ('spam.tar.bz2', SPAM_URL)):
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=SPAM_PATH)
        tar_bz2_file.close()


# fazendo download dos dados
fetch_spam_data()

# carregar os arquivos em memoria
HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")
SPAM_DIR = os.path.join(SPAM_PATH, "spam")

ham_filenames = [name for name in sorted(
    os.listdir(HAM_DIR)) if len(name) > 20]
spam_filenames = [name for name in sorted(
    os.listdir(SPAM_DIR)) if len(name) > 20]

len(ham_filenames)
len(spam_filenames)

# vamos usar o modulo email do python para fazer o cast dos arquivos para um objeto tipado
# com todos os dados


def load_email(is_spam, filename, spam_path=SPAM_PATH):
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(spam_path, directory, filename), 'rb') as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)


ham_emails = [load_email(is_spam=False, filename=name)
              for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name)
               for name in spam_filenames]

# Vamos ver a estrutura de um email ham
print(ham_emails[0].get_content().strip())

print(spam_emails[0].get_content().strip())

# Alguns emails possuem estrutura diferenciadas, arquivos em anexo, imagens.


def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()


def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures


structures_counter(ham_emails).most_common()
structures_counter(spam_emails).most_common()

# Vamos analisar os headers de um email
for header, value in spam_emails[0].items():
    print(header, ":", value)

spam_emails[0]["Subject"]

# Vamos criar a base e quebrar os dados entre treino e teste

x = np.array(ham_emails + spam_emails)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y)

# Vamos remover o conteudo HTML e deixar apenas os dados dos textos


def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)


html_spam_emails = [email for email in x_train[y_train == 1]
                    if get_email_structure(email) == 'text/html']
sample_html_spam = html_spam_emails[7]
print(sample_html_spam.get_content().strip()[:1000], "...")
print(html_to_plain_text(sample_html_spam.get_content())[:1000], "...")

# Vamos escrever uma funçao que receba um email e devolva o conteudo como texto plano


def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ('text/plain', 'text/html'):
            continue
        try:
            content = part.get_content()
        except:
            content = str(part.get_payload())

        if ctype == 'text/plain':
            return content
        else:
            html = content

        if html:
            return html_to_plain_text(html)


print(email_to_text(sample_html_spam)[:100], '...')

# pip install nltk

try:
    import nltk

    stemmer = nltk.PorterStemmer()
    for word in ('Computations', 'Computation', 'Computing', 'Computed', 'Compute', 'Compulsive'):
        print(word, ' => ', stemmer.stem(word))
except ImportError:
    print('Error: stemming requires the NLTK module..')
    stemmer = None


# Vamos substituir agora as urls pela palavra URL
# pip install urlextract

try:
    import urlextract
    url_extractor = urlextract.URLExtract()
    print(url_extractor.find_urls(
        'Will it detect github.com and https://youtu.be/7Pq-S557XQU?t=3m32s'))
except ImportError:
    print('Error: stemming requires the urlextract module..')
    url_extractor = None

# Vamos juntar tudo agora em um transformer para converter os emails em um dicionario
# com o contador de palavras


class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True,
                 replace_urls=True, replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email_to_text(email) or ''
            if self.lower_case:
                text = text.lower()
            if self.replace_numbers and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, ' URL ')
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', ' NUMBER ', text)
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)

            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, counter in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += counter
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)


x_few = x_train[:3]
x_few_wordcounts = EmailToWordCounterTransformer().fit_transform(x_few)
x_few_wordcounts

# Vamos converter nossa bolsa de palavras em um vetor, para isso vamos usar
# outro transformer. O metodo fit ira construir o vocabulario e o metodo transform
# vai usar o vocabulario para converter a bolsa de palavras em um vetor resultando
# em uma matrix esparsa


class WordCounterToVectorTransform(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)

        most_common = total_count.most_common()[:self.vocabulary_size]
        self.most_common = most_common
        self.vocabulary_ = {word: index + 1 for index,
                            (word, count) in enumerate(most_common)}
        return self

    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []

        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))


vocab_transform = WordCounterToVectorTransform(vocabulary_size=10)
x_few_vector = vocab_transform.fit_transform(x_few_wordcounts)
x_few_vector.toarray()
vocab_transform.vocabulary_

# Agora vamos escrever nosso primeiro classificador de spam

preprocess_pipeline = Pipeline([
    ('email_to_wordcount', EmailToWordCounterTransformer()),
    ('wordcount_to_vector', WordCounterToVectorTransform())
])

x_train_transformed = preprocess_pipeline.fit_transform(x_train)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

log_clf = LogisticRegression(solver='lbfgs', random_state=42)
score = cross_val_score(log_clf, x_train_transformed, y_train, cv=3, verbose=3)
score.mean()

from sklearn.metrics import precision_score, recall_score

x_test_transformed = preprocess_pipeline.transform(x_test)
log_clf = LogisticRegression(solver='lbfgs', random_state=42)
log_clf.fit(x_train_transformed, y_train)

y_pred = log_clf.predict(x_test_transformed)

print('Precision {:.2f}'.format(precision_score(y_test, y_pred) * 100))
print('Recall {:.2f}'.format(recall_score(y_test, y_pred) * 100))
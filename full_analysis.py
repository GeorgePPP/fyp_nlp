import pandas as pd
import nltk
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import re

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def load_data(file_path):
    """Load data from CSV file."""
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
    return df

def preprocess_text(text, stemmer):
    """Preprocess text: lowercase, remove non-alphabetic characters, stem, remove stopwords."""
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(WordNetLemmatizer().lemmatize(word, pos='v')) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def pos_tagging_and_chunking(text):
    """Perform POS tagging and chunking."""
    doc = nlp(text)
    chunks = [(chunk.text, chunk.root.dep_, chunk.root.head.text) for chunk in doc.noun_chunks]
    return chunks

def dependency_parsing(text):
    """Perform dependency parsing."""
    doc = nlp(text)
    return [(token.text, token.dep_, token.head.text) for token in doc]

def perform_lda(X, n_topics, max_iter):
    """Perform LDA topic modeling."""
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, 
                                    max_iter=max_iter, learning_method='online', 
                                    learning_offset=50., 
                                    topic_word_prior=0.1, 
                                    doc_topic_prior=0.1)
    lda_topics = lda.fit_transform(X)
    return lda, lda_topics

def visualize_topics(lda, lda_topics, vectorizer):
    """Visualize LDA topics."""
    n_topics = lda.components_.shape[0]
    words = vectorizer.get_feature_names_out()
    
    for topic_idx, topic in enumerate(lda.components_):
        print(f"Topic #{topic_idx}:")
        print(" ".join([words[i] for i in topic.argsort()[:-10 - 1:-1]]))
    
    topic_weights = pd.DataFrame(lda.components_.T, index=words, columns=[f"Topic #{i}" for i in range(n_topics)])
    topic_weights["max_weight"] = topic_weights.max(axis=1)
    topic_weights_sorted = topic_weights.sort_values(by="max_weight", ascending=False).head(10)

    print("\nTop 10 words contributing most to any topic:")
    print(topic_weights_sorted.drop("max_weight", axis=1))
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=range(lda_topics.shape[1]), y=lda_topics[0], s=lda_topics[0]*1000, legend=False)
    plt.title("Bubble Plot for LDA Topic Modelling")
    plt.show()

def experiment_lda_parameters(X, max_iter_values):
    """Experiment with different LDA parameters."""
    perplexity_values = []
    for max_iter in max_iter_values:
        print(f"Fitting LDA with max_iter = {max_iter}")
        lda = LatentDirichletAllocation(n_components=10, random_state=42, 
                                        max_iter=max_iter, learning_method='online', 
                                        learning_offset=50., 
                                        topic_word_prior=0.1, 
                                        doc_topic_prior=0.1)
        lda.fit(X)
        perplexity = lda.perplexity(X)
        perplexity_values.append(perplexity)
        print(f"Perplexity for max_iter = {max_iter}: {perplexity}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(max_iter_values, perplexity_values, marker='o')
    plt.title('Perplexity vs. max_iter')
    plt.xlabel('max_iter')
    plt.ylabel('Perplexity')
    plt.grid(True)
    plt.show()

def main(file_path, n_topics=5, max_iter=10):
    # Load data
    df = load_data(file_path)
    print(df.head())

    # Preprocess text
    stemmer = SnowballStemmer("english")
    df['tokens'] = df['title'].apply(lambda x: preprocess_text(x, stemmer))
    print("\nPreprocessed text:")
    print(df['tokens'].head())

    # POS tagging and chunking
    df['pos_chunks'] = df['tokens'].apply(pos_tagging_and_chunking)
    print("\nPOS chunks:")
    print(df['pos_chunks'].head())

    # Dependency parsing
    df['dependency_parsing'] = df['tokens'].apply(dependency_parsing)
    print("\nDependency parsing:")
    print(df['dependency_parsing'].head())

    # Vectorize text
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2)
    X = vectorizer.fit_transform(df['tokens'])

    # Perform LDA
    lda, lda_topics = perform_lda(X, n_topics, max_iter)
    print(f"\nPerplexity: {lda.perplexity(X)}")

    # Visualize topics
    visualize_topics(lda, lda_topics, vectorizer)

    # Experiment with different max_iter values
    max_iter_values = [10, 30, 50, 100]
    experiment_lda_parameters(X, max_iter_values)

if __name__ == "__main__":
    file_path = 'Compiled dataset.csv'  # Change this to your file path
    main(file_path)
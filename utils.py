import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from langdetect import detect, LangDetectException
from wordcloud import WordCloud
import emoji
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from spellchecker import SpellChecker
import contractions
import string
from tqdm import tqdm
from textblob import TextBlob
from sklearn.model_selection import cross_val_predict, cross_validate, StratifiedKFold, KFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, make_scorer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer,WordNetLemmatizer
import nltk
import numpy as np

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('snowball_data')

def get_word2vec_vectors(data, model, vector_size):
    vectors = np.zeros((len(data), vector_size))
    for i, tokens in enumerate(data):
        valid_tokens = [model.wv[token] for token in tokens if token in model.wv]
        if valid_tokens:
            vectors[i] = np.mean(valid_tokens, axis=0)
    return vectors

def get_embedding_vectors(sequences, embedding_matrix):
    embedding_vectors = np.zeros((len(sequences), embedding_matrix.shape[1]))
    for i, seq in enumerate(sequences):
        valid_words = [embedding_matrix[word] for word in seq if word != 0]
        if valid_words:
            embedding_vectors[i] = np.mean(valid_words, axis=0)
    return embedding_vectors

def plot_model_metrics(y_val, y_preds, model_names):
    """
    Plots accuracy, F1 score, precision, and recall for given models in a single figure with subplots.

    Parameters:
    y_val (array-like): True labels.
    y_preds (list of array-like): List of predicted labels from different models.
    model_names (list of str): List of model names.
    """
    metrics = {
        'Accuracy': [accuracy_score(y_val, y_pred) for y_pred in y_preds],
        'F1 Score': [f1_score(y_val, y_pred) for y_pred in y_preds],
        'Precision': [precision_score(y_val, y_pred) for y_pred in y_preds],
        'Recall': [recall_score(y_val, y_pred) for y_pred in y_preds],
    }

    sns.set(style="white")  

    colors = ['skyblue', 'lightgreen', 'salmon', '#dda0dd']  

    fig, axs = plt.subplots(2, 2, figsize=(14,8)) 

    bar_width = 0.4
    index = np.arange(len(model_names))

    for ax, (metric_name, metric_values), color in zip(axs.flatten(), metrics.items(), colors):
        bars = ax.bar(index, metric_values, bar_width, color=color)

        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} of Models', fontsize=12)
        ax.set_xticks(index)
        ax.set_xticklabels(model_names, rotation=45, ha='right')


        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom', fontsize=8)

    plt.subplots_adjust(wspace=0.4, hspace=0.6) 
    plt.suptitle('Model Performance Metrics', fontsize=14) 
    plt.show()

def print_evaluation_metrics(y_test, y_pred, model_name, description_type):
    print(f"Evaluation Metrics for {model_name} with {description_type} Descriptions:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='macro'):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    print("\n")

def word_freq(words, title, ax, color, axes):
    freq = pd.Series(words).value_counts()
    axes[ax].bar(freq.index[0:10], freq[:10], align='center', alpha=0.7, color=color)
    axes[ax].set_xticks(freq.index[0:10])
    axes[ax].set_xticklabels(labels=freq.index[0:10], rotation=45)
    axes[ax].set_title(title, fontsize=14)
    axes[ax].set_ylabel('Frequency', fontsize=12)
    axes[ax].set_xlabel('Word', fontsize=12)
    axes[ax].tick_params(axis='both', which='major', labelsize=10)


def wordCloud(text_sources):
    plt.figure(figsize=(15, 5))
    for i, (title, text) in enumerate(text_sources.items(), start=1):
        plt.subplot(1, 3, i)
        wordcloud = WordCloud(width=400, height=200, background_color="white").generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')  # Removed unnecessary equal sign
        plt.axis("off")
        plt.title(title)
    plt.tight_layout()
    plt.show()


def detect_lang(x):
  try:
    return detect(x)
  except:
    return 'error'
  
  
def plot_top_languages_frequency(data_sources, titles, figsize=(14, 4)):
    """

    Args:
        data_sources (list): List of pandas Series containing language data from different sources.
        titles (list): List of titles for each subplot.
        figsize (tuple, optional): Size of the figure (width, height) in inches. Defaults to (25, 10).
    """

    if not len(data_sources) == len(titles):
        raise ValueError("Lengths of data_sources and titles must be equal.")

    num_plots = len(data_sources)
    fig, axes = plt.subplots(1, num_plots, figsize=figsize, facecolor='none')

    for i, (source_data, title) in enumerate(zip(data_sources, titles)):
        
        language_counts = source_data.value_counts().sort_values(ascending=False)

        
        top_languages = language_counts.index[0:20]
        top_frequencies = language_counts.iloc[0:20]

        
        colors = sns.color_palette('Set2', len(top_languages))
        ax = axes[i]
        ax.bar(top_languages, top_frequencies, color=colors)
        ax.set_xticks(top_languages)
        ax.set_xticklabels(labels=top_languages, rotation=45, ha='right')
        ax.set_ylabel('Number of Occurrences', fontsize=12)
        ax.set_title(title, fontsize=14)

        
        for bar, value in zip(ax.patches, top_frequencies):
            height = bar.get_height()
            x_loc = bar.get_x() + bar.get_width() / 2

    
    plt.tight_layout()
    plt.show()

def remove_stop_words(text, language, stopwords_dict):
    all_stopwords = set()
    for lang in stopwords_dict:
        all_stopwords.update(stopwords_dict[lang])
    filtered_text = []
    for word in text.split():
        if word.lower() not in all_stopwords:
            filtered_text.append(word)
    return ' '.join(filtered_text)



def remove_non_alphanumeric(text):

    
    patterns = [
          html_tag_pattern := r'<[^>]*>',  # Define and assign for efficiency
          notalphanum := r'[^A-Za-z0-9\s.,?!;:()\'"-_]',
          email := r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
          url_pattern := r'http[s]?://\S+|www\S+',  # Include http and https
          license_number := r'license number\s*[a-zA-Z0-9_]*',  # Capture underscores too
          extras := r'(?:[$€£¥]\d+|\d+[$€£¥])|[$€£¥] \d+|\b\d+\s*[€€]\b|\b[€€]\s*\d+\b'
          
      ]


    cleaned_text = text
    for pattern in patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text)

    # Replace consecutive spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\d+', '', cleaned_text)
  


    cleaned_text = cleaned_text.translate(str.maketrans('', '', string.punctuation))

    return cleaned_text


def spell_checking(text):
    try:
        if not text.strip():
            return text
        
        detected_language = detect(text)
        #print("Detected language:", detected_language)
        
        spell_checker = SpellChecker(language=detected_language)
        
        words = text.split()
        
        corrected_words = []
        
        for word in words:
            # if the word is misspelled
            if spell_checker.unknown([word]):
                # correct the misspelled word
                corrected_word = spell_checker.correction(word)
                if corrected_word is not None:
                    corrected_words.append(corrected_word)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        
        corrected_text = ' '.join(corrected_words)
        #print("Text corrected successfully.")
        
        return corrected_text
    except Exception as e:
        #print("An error occurred:", str(e))
        return text


def replace_emojis_with_text(text):
    replaced_text = emoji.demojize(text)
    replaced_text = replaced_text.replace("_", " ")
    return replaced_text.replace("::", ": :")


def stemmer_checking(text,detected_language):

    
    language_mapping = {
        'ar': 'arabic', 'da': 'danish', 'nl': 'dutch', 'en': 'english',
        'fi': 'finnish', 'fr': 'french', 'de': 'german', 'hu': 'hungarian', 'it': 'italian',
        'no': 'norwegian', 'pt': 'portuguese', 'ru': 'russian', 'es': 'spanish', 'sv': 'swedish'
        }
    snowball_language = language_mapping.get(detected_language, 'english')
    #print("Snowball language:", snowball_language)
    stemmer = SnowballStemmer(snowball_language)
    stemmed_text = [stemmer.stem(word) for word in text.split()]
    stemmed_text = ' '.join(stemmed_text)
    return stemmed_text

def expand_contractions(text, contractions_dict):
    words = text.split()
    expanded_words = []
    for word in words:
        # If the word is a contraction, expand it
        if word.lower() in contractions_dict:
            expanded_words.append(contractions_dict[word.lower()])
        else:
            expanded_words.append(word)
    expanded_text = ' '.join(expanded_words)
    return expanded_text


def lemmatize_text(text):

    lemma = WordNetLemmatizer()
    return " ".join(lemma.lemmatize(word) for word in text.split())


def clean_data(text_series, stopwords_dict, contractions_dict, stemming=False, lemmitize = False):
    cleaned_text_stemmed = []
    cleaned_text_lemmitize = []
    stopwords_dict = stopwords_dict
    contractions_dict = contractions_dict
    
    for text in tqdm(text_series):
        text = text.lower()
        
        
        if not text.strip():
            cleaned_text_stemmed.append(text)
            cleaned_text_lemmitize.append(text)
            continue
   
        
        try:
            language = detect(text)
        except LangDetectException:
            language = 'unsupported'

        text = re.sub('x000d', ' ', text)

        text = expand_contractions(text, contractions_dict)

        text = remove_stop_words(text, language,stopwords_dict)

        text = emoji.demojize(text)
        text = text.replace("::", ": :")
        text = remove_non_alphanumeric(text)
        
        # Stem text
        if stemming and language != 'unsupported':
            text_stemmed = stemmer_checking(text, language)
            cleaned_text_stemmed.append(text_stemmed)
        else:
            cleaned_text_stemmed.append(text)
        # lemmitize text
        if lemmitize and language != 'unsupported':
            text_lemmitize = lemmatize_text(text)
            cleaned_text_lemmitize.append(text_lemmitize)
        else:
            cleaned_text_lemmitize.append(text)
        
        
    
    return cleaned_text_stemmed, cleaned_text_lemmitize

def merge_datasets(properties, reviews):
    
    if 'index' not in reviews.columns:
        reviews = reviews.reset_index()

    
    merged_reviews_stemming = reviews.groupby('index')['comments_stemming'].agg(lambda x: ' '.join(str(comment) for comment in x)).reset_index()
    merged_reviews_lemmitize = reviews.groupby('index')['comments_lemmitize'].agg(lambda x: ' '.join(str(comment) for comment in x)).reset_index()
    
    
    merged_reviews = pd.merge(merged_reviews_stemming, merged_reviews_lemmitize, on='index')
    
    
    merged_df = properties.merge(merged_reviews, how='left', left_index=True, right_on='index')
    
   
    merged_df.set_index('index', inplace=True)
    
    # Fill NaN values in the 'comments_stemming' and 'comments_lemmitize' columns with empty strings
    merged_df['comments_stemming'] = merged_df['comments_stemming'].fillna('')
    merged_df['comments_lemmitize'] = merged_df['comments_lemmitize'].fillna('')
    merged_df['description_stemming'] = merged_df['description_stemming'].fillna('')
    merged_df['description_lemmitize'] = merged_df['description_lemmitize'].fillna('')
    merged_df['host_about_stemming'] = merged_df['host_about_stemming'].fillna('')
    merged_df['host_about_lemmitize'] = merged_df['host_about_lemmitize'].fillna('')

    return merged_df




def analyze_sentiment(text):
    # Convert NaNs to empty strings
    text = str(text)
    
    # Perform sentiment analysis using TextBlob
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity

    # Interpret the sentiment score
    if sentiment_score > 0:
        sentiment_label = 'positive'
    elif sentiment_score < 0:
        sentiment_label = 'negative'
    else:
        sentiment_label = 'neutral'

    return sentiment_label


def sentiment_chart(sentiment_counts, ax):
    sentiment_counts[['positive', 'neutral', 'negative']].plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'], alpha=0.7, ax=ax)

    # Customize the plot
    ax.set_title('Sentiment Analysis of Reviews')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Number of Reviews')
    for i, count in enumerate(sentiment_counts[['positive', 'neutral', 'negative']]):
        ax.text(i, count + 10, str(count), ha='center', va='bottom')


#     # Define function to get word vector
# def get_word(word):
#     return glove.vectors[glove.stoi[word]] if word in glove.stoi else np.zeros(50)  # Handle unknown words

# # Define function to calculate mean embedding for a description
# def description_embedding(description):
#     if not isinstance(description, str):
#         return np.zeros(50)  # Handle non-string data
#     description_vector = [get_word(word) for word in description.split()]
#     if description_vector:
#         return np.mean(description_vector, axis=0)  # Average word vectors
#     else:
#         return np.zeros(50)  # No valid words found

# # Function to add embedding columns
# def embed_descriptions(df, text_column, embedding_column):
#     embeddings = []
#     for description in tqdm(df[text_column]):
#         embeddings.append(description_embedding(description).tolist())  # Convert numpy array to list
#     df[embedding_column] = embeddings


# # Function to evaluate the embeddings
# def evaluate_embeddings(df, output_column, target_column):
#     """
#     Evaluate the embeddings generated by embed_descriptions function.

#     Args:
#     df (pd.DataFrame): DataFrame containing embedded descriptions and target labels.
#     output_column (str): Name of the column containing the embedded descriptions.
#     target_column (str): Name of the column containing the target labels.

#     Returns:
#     dict: A dictionary containing accuracy, precision, recall, F1-score, and classification report.
#     """
#     embeddings = df[output_column].tolist()
#     targets = df[target_column].tolist()

#     # Convert embeddings to numpy array
#     embeddings = np.array(embeddings)

#     # Predict labels from embeddings
#     # Here you would replace the prediction logic with your actual model prediction
#     # For demonstration purposes, let's assume a simple threshold-based prediction
#     predictions = (embeddings.sum(axis=1) > 0).astype(int)

#     # Compute evaluation metrics
#     accuracy = accuracy_score(targets, predictions)
#     precision = precision_score(targets, predictions)
#     recall = recall_score(targets, predictions)
#     f1 = f1_score(targets, predictions)
#     report = classification_report(targets, predictions)

#     evaluation_metrics = {
#         'Accuracy': accuracy,
#         'Precision': precision,
#         'Recall': recall,
#         'F1-Score': f1,
#         'Classification Report': report
#     }

#     return evaluation_metrics


# def calculate_tfidf_scores(X_tfidf, X_ngram_tfidf, y, model, nr_splits=10, scoring=None, verbose=False):
#     """
#     Calculates performance scores of a model using TF-IDF representations with Stratified K-Fold Cross Validation.

#     Args:
#         X_tfidf (sparse matrix): TF-IDF features.
#         X_ngram_tfidf (sparse matrix): N-Gram TF-IDF features.
#         y (array-like): Target variable.
#         model (object): Classification model.
#         nr_splits (int, optional): Number of splits for Stratified K-Fold CV. Defaults to 10.
#         scoring (dict, optional): Scoring metrics to use. Defaults to None, which uses accuracy, precision, recall, and F1-score.
#         verbose (bool, optional): Controls verbosity of cross-validation output. Defaults to False.

#     Returns:
#         dict: Dictionary containing average performance scores for TF-IDF and N-Gram TF-IDF.
#     """
#     if scoring is None:
#         scoring = {
#             'accuracy': 'accuracy',
#             'precision': make_scorer(precision_score, pos_label=1),
#             'recall': make_scorer(recall_score, pos_label=1),
#             'f1': make_scorer(f1_score, pos_label=1)
#         }

#     # Create Stratified K-Fold object
#     kfold = StratifiedKFold(n_splits=nr_splits)

#     results = {}

#     # Perform Stratified K-Fold CV for each TF-IDF representation
#     for name, tfidf_features in zip(["TF-IDF", "N-Gram TF-IDF"], [X_tfidf, X_ngram_tfidf]):
#         cv_scores = cross_validate(model, tfidf_features, y, cv=kfold, scoring=scoring, return_train_score=True, verbose=verbose)

#         # Calculate and store average scores across folds
#         average_scores = {metric: np.mean(cv_scores['test_' + metric]) for metric in scoring}
#         y_pred = cross_val_predict(model, X_tfidf, y, cv=kfold)
#         report = classification_report(y, y_pred)
#         results[name] = average_scores

#     return results


# def evaluate_model(tfidf_matrix, y, model, n_splits=5):
#     """
#     Apply the given model to the TF-IDF scores and evaluate using various metrics with k-fold cross-validation.
    
#     Parameters:
#     tfidf_matrix : scipy.sparse.csr_matrix
#         Sparse matrix containing TF-IDF scores.
#     y : array-like
#         Target variable.
#     model : object
#         Model object compatible with scikit-learn API.
#     n_splits : int, default=5
#         Number of folds for cross-validation.
        
#     Returns:
#     scores : dict
#         Dictionary containing mean evaluation metric scores across all folds.
#     classification_report : str
#         String containing the classification report.
#     """
#     kfold = KFold(n_splits=n_splits, shuffle=True)
    
#     scoring = ['accuracy', 'precision', 'recall', 'f1']
#     cv_results = cross_validate(model, tfidf_matrix, y, cv=kfold, scoring=scoring, return_train_score=False)
    
#     scores = {metric: cv_results[f"test_{metric}"].mean() for metric in scoring}
    
#     # Generate classification report using cross-validated predictions
#     y_pred = cross_val_predict(model, tfidf_matrix, y, cv=kfold)
#     report = classification_report(y, y_pred)
    
#     return report


# def compare_tfidf_performance(clf, X_tfidf, y, k_fold=5):
#     """
#     Compare the performance of a TF-IDF representation using stratified k-fold cross-validation with multiple scoring metrics.

#     Parameters:
#     - clf: The classifier model to use (e.g., Logistic Regression, MLP, etc.).
#     - X_tfidf (array-like): TF-IDF transformed input data.
#     - y (array-like): Target labels.
#     - k_fold (int): Number of folds for cross-validation. Default is 5.

#     Returns:
#     - results (dict): A dictionary containing mean scores for each metric and the classification report.
#     """
#     skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    
#     scoring = ['accuracy', 'precision', 'recall', 'f1']
#     cv_results = cross_validate(clf, X_tfidf, y, cv=skf, scoring=scoring, return_train_score=False)
    
#     scores = {metric: cv_results[f"test_{metric}"].mean() for metric in scoring}
    
#     # Generate classification report using cross-validated predictions
#     y_pred = cross_val_predict(clf, X_tfidf, y, cv=skf)
#     report = classification_report(y, y_pred)
    
#     results = {
#         'mean_scores': scores,
#         'classification_report': report
#     }
    
#     return results
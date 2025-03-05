import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import AgglomerativeClustering
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

class TextProcessing:
    def __init__(self, use_tokenization=True):
        custom_stopwords = set(stopwords.words('english'))
        # Add any domain-specific stopwords here
        custom_stopwords.update(['cid', 'th','rd','ye', 'tha', 'nd','ng','ee','aa','ne','oe','te','ei','er'])
        self.stop_words = custom_stopwords
        self.use_tokenization = use_tokenization

    def clean_text(self, text):
        """
        Cleans the text by removing HTML tags and extra spaces.
        """
        text = re.sub(r'<[^>]+>', '', str(text))  # Remove HTML tags
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        return text.strip()

    def tokenize_text(self, text):
        """
        Tokenizes the text.
        """
        try:
            return word_tokenize(text.lower())
        except LookupError:
            print("NLTK tokenizer issue detected, using simple tokenization as a fallback.")
            return text.lower().split()

    def filter_tokens(self, tokens):
        """
        Filters tokens by removing stopwords and non-alphabetic tokens.
        """
        filtered_tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]
        return filtered_tokens

    def preprocess_texts(self, texts):
        """
        Preprocesses texts by cleaning, tokenizing, and filtering them.
        """
        processed_texts = []
        for text in texts:
            text = self.clean_text(text)
            if self.use_tokenization:
                tokens = self.tokenize_text(text)
                filtered_tokens = self.filter_tokens(tokens)
                processed_text = ' '.join(filtered_tokens)
            else:
                processed_text = text
            processed_texts.append(processed_text)
        return processed_texts

class BERTopicProcessor:
    def __init__(self, num_topics=None, bert_clean_internal=True):
        """
        Initializes the BERTopicProcessor with optional number of topics and internal vectorization.
        """
        if num_topics == 'auto':
            self.num_topics = None 
            print("Auto mode: The number of topics will be determined automatically.")
        else:
            self.num_topics = int(num_topics)  # Ensure it's an integer
            print(f"Manual mode: Using {self.num_topics} topics.")
        self.topic_model = None
        self.bert_clean_internal = bert_clean_internal

    def initialize_model(self, dataset_size):
        """
        Initializes the BERTopic model based on the dataset size.
        """
        if dataset_size <= 1500:
            embedding_model = SentenceTransformer('msmarco-distilbert-base-v3')
            umap_model = umap.UMAP(
                n_neighbors=25,
                n_components=10,
                min_dist=0.2,
                metric='cosine',
                random_state=42
            )
            clustering_model = AgglomerativeClustering(
                distance_threshold=0.5,
                n_clusters=None,
                linkage='ward'
            )
            print("Using small dataset configuration with Agglomerative Clustering.")
        else:
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            umap_model = umap.UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )
            clustering_model = HDBSCAN(
                min_cluster_size=2,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )
            print("Using large dataset configuration with HDBSCAN clustering.")

        # Internal vectorization based on the toggle
        if self.bert_clean_internal:
            vectorizer_model = CountVectorizer(
                stop_words='english',
                max_df=0.95,
                min_df=2,
                ngram_range=(1, 2),
                max_features=5000  # Adjust as needed
            )
            print("Internal vectorization with stopword removal is enabled.")
        else:
            vectorizer_model = None
            print("Internal vectorization is disabled.")

        # Initialize the BERTopic model
        self.topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=clustering_model,
            vectorizer_model=vectorizer_model,  # Include vectorizer_model
            nr_topics=self.num_topics,
            verbose=True
        )

    def perform_bertopic(self, texts):
        """
        Performs BERTopic analysis on the given texts and returns the topics.
        """
        print("Starting BERTopic analysis...")
        topics, _ = self.topic_model.fit_transform(texts)
        print(f"Generated {len(np.unique(topics))} unique topics.")
        return topics

    def get_top_words_for_topic(self, topic_id, num_words=10):
        """
        Returns the top words associated with the given topic_id.
        """
        if self.topic_model is None:
            raise ValueError("Topic model is not initialized. Please initialize and train the model first.")
        
        # Extract top words for the given topic
        topic_info = self.topic_model.get_topic(topic_id)
        
        if not topic_info:
            raise ValueError(f"No words found for topic {topic_id}. Ensure the topic ID is valid.")
        
        return topic_info[:num_words]

class BERTopicCSVDataSaver:
    def __init__(self, dataset_handler, bertopic_processor):
        """
        Initializes the BERTopicCSVDataSaver with a reference to DatasetHandler and BERTopicProcessor.
        """
        self.dataset_handler = dataset_handler
        self.bertopic_processor = bertopic_processor

    def save_results(self, predictions):
        """
        Saves BERTopic results to the CSV file in the next available columns: one for the topic and one for the top words.
        """
        df = self.dataset_handler.read_csv()

        # Initialize columns for topic and top words
        topic_column_name = 'BERTopic_Topic'
        words_column_name = 'BERTopic_Top_Words'

        # Ensure the BERTopic model is available
        if self.bertopic_processor.topic_model is None:
            raise ValueError("Topic model is not initialized. Please initialize and train the model first.")
        
        # Function to get the top words for a topic
        def get_top_words_for_topic(topic_id, num_words=10):
            topic_info = self.bertopic_processor.topic_model.get_topic(topic_id)
            if topic_info:
                return ', '.join([word for word, _ in topic_info[:num_words]])
            return "No words available"

        # Fill in the columns with predictions and corresponding top words
        df[topic_column_name] = predictions
        df[words_column_name] = [get_top_words_for_topic(topic) for topic in predictions]

        # Write back to CSV
        self.dataset_handler.write_csv(df)

    def get_topics_over_time(self, texts, timestamps):
        """
        Generates topics over time data required for DTM visualization.
        """
        topics_over_time = self.topic_model.topics_over_time(texts, timestamps)
        return topics_over_time

import pandas as pd
import re
import unidecode
import nltk
from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer



class DataPreprocessor:
    """
    The DataPreprocessor class provides a suite of methods to preprocess text data,
    including cleaning the text, removing stopwords, splitting hashtags, and usernames.
    """

    # Regular expressions to match special characters, usernames, and website links
    SPECIAL_CHARS_REGEX = r"[\*\+'\/\(\)\]\[\_\|]"
    USERNAME_REGEX = r'@\w*'
    WEBSITE_REGEX = r'http\S*'

    # Create an instance of TweetTokenizer
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

    @staticmethod
    def remove_usernames_weblinks_and_special_chars(text):
        """
        Replaces usernames and website links in the given text with placeholders,
        replaces "&amp" with empty string, special characters with spaces, and removes apostrophes.

        Args:
            text (str): The original text string.

        Returns:
            str: The cleaned text string.
        """
        # Replace usernames and website links with placeholders
        text = re.sub(DataPreprocessor.USERNAME_REGEX, 'USERNAME', text)
        text = re.sub(DataPreprocessor.WEBSITE_REGEX, 'WEBSITE', text)

        # Replace "&amp" with empty string
        text = text.replace("&amp", '')

        # Replace special characters with spaces
        text = re.sub(DataPreprocessor.SPECIAL_CHARS_REGEX, ' ', text)

        # Remove apostrophes
        text = text.replace("'", "")

        # Replace hyphens, commas, and ampersands with spaces
        text = re.sub(r"[-&,]", ' ', text)

        # Replace punctuation marks with periods
        text = re.sub(r"[:;?!]", '.', text)

        # Replace multiple periods with a single period
        text = re.sub(r'\.+', '.', text)

        # Replace multiple periods separated by spaces with a single period
        text = re.sub(r'\. \.+', '.', text)

        return text.strip()

    @staticmethod
    def simplify_text(text):
        """
        Simplifies the cleaned text by removing placeholders for usernames and website links,
        removing punctuation marks, and eliminating short words and extra whitespace.

        Args:
            text (str): The cleaned text string.

        Returns:
            str: The further simplified text string.
        """
        # Replace usernames and website links with empty strings
        text = re.sub("USERNAME", '', text)
        text = re.sub("WEBSITE", '', text)

        # Remove punctuation marks, short words, and extra whitespace
        text = re.sub(r"\b\w{1,2}\b", '', text)
        text = re.sub(r"\s\s+", ' ', text)
        text = text.translate(str.maketrans(
            '', '', '!"$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
        return text.strip()

    @staticmethod
    def eliminate_stopwords(tokens):
        """
        Removes the stopwords from the given list of tokens and lemmatizes the words for normalization.

        Args:
            tokens (list): A list of word tokens.

        Returns:
            list: A list of tokens with stopwords removed.
        """
        stop = stopwords.words('english')
        stop.extend(['from', 'subject', 're', 'edu', 'use', 'via', 'like', 'ha'])
        lemmatizer = WordNetLemmatizer()
        new_tokens = []

        for token in tokens:
            # Handle contractions
            if "'" in token:
                parts = token.split("'")
                if parts[1].lower() == "t":
                    token = parts[0] + " not"
                elif parts[1].lower() == "ve":
                    token = parts[0] + " have"

            # Lemmatize token
            lemma = lemmatizer.lemmatize(token)

            # Check if token is a stopword or lemma
            if lemma not in stop:
                new_tokens.append(lemma)

        return new_tokens

    @staticmethod
    def separate_hashtags_usernames(text):
        """
        Splits hashtags and usernames in the given text.

        Args:
            text (str): The text string containing hashtags and usernames.

        Returns:
            str: The text string with hashtags and usernames split.
        """
        tokens = text.split()
        for i in range(len(tokens)):
            if (tokens[i][0] == '#') or (tokens[i][0] == '@'):
                tokens[i] = tokens[i].replace('#', '')
                tokens[i] = tokens[i].replace('@', '')
                out = re.split(r'(?<=[a-z])(?=[A-Z])', tokens[i])
                tokens[i] = ' '.join(out)
        tokens = ' '.join(tokens)
        return tokens

    @classmethod
    def preprocess_dataset(cls, data_file, data_type):
        """
        Processes the whole dataset performing all of the cleaning and preprocessing operations on the text data.

        Args:
            data_file (str): The path to the CSV data file.
            data_type (str): The type of data file.

        Returns:
            pandas.DataFrame: The preprocessed DataFrame.
        """
        # Read in data and clean text column
        # df = pd.read_csv(data_file, quotechar='"', encoding='utf-8')
        # specificlaly for sentiment 140
        df = pd.read_csv(data_file, encoding='latin', names=['polarity', 'id', 'date', 'query', 'user', 'text'])

        df['clean_text'] = df['text'].apply(cls.remove_usernames_weblinks_and_special_chars)


        # Remove non-ASCII characters
        df['clean_text'] = df['clean_text'].apply(unidecode.unidecode)
        df['clean_text'] = df['clean_text'].apply(cls.separate_hashtags_usernames)
        # Create simplified text column without usernames, websites, punctuation, and short words
        df['clean_text_simple'] = df['clean_text'].apply(cls.simplify_text)

        # Tokenize text, remove stopwords and lemmatize, and untokenize
        df['tokens'] = df['clean_text_simple'].apply(cls.tokenizer.tokenize)
        df['tokens'] = df['tokens'].apply(cls.eliminate_stopwords)
        df['text_simple'] = df['tokens'].apply(' '.join)

        # Remove tokens column
        df.drop('tokens', axis=1, inplace=True)
        # Rename columns for clarity
        columns = {
            'text': 'original_text',
            'clean_text': 'clean_text_with_usernames_and_hashtags',
            'clean_text_simple': 'clean_text_without_usernames_and_hashtags',
            'text_simple': 'clean_text_without_usernames_hashtags_or_stopwords'
        }

        if data_type == "label":
            columns['polarity'] = 'label'
        df = df.rename(columns=columns)

        # Reorder columns for readability
        cols_to_keep = ['original_text', 'clean_text_without_usernames_hashtags_or_stopwords']
        if data_type == "label":
            cols_to_keep.append('label')
        df = df[cols_to_keep]

        return df

    '''
    #Demo Usage
    preprocessor = DataPreprocessor()
    cleaned_data = preprocessor.preprocess_dataset(data_file='path_to_your_data.csv', type='your_type')

    '''

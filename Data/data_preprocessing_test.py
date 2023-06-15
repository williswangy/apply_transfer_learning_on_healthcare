import pytest
from Data.data_preprocessing import DataPreprocessor  # replace with your actual module name
import pandas as pd

class TestDataPreprocessor:
    @pytest.fixture(scope='module')
    def data_preprocessor(self):
        return DataPreprocessor()

    def test_remove_usernames_weblinks_and_special_chars(self, data_preprocessor):
        text = '@username check out this link http://somewebsite.com &amp awesome!'
        expected = 'USERNAME check out this link WEBSITE  awesome!'
        assert data_preprocessor.remove_usernames_weblinks_and_special_chars(text) == expected

        text = 'no usernames or links here &amp'
        expected = 'no usernames or links here '
        assert data_preprocessor.remove_usernames_weblinks_and_special_chars(text) == expected

    def test_simplify_text(self, data_preprocessor):
        text = 'USERNAME check out this link WEBSITE  awesome!'
        expected = 'check out this link awesome'
        assert data_preprocessor.simplify_text(text) == expected

        text = 'no username or website'
        expected = 'username website'
        assert data_preprocessor.simplify_text(text) == expected

    def test_eliminate_stopwords(self, data_preprocessor):
        tokens = ['this', 'is', 'a', 'test', 'sentence']
        expected = ['test', 'sentence']  # assuming 'this', 'is', 'a' are in the stopword list
        assert data_preprocessor.eliminate_stopwords(tokens) == expected

        tokens = ['no', 'stop', 'words']
        expected = ['no', 'stop', 'words']
        assert data_preprocessor.eliminate_stopwords(tokens) == expected

    def test_separate_hashtags_usernames(self, data_preprocessor):
        text = 'This is a #TestSentence with a @Username'
        expected = 'This is a Test Sentence with a Username'
        assert data_preprocessor.separate_hashtags_usernames(text) == expected

        text = 'No hashtags or usernames here'
        expected = 'No hashtags or usernames here'
        assert data_preprocessor.separate_hashtags_usernames(text) == expected

    def test_preprocess_dataset(self, data_preprocessor, tmp_path):
        d = tmp_path / "sub"
        d.mkdir()
        p = d / "data.csv"
        p.write_text('date,text,label\n2023-06-01,"@username check out this link http://somewebsite.com &amp awesome!",1')
        df = data_preprocessor.preprocess_dataset(p, data_type='label')
        assert df['original_text'][0] == '@username check out this link http://somewebsite.com &amp awesome!'
        assert df['clean_text_with_usernames_and_hashtags'][0] == 'USERNAME check out this link WEBSITE  awesome!'
        assert df['clean_text_without_usernames_and_hashtags'][0] == 'check out this link awesome'
        assert df['clean_text_without_usernames_hashtags_or_stopwords'][0] == 'check link awesome'
        assert df['label'][0] == 1

        # check that the preprocess_dataset works with DataFrame input
        data = {
            'date': ['2023-06-01'],
            'text': ['@username check out this link http://somewebsite.com &amp awesome!'],
            'label': [1]
        }
        df = pd.DataFrame(data)
        df_processed = data_preprocessor.preprocess_dataset(df, data_type='label')
        assert df_processed.equals(df)  # assuming

import pickle
from nltk.stem import WordNetLemmatizer 

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

LemmaTokenizer = LemmaTokenizer()

class ModelWrapper:
    """
    This class should be used to load and invoke the serialized model and any other required
    model artifacts for pre/post-processing.
    """

    def __init__(self):
        """
        Load the model + required pre-processing artifacts from disk. Loading from disk is slow,
        so this is done in `__init__` rather than loading from disk on every call to `predict`.

        Use paths relative to the project root directory.

        Tensorflow example:

            self.model = load_model("models/model.h5")

        Pickle example:

            with open('models/tokenizer.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)
        """
        with open('models/repo_lookup.pickle', 'rb') as handle:
                self.repo_lookup = pickle.load(handle)

        with open('models/sparse_vector_matrix.pickle', 'rb') as handle:
                self.sparse_vector_matrix = pickle.load(handle)

        with open('models/word_vectorizer.pickle', 'rb') as handle:
                self.word_vectorizer = pickle.load(handle)

    
    def text_recommender(self
                        , input_df
                        , word_vectorizer
                        , sparse_vector_matrix
                        , repo_lookup):
        
        input_df['bag_of_words'] = input_df.apply(lambda x: ' '.join(x), axis = 1)
        
        # vectorize the inputted string
        #inputted_vector = word_vectorizer.transform(pd.Series(str(input_string)))
        inputted_vector = word_vectorizer.transform(input_df['bag_of_words'])
        
        # calculate cosine similarity with existing matrix
        one_dimension_cosine_sim = cosine_similarity(inputted_vector, sparse_vector_matrix)
    
        # creating a Series with the similarity scores in descending order
        score_series = pd.Series(one_dimension_cosine_sim[0]).sort_values(ascending = False)
        # only show matches that have some similarity
        score_series = score_series[score_series>0]
    
        # getting the indexes of the 10 most similar repos
        top_10_indexes = list(score_series.iloc[1:11].index)
        
        # initializing the empty list of recommended repo
        
        recommended_repos = repo_lookup.loc[top_10_indexes]
            
        return recommended_repos

    def predict(self,data):
        """
        Returns model predictions.
        """
        # Add any required pre/post-processing steps here.
        return self.text_recommender(data)

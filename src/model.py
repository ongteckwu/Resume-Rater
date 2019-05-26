import pandas as pd
import numpy as np
import spacy
import math
import os
import time
import gensim
import os
import subprocess
import json

from gensim import corpora
from gensim.models import LdaModel
from gensim.test.utils import datapath
from typing import *
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from nltk import FreqDist
from .info_extractor import InfoExtractor
from .utils import (
    findDocumentsRecursive,
    generateDFFromData,
    loadDocumentIntoSpacy,
    getAllTokensAndChunks,
    loadDefaultNLP,
)


class RatingModel:
    class RatingModelError(Exception):
        pass

    def __init__(
        self,
        _type: Optional[str] = None,
        pre_trained_model_json: Optional[str] = None,
        spacy_nlp: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize a pre-trained or empty model
        """
        if _type is None:
            # empty model
            self.model = None
            self.keywords = None
        elif _type == "fixed":
            if pre_trained_model_json is None:
                raise RatingModel.RatingModel.Error("pre_trained_model_json is None")
            self.loadModelFixed(pre_trained_model_json)
        elif _type == "lda":
            if pre_trained_model_json is None:
                raise RatingModel.RatingModel.Error("pre_trained_model_json is None")
            self.loadModelLDA(pre_trained_model_json)
        else:
            raise RatingModel.RatingModelError(
                "type of test not valid. Either 'fixed' or 'lda'"
            )

        print("Loading nlp tools...")
        if spacy_nlp is None:
            # load default model
            self.nlp = loadDefaultNLP()
        else:
            self.nlp = spacy_nlp

        print("Loading pdf parser...")
        # takes some time
        from tika import parser

        self.parser = parser

    def loadModelFixed(self, model_json: str) -> None:
        """
        Function to load a pre-trained fixed model
        :param model_json: the json filename of the model
        """
        dirname = os.path.dirname(model_json)

        try:
            with open(model_json, "r") as f:
                j = json.load(f)
        except Exception as e:
            print(e)
            raise RatingModel.RatingModelError(
                "model_json %s is not a valid path" % model_json
            )

        try:
            path = os.path.join(dirname, j["model_csv"])
            self.model = pd.read_csv(path)
        except Exception as e:
            print(e)
            raise RatingModel.RatingModelError(
                "model_csv %s in model_json is not a valid path" % path
            )

        try:
            self.keywords = []
            path = os.path.join(dirname, j["keywords"])
            with open(path, "r") as f:
                for line in f:
                    if line:
                        self.keywords.append(line.strip())
        except Exception as e:
            print(e)
            raise RatingModel.RatingModelError(
                "model_keywords %s in model_json is not a valid path" % path
            )

        self._type = "fixed"

    def loadModelLDA(self, model_json: str) -> None:
        """
        Function to load a pre-trained ;da model
        :param model_csv: the json filename of the model
        """
        dirname = os.path.dirname(model_json)
        try:
            with open(model_json, "r") as f:
                j = json.load(f)
        except Exception as e:
            print(e)
            raise RatingModel.RatingModelError(
                "model_json %s is not a valid path" % model_json
            )

        try:
            path = os.path.join(dirname, j["model_csv"])
            self.model = pd.read_csv(path)
        except Exception as e:
            print(e)
            raise RatingModel.RatingModelError(
                "model_csv %s in model_json is not a valid path" % path
            )

        try:
            path = os.path.join(dirname, j["lda"])
            self.lda = LdaModel.load(path)
            self.dictionary = self.lda.id2word
        except Exception as e:
            print(e)
            raise RatingModel.RatingModelError(
                "lda %s in model_json is not a valid path" % path
            )

        try:
            path = os.path.join(dirname, j["top_k_words"])
            self.top_k_words = []
            with open(path, "r") as f:
                for line in f:
                    if line:
                        self.top_k_words.append(line.strip())
        except Exception as e:
            print(e)
            raise RatingModel.RatingModelError(
                "top_k_words %s in model_json is not a valid path" % path
            )

        self._type = "lda"

    def train(
        self,
        base_dir_path: str,
        train_type: str,
        model_name: str = "model",
        keywords: Optional[List[str]] = None,
    ) -> None:
        """
        Function to train a rating model. Model trained in self.model
        :param base_dir_path: directory of training data
        :param train_type: training type of "fixed" or "lda"
        :param model_name: name of model to save with
        :param keywords: keywords to use if "fixed"
        """
        if train_type == "fixed":
            if keywords is None:
                raise RatingModel.RatingModelError(
                    "For fixed keywords training, keywords given is None"
                )
        elif train_type != "lda":
            raise RatingModel.RatingModelError("No such train type: %s" % train_type)

        print("Loading resumes...")
        pdfs = findDocumentsRecursive(base_dir_path)
        if pdfs is None:
            raise RatingModel.RatingModelError(
                "base_dir_path %s is not a proper directory" % base_dir_path
            )

        if train_type == "fixed":
            self.__trainFixed(pdfs, model_name, keywords)
        elif train_type == "lda":
            self.__trainLDA(pdfs, model_name)

    def __keep_top_k_words(self, text):
        return [word for word in text if word in self.top_k_words]

    def __trainLDA(self, pdfs: List[str], model_name: str) -> None:
        """
        Hidden function to train an lda rating model. Model trained saved in self.model
        :param pdfs: list of pdfs
        :param model_csv_name: name of model to save with
        """
        self._type = "lda"

        pdf_data = defaultdict(list)

        print("Getting resume tokens and chunks...")
        for p in pdfs:
            # convert to spacy doc
            doc, _ = loadDocumentIntoSpacy(p, self.parser, self.nlp)
            seen_chunks_words, all_tokens_chunks = getAllTokensAndChunks(doc)
            pdf_data["name"].append(p)
            pdf_data["words"].append(list(seen_chunks_words))
            pdf_data["all_t_c"].append(list(all_tokens_chunks))

        pdf_df = pd.DataFrame(data=pdf_data)

        print("Pruning resumes to have minimum length...")
        # first get a list of all words
        all_words = np.concatenate([list(s) for s in pdf_df["words"]])
        # use nltk fdist to get a frequency distribution of all words
        fdist = FreqDist(all_words)
        # get min freq for words and filter all words whose counts are less
        # than the min freq
        # and then get k, the number of unique words
        min_count = int(math.floor(len(pdf_df["words"]) ** (0.5)))
        fdist = FreqDist({k: v for k, v in fdist.items() if v >= min_count})
        k = len(fdist)
        # keep the top k words; generate keywords from top k words
        self.top_k_words, _ = zip(*fdist.most_common(k))
        self.top_k_words = set(self.top_k_words)
        temp = pdf_df["words"].apply(self.__keep_top_k_words)
        # only keep articles with more than 50 tokens, otherwise too short
        new_pdf_df = pdf_df[temp.map(len) >= 50]  # magic

        # train LDA model
        self.dictionary, corpus, self.lda = self.__trainLDAModel(new_pdf_df)

        print("Training model...")
        data = defaultdict(list)
        pdf_count = 0
        for i in new_pdf_df.index.values:
            # training the keyword vectors
            p = pdf_df["name"][i]
            # pdf's bag-of-words
            bow = self.dictionary.doc2bow(new_pdf_df["words"].loc[i])
            doc_distribution = np.array(
                [tup[1] for tup in self.lda.get_document_topics(bow=bow)]
            )
            # get keywords and weights
            keywords = []
            all_pair_scores = []
            all_topic_scores = []
            all_diff_scores = []
            # take top 5 topics
            for j in doc_distribution.argsort()[-5:][::-1]:
                topic_prob = doc_distribution[j]
                # take top 5 words for each topic
                st = self.lda.show_topic(topicid=j, topn=5)
                sum_st = np.sum(list(map(lambda x: x[1], st)))
                pair_scores = []
                for pair in st:
                    keywords.append(pair[0])
                    pair_scores.append(pair[1])
                all_pair_scores.append(np.array(pair_scores))
                all_topic_scores.append(np.array(topic_prob))

            all_pair_scores = np.array(all_pair_scores)
            norm_all_pair_scores = all_pair_scores.T / np.sum(all_pair_scores, axis=1)
            norm_all_topic_scores = all_topic_scores / np.sum(all_topic_scores)
            all_diff_scores = (norm_all_pair_scores * norm_all_topic_scores).flatten()
            weights = pd.Series(all_diff_scores, index=keywords)
            weights.sort_values(ascending=False, inplace=True)

            # training the scores
            # does not matter whether words are filtered or not since the words
            # are only being used to compare with keywords generated
            seen_chunks_words = pdf_df["words"][i]
            all_tokens_chunks = pdf_df["all_t_c"][i]
            temp_out = self.__trainKMWM(seen_chunks_words, all_tokens_chunks, keywords)
            if temp_out is None:
                print(
                    "Either parser cannot detect text or too few words in resume for analysis. Most usually the former. Skip document."
                )
                continue
            km_scores, wm_scores = temp_out

            # average of km/wm scores for all keywords
            km_score = np.dot(weights.values, km_scores)
            wm_score = np.dot(weights.values, wm_scores)

            final_score = km_score * wm_score

            # add scores to data
            data["name"].append(os.path.basename(p))
            data["full_name"].append(p)
            data["score"].append(final_score)

            # counter
            pdf_count += 1
            if pdf_count % 100 == 0:
                print("%d data trained" % pdf_count)

        dirname = os.path.dirname(os.path.abspath(__file__))
        full_model_path = os.path.join(dirname, "models/model_lda", model_name + ".csv")
        full_lda_path = os.path.join(dirname, "models/model_lda", model_name + "_lda")
        full_top_k_words_path = os.path.join(
            dirname, "models/model_lda", model_name + "_top_k_words.txt"
        )
        full_json_path = os.path.join(dirname, "models/model_lda", model_name + ".json")

        self.model = generateDFFromData(data, save_csv=True, filename=full_model_path)
        # Save lda to disk.
        self.lda.save(full_lda_path)
        # save top_k_words as text
        with open(full_top_k_words_path, "w") as f:
            f.write("\n".join(self.top_k_words))
        jmodel = {
            "model_csv": model_name + ".csv",
            "lda": model_name + "_lda",
            "top_k_words": model_name + "_top_k_words.txt",
        }
        with open(full_json_path, "w") as f:
            json.dump(jmodel, f)

        print("Training done and models saved.")
        print("JSON model saved in %s" % full_json_path)
        print("Model csv saved in %s" % full_model_path)
        print("LDA saved in %s" % full_lda_path)
        print("Top K words text file saved in %s" % full_top_k_words_path)

    def __trainLDAModel(self, data, num_topics=50) -> Tuple[Any, Any, Any]:
        """
        Function trains the lda model
        We setup parameters like number of topics, the chunksize to use in Hoffman method
        We also do 2 passes of the data since this is a small dataset, so we want the distributions to stabilize
        :param data: pd df
        :param num_topics: number of topics to generate
        :return Tuple[Any, Any, Any]: the corpora dict, the corpus, and the lda model
        """
        print("Training Model...")
        num_topics = num_topics
        dictionary = corpora.Dictionary(data["words"])
        corpus = [dictionary.doc2bow(doc) for doc in data["words"]]
        t1 = time.time()
        # low alpha means each document is only represented by a small number of topics, and vice versa
        # low eta means each topic is only represented by a small number of words, and vice versa
        lda = LdaModel(
            corpus=corpus,
            num_topics=num_topics,
            id2word=dictionary,
            alpha=50 / num_topics,
            eta=0.01,
            minimum_probability=0.0,
            chunksize=100,
            passes=10,
        )
        t2 = time.time()
        print(
            "Time to train LDA model on ",
            len(data),
            "articles: ",
            (t2 - t1) / 60,
            "min",
        )
        return dictionary, corpus, lda

    def __trainFixed(self, pdfs: List[str], model_name: str, keywords: List[str]):
        """
        Hidden function to train a fixed rating model. Model trained saved in self.model
        :param pdfs: list of pdfs
        :param model_csv_name: name of model csv to save with
        :param keywords: keywords to train on
        """
        self.keywords = keywords
        self._type = "fixed"

        pdf_data = defaultdict(list)

        print("Getting resume tokens and chunks...")
        for p in pdfs:
            # convert to spacy doc
            doc, _ = loadDocumentIntoSpacy(p, self.parser, self.nlp)

            seen_chunks_words, all_tokens_chunks = getAllTokensAndChunks(doc)

            pdf_data["name"].append(p)
            pdf_data["words"].append(list(seen_chunks_words))
            pdf_data["all_t_c"].append(list(all_tokens_chunks))

        pdf_df = pd.DataFrame(data=pdf_data)

        data = defaultdict(list)
        pdf_trained_count = 0
        print("Training model...")
        for i in range(len(pdf_df["name"])):
            p = pdf_df["name"][i]
            # load pdf
            seen_chunks_words, all_tokens_chunks = (
                pdf_df["words"][i],
                pdf_df["all_t_c"][i],
            )
            temp_out = self.__trainKMWM(
                seen_chunks_words, all_tokens_chunks, self.keywords
            )
            if temp_out is None:
                print("Likely significant words count too small. Skip document.")
                continue
            km_scores, wm_scores = temp_out

            # average of km/wm scores for all keywords
            km_score = np.mean(km_scores)
            wm_score = np.mean(wm_scores)

            final_score = km_score * wm_score

            # add scores to data
            data["name"].append(os.path.basename(p))
            data["full_name"].append(p)
            data["score"].append(final_score)

            # counter
            pdf_trained_count += 1
            if pdf_trained_count % 100 == 0:
                print("%d data trained" % pdf_trained_count)

        dirname = os.path.dirname(os.path.abspath(__file__))
        full_model_path = os.path.join(
            dirname, "models/model_fixed", model_name + ".csv"
        )
        full_keywords_path = os.path.join(
            dirname, "models/model_fixed", model_name + "_keywords.txt"
        )
        full_json_path = os.path.join(
            dirname, "models/model_fixed", model_name + ".json"
        )

        self.model = generateDFFromData(data, save_csv=True, filename=full_model_path)

        keywords_file = "\n".join(self.keywords)
        with open(full_keywords_path, "w") as f:
            f.write(keywords_file)

        jmodel = {"model_csv": model_name + ".csv",
                  "keywords": model_name + "_keywords.txt"}
        with open(full_json_path, "w") as f:
            json.dump(jmodel, f)

        print("Training done and keywords and model saved.")
        print("JSON model saved in %s" % full_json_path)
        print("Model saved in %s" % full_model_path)
        print("Keywords saved in %s" % full_keywords_path)

    def __trainKMWM(
        self,
        seen_chunks_words: List[str],
        all_tokens_chunks: List[Any],
        keywords: List[str],
    ) -> Optional[Tuple[List[float], List[float]]]:
        """
        Hidden function to obtain KM and WM scores from keywords
        :param seen_chunks_words: n-grams of words in doc
        :param all_tokens_chunks: list of all tokens and chunks
        :param keywords: keywords to train on
        :return: Optional[Tuple[List[float], List[float]]]: kmscores, wmscores
                                                             if no errors.
                                                             Else None
        """

        # get word2vec correlation matrix of all tokens + keyword_tokens
        keywords_tokenized = self.nlp(" ".join(keywords))
        # prepare word embedding matrix
        pd_series_all = []

        # convert tokens and chunks into word embeddings and put them into a pd.Series
        for tc in all_tokens_chunks:
            name = tc.lemma_.lower()
            pd_series_all.append(pd.Series(tc.vector, name=name))

        # convert keywords into word embeddings and put them into a pd.Series
        for kwt in keywords_tokenized:
            name = kwt.text.lower()
            if name not in seen_chunks_words:
                pd_series_all.append(pd.Series(kwt.vector, name=name))
                seen_chunks_words.append(name)
        # get embedding matrix by concatenating all pd.Series
        embedd_mat_df = pd.concat(pd_series_all, axis=1).reset_index()
        corrmat = embedd_mat_df.corr()

        # top n words correlated to keyword
        top_n = list(range(10, 100, 10))
        km_scores = []
        wm_scores = []
        try:
            for kw in keywords:
                km_similarities = []
                wm_similarities = []
                # for top n words based on correlation to kw
                for n in top_n:
                    cols = np.append(
                        corrmat[kw]
                        .drop(keywords)
                        .sort_values(ascending=False)
                        .index.values[: n - 1],
                        kw,
                    )
                    cm = np.corrcoef(embedd_mat_df[cols].values.T)

                    # KM score
                    # avg of top n correlations wrt kw (less the keyword
                    # itself since it has corr = 1)
                    avg_sim = np.mean(cm[0, :][1:])
                    km_similarities.append(avg_sim)

                    # WM score
                    # avg of top n correlations (without kw)
                    # amongst each other

                    len_minus = (
                        cm.shape[0] - 1
                    )  # cm.shape to remove all the self correlations
                    len_minus_sq = len_minus ** 2
                    # 1. sum the correlations less the
                    # correlations with the keyword
                    # 2. subtract len_minus since there are
                    # len_minus autocorrelations
                    # 3. get mean by dividing the size of the rest
                    # i.e. (len_minus_sq - len_minus)
                    avg_wm = (np.sum(cm[1:, 1:]) - len_minus) / (
                        len_minus_sq - len_minus
                    )
                    wm_similarities.append(avg_wm)

                # get 8th degree of X and perform LR to get intercept
                X = np.array(top_n)
                Xes = [X]
                # for i in range(2, 9):
                #     Xes.append(X ** i)
                X_transformed = np.array(Xes).T

                lm = LinearRegression()

                # KM score
                y = np.array(km_similarities)
                lm.fit(X_transformed, y)
                km_scores.append(lm.intercept_)

                # WM score
                y = np.array(wm_similarities)
                lm.fit(X_transformed, y)
                wm_scores.append(lm.intercept_)

        except Exception as e:
            print(e)
            return None

        return km_scores, wm_scores

    def test(self, filename: str, info_extractor: Optional[InfoExtractor]):
        """
        Test a document and print the extracted information and rating
        :param filename: name of resume file
        :param info_extractor: InfoExtractor object
        """
        if self.model is None:
            raise RatingModel.RatingModelError("model is not loaded or trained yet")
        doc, _ = loadDocumentIntoSpacy(filename, self.parser, self.nlp)

        print("Getting rating...")
        if self._type == "fixed":
            if self.keywords is None:
                raise RatingModel.RatingModelError("Keywords not found")

            seen_chunks_words, all_tokens_chunks = getAllTokensAndChunks(doc)

            # scoring
            temp_out = self.__trainKMWM(
                list(seen_chunks_words), list(all_tokens_chunks), self.keywords
            )
            if temp_out is None:
                raise RatingModel.RatingModelError(
                    "Either parser cannot detect text or too few words in resume for analysis. Most usually the former."
                )
            km_scores, wm_scores = temp_out
            # average of km/wm scores for all keywords
            km_score = np.mean(km_scores)
            wm_score = np.mean(wm_scores)
            final_score = km_score * wm_score
        elif self._type == "lda":
            if self.lda is None or self.dictionary is None or self.top_k_words is None:
                raise RatingModel.RatingModelError("No LDA found")

            seen_chunks_words, all_tokens_chunks = getAllTokensAndChunks(doc)
            seen_chunks_words, all_tokens_chunks = (
                list(seen_chunks_words),
                list(all_tokens_chunks),
            )

            # scoring
            new_seen_chunks_words = self.__keep_top_k_words(seen_chunks_words)
            bow = self.dictionary.doc2bow(new_seen_chunks_words)
            doc_distribution = np.array(
                [tup[1] for tup in self.lda.get_document_topics(bow=bow)]
            )
            # get keywords and weights
            keywords = []
            all_pair_scores = []
            all_topic_scores = []
            all_diff_scores = []
            # take top 5 topics
            for j in doc_distribution.argsort()[-5:][::-1]:
                topic_prob = doc_distribution[j]
                # take top 5 words for each topic
                st = self.lda.show_topic(topicid=j, topn=5)
                sum_st = np.sum(list(map(lambda x: x[1], st)))
                pair_scores = []
                for pair in st:
                    keywords.append(pair[0])
                    pair_scores.append(pair[1])
                all_pair_scores.append(np.array(pair_scores))
                all_topic_scores.append(np.array(topic_prob))

            all_pair_scores = np.array(all_pair_scores)
            norm_all_pair_scores = all_pair_scores.T / np.sum(all_pair_scores, axis=1)
            norm_all_topic_scores = all_topic_scores / np.sum(all_topic_scores)
            all_diff_scores = (norm_all_pair_scores * norm_all_topic_scores).flatten()
            weights = pd.Series(all_diff_scores, index=keywords)
            weights.sort_values(ascending=False, inplace=True)

            temp_out = self.__trainKMWM(seen_chunks_words, all_tokens_chunks, keywords)
            if temp_out is None:
                print(
                    "Either parser cannot detect text or too few words in resume for analysis. Most usually the former. Skip document."
                )
            km_scores, wm_scores = temp_out

            # average of km/wm scores for all keywords
            km_score = np.dot(weights.values, km_scores)
            wm_score = np.dot(weights.values, wm_scores)

            final_score = km_score * wm_score

        # max_score = self.model["score"].iloc[0] - np.std(self.model["score"])
        # min_score = self.model["score"].iloc[-1]
        mean = np.mean(self.model["score"])
        sd = np.std(self.model["score"])

        rating = min(10, max(0, round(5 + (final_score-mean)/sd, 2)))
        if info_extractor is not None:
            print("-" * 10)
            info_extractor.extractFromFile(filename)
            print("-" * 10)
        print("Rating: %.1f" % rating)
        if info_extractor is not None:
            subprocess.call(["open", filename])


if __name__ == "__main__":
    r = RatingModel()
    # r.train(
    #     "/Users/teckwuong/coderoom/datasets/Resume&Job_Description/Original_Resumes",
    #     "lda",
    #     "model_lda.csv",
    #     "lda_test",
    # )
    filename = "abc"
    r.train(
        "/Users/teckwuong/coderoom/datasets/Resume&Job_Description/Original_Resumes",
        "fixed",
        keywords=["assets", "fund", "investment", "accounting", "trust", "strategy"],
    )
    infoExtractor = InfoExtractor(r.nlp, r.parser)
    r.test(filename, infoExtractor)

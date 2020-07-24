import multiprocessing
import os
import logging

from sklearn.cluster import Birch
from gensim.models.doc2vec import Doc2Vec

from assets import kw_utils


POSs = {
    "zhtw": {"Nb": 1, "Nc": 1, "Nd": 1},
    "en": {"NNP": 1},
    "zhcn": {
        "nr": 1,
        "ns": 1,
        "nt": 1,
        "nw": 1,
        "nz": 1,  # nr    人名    ns  地名    nt  机构名  nw  作品名 nz  其他专名
    },
}


DEFAULT_BRANCHING_FACTOR = int(os.getenv("DEFAULT_BRANCHING_FACTOR", 50))
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", 0.5))


def preprocessing(news, lang):
    """
    news: dataset.News
    """
    logger = logging.getLogger(__name__)

    # load keywords
    keywords = kw_utils.load_keyword(lang)

    # core preprocessing
    filtered = news.generate_keywords(keywords.all_stemming_values, POS=POSs[lang])
    segmented = list(filtered.keys())
    return segmented


def document_embeddings(x_train, vector_size=5, epochs=100):
    """
    x_train: list of TaggedDocument
    """
    logger = logging.getLogger(__name__)
    cores = multiprocessing.cpu_count()

    # PV-DBOW
    model_dm = Doc2Vec(
        dm=0,
        dbow_words=1,
        vector_size=vector_size,
        window=3,
        min_count=1,
        workers=cores,
    )
    model_dm.build_vocab(x_train)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=epochs)

    return model_dm


def cluster_by_birch(
    model_dm,
    x_train,
    branching_factor=DEFAULT_BRANCHING_FACTOR,
    threshold=DEFAULT_THRESHOLD,
):
    """
    model_dm: Doc2Vec,
    x_train: a list of TaggedDocument
    """
    logger = logging.getLogger(__name__)

    reverse_map = {}
    vectors = []
    # prepare vectors
    for text_list, label_str in x_train:
        logger.debug("text_list: {0}".format(text_list))
        logger.debug("label_str: {0}".format(label_str))
        vector = model_dm.infer_vector(text_list)
        reverse_map[vector.tobytes()] = label_str[0]
        vectors.append(vector)

    brc = Birch(
        branching_factor=branching_factor,
        n_clusters=None,
        threshold=threshold,
        compute_labels=True,
    )
    clrs = brc.fit_predict(vectors)
    logger.debug("clrs: {0}".format(clrs))

    # similarity
    clusters = [[] for i in range(len(list(set(clrs))))]
    for clr, vector in list(zip(clrs, vectors)):
        _, url = reverse_map[vector.tobytes()].split(":", 1)
        clusters[clr].append(url)
    return clusters

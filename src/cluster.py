# import system and third-party modules
import os
import sys
import logging
from operator import itemgetter

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from gensim.models.doc2vec import TaggedDocument
from sklearn import cluster, metrics

logging.basicConfig(level=logging.WARNING)

# import user-defined modules
from tasks import algo
from tasks import dataset as ds
from tasks import nlpclient

logger = logging.getLogger(__name__)


CKIPTAGGER_MODEL = Path(os.path.abspath(__file__)).parent.parent.joinpath(
    "data/models/ckiptagger/data"
    )

def preprocessing(news_dict_list):
    """
    value: list of {"id": "", "content": "", "subject": "", "lang": ""}
    """
    nlp_cli = nlpclient.NLPClient(CKIPTAGGER_MODEL_PATH=CKIPTAGGER_MODEL)
    nlp_cli.check_models()

    x_train_list = []

    # create dataset with list of news items
    dataset = ds.Dataset("dataset", seg_func=nlp_cli.post_document_analyze_syntax)
    for news_dict in news_dict_list:
        dataset.add_case(news_dict["id"], news_dict)
        segmented = algo.preprocessing(dataset[news_dict["id"]], news_dict["lang"])
        x_train_list.append(
            TaggedDocument(segmented, [news_dict["id"]])
            )

    return x_train_list


def evaluate(df, thres_list=[0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]):

    # preprocessing
    x_train_list = preprocessing(df.T.to_dict().values())

    # modeling and embedding projection
    model_dm = algo.document_embeddings(x_train_list, vector_size=3, epochs=100)

    vectors = []
    for text_list, label_str in x_train_list:
        vector = model_dm.infer_vector(text_list)
        vectors.append(vector)

    # clustring with average silhouette method
    results = []
    for thres in thres_list:
        brc = cluster.Birch(
            branching_factor=50, n_clusters=None, threshold=thres, compute_labels=True,
            )
        clrs = brc.fit_predict(vectors)
        logger.warning("clrs: {0}".format(clrs))
        silhouette_avg = metrics.silhouette_score(vectors, clrs)
        logger.warning("[thres {0}] silhouette_avg: {1}".format(thres, silhouette_avg))
        results.append(
            {"score": silhouette_avg, "clrs": clrs, "thres": thres, "vectors": vectors}
            )

    return results

def ploit_result(results):
    plt.figure(figsize=(8, 6))
    plt.xlabel("threshold")
    plt.ylabel("silhouette_avg")
    plt.plot(
        [res["thres"] for res in results],
        [res["score"] for res in results],
        linewidth=2,
        markersize=10,
        )
    plt.grid(True)
    plt.show()

    sorted_results = sorted(results, key=itemgetter("score"), reverse=True)

    plt.figure(figsize=(8, 6))
    scatter = plt.figure(figsize=(8, 6))
    ax = scatter.add_subplot(111, projection="3d")
    top_result = sorted_results[0]
    marks = {
        0: "8",  # octangon
        1: "s",  # square
        2: "*",  # star
        3: "+",  # plus
        4: "x",  # x
        5: "^",  # triangle
        6: "o",  # circle
        7: ".",  # point
        8: "D",  # diamond
        9: "H",  # diamond
        10: "1",  # diamond
        11: "2",  # diamond
        12: "3",  # diamond
        13: "4",  # diamond
        14: "h",  # diamond
    }
    for vec, clr in zip(top_result["vectors"], top_result["clrs"]):
        m = marks[clr]
        data1 = vec[0]
        data2 = vec[1]
        data3 = vec[2]
        print(
            "mark: {0}, data1: {1}, data2: {2}, data3: {3}".format(
                m, data1, data2, data3
                )
            )
        ax.scatter(data1, data2, data3, marker=m)

    plt.show()

if __name__ == "__main__":
    csv_file = Path(os.path.abspath(__file__)).parent.joinpath(
        "assets/news.csv"
        )

    # read csv to dataframe (df)
    df = pd.read_csv(csv_file)

    # preprocessing, modeling, and clustering
    results = evaluate(df)

    # output result as figures by metaploit
    ploit_result(results)


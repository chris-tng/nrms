import csv
from dataclasses import dataclass
import tqdm as tq
import numpy as np
from typing import Dict


@dataclass
class News:
    id: str
    category: str
    subcategory: str
    title: str
    abstract: str


# +
from typing import List

@dataclass
class Impression:
    id: str
    user_id: str
    datetime: str
    past_clicked: List
    clicked_news: List
    non_clicked_news: List


# -

def load_data(data_dir)

    with open(f"{data_dir}/news.tsv", "r") as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        news = [News(id=row[0], category=row[1], subcategory=row[2], title=row[3], abstract=row[4]) for row in tq.tqdm(read_tsv)]

    ids_to_news = list(set([n.id for n in news]))
    news_to_ids = {n:i for i, n in enumerate(ids_to_news)}
    news_content = {news_to_ids[n.id]: n.title + " " + n.abstract for n in news}
    
    with open(f"{data_dir}/behaviors.tsv", "r") as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")

        impressions = [
            Impression(id=row[0], user_id=row[1], datetime=row[2], past_clicked=row[3].split(), clicked_news=[i.split("-")[0] for i in row[-1].split() if i.endswith("-1")], non_clicked_news=[i.split("-")[0] for i in row[-1].split() if i.endswith("-0")])
            for row in tq.tqdm(read_tsv)
        ]

    return ids_to_news, news_to_ids, news_content, impressions


def read_glove(fname: str) -> Dict[str, np.ndarray]:
    w2v = {}
    with open(fname, "r") as f:
        for line in f:
            tokens = line.split()
            w = tokens[0]
            v = np.array([float(n) for n in tokens[1:]], dtype=np.float32)
            w2v[w] = v
    return w2v

import json
import random
import os
from itertools import chain
from tqdm import tqdm
from Bio import Entrez

Entrez.email = "XXX"

def process_searchrefiner_logs():
    def read_logs(name):
        with open(name, "r") as f:
            for line in f:
                yield json.loads(line)
    def process_logs(name):
        for x in tqdm(read_logs(name), desc=name):
            missing = ["32740058","1156684355"]
            for m in missing:
                if m in x["pmids"]:
                    x["pmids"].remove(m)
            quality = "medium quality: "
            if x["num_ret"] == 0 or x["num_ret"] > 1_000_000:
                quality = "low quality: "
            if len(x["pmids"]) == 0:
                continue
            yield {
                "pmid": random.choice(x["pmids"]),
                "q": x["query"],
                "quality": quality
            }
    items = [a for a in chain(
        process_logs("./data/2022-searchrefiner.log.jsonl"), 
        process_logs("./data/2024-searchrefiner.log.jsonl")
    )]
    pmids = [x["pmid"] for x in items]
    handle = Entrez.esummary(db="pubmed", id=",".join(pmids))
    records = list(Entrez.parse(handle))
    l = []
    for item, record in tqdm(zip(items, records)):
        item["d"] = record["Title"]
        # Maybe do something here with the quality?
        del item["pmid"]
        del item["quality"]
        l.append(item)
    return l

if __name__ == "__main__":
    with open("data/pubmed-queries", "r", encoding="latin-1") as f:
        Q = set()
        with open("data/pubmed-queries.short", "w") as f2:
            for line in f:
                q = "|".join(line.split("|")[2:])
                qq = q.strip()
                if " AND " in q \
                    or " OR " in q \
                    or qq.startswith("("):
                    if not qq.startswith("#") and not qq.startswith("1"):
                        if len(qq) > 64:
                            if qq not in Q:
                                Q.add(qq)
                                f2.write(q)
    
    with open("data/pubmed-queries.short", "r") as f:
        Q = []
        for line in tqdm(f, desc="getting pmids"):
            handle = Entrez.esearch(db="pubmed", retmax=1, term=line)
            record = Entrez.read(handle)
            handle.close()
            if int(record["Count"]) > 0:
                pmid = record["IdList"][0]
                Q.append((line, pmid))
        
        # Some example code that should get the titles for each PMID.
        # Commented out because it will be pretty slow, likely.
        # handle = Entrez.esummary(db="pubmed", id=",".join(pmids))
        # records = list(Entrez.parse(handle))
        # with open("data/pubmed-queries.jsonl", "w") as f:
        #     for pmid in tqdm(Q, desc="getting pmid titles"):
        #         f.write(json.dumps)



    with open("data/raw.jsonl", "r") as f:
        x = []
        for line in tqdm(f):
            d = json.loads(line)
            if d["pubmed"] is not None \
                and d["pubmed"].strip().startswith("(") \
                and len(d["pubmed"].strip()) > 64 \
                and "#" not in d["pubmed"].strip()[:8]:
                x.append({"d": d["title"], "q": d["pubmed"]})

    if not os.path.exists("./data/training-logs.jsonl"):
        with open("./data/training-logs.jsonl", "w") as f:
            for item in process_searchrefiner_logs():
                f.write(json.dumps(item)+"\n")
    
    with open("./data/training-logs.jsonl", "r") as f:
        for line in f:
            x.append(json.loads(line))

    with open("data/training.jsonl", "w") as f:
        for l in tqdm(x):
            l["q"] = l["q"].replace("\\t"," ")\
                .replace("\\n"," ")\
                .replace("\\r"," ")\
                .replace("\\","")\
                .replace('\"', '"')\
                .replace("\u201c", '"').replace("\u201d", '"')
            f.write(json.dumps(l)+"\n")
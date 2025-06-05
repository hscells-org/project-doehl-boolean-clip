import json
import random
import os
from itertools import chain
from tqdm import tqdm
from Bio import Entrez
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

Entrez.email = "simon.doehl@student.uni-tuebingen.de"

class PubmedQueries:
    def __init__(self):
        self.bool_key = "bool_query"
        self.nl_key = "nl_query"

    def process_searchrefiner_logs(self):
        def read_logs(name):
            with open(name, "r") as f:
                for line in f:
                    yield json.loads(line)

        def process_logs(name):
            for x in tqdm(read_logs(name), desc=name):
                missing = ["32740058", "1156684355"]
                for m in missing:
                    if m in x["pmids"]:
                        x["pmids"].remove(m)
                source = "pubmed-medium"
                if x["num_ret"] == 0 or x["num_ret"] > 1_000_000:
                    source = "pubmed-low"
                if len(x["pmids"]) == 0:
                    continue
                yield {
                    "pmid": random.choice(x["pmids"]),
                    self.bool_key: x["query"],
                    "source": source
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
            item[self.nl_key] = record["Title"]
            # Maybe do something here with the quality?
            del item["pmid"]
            # del item["quality"]
            l.append(item)
        return l


    def main(self):
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

        CACHE_FILE = "data/pubmed-cache.json"
        # Load or initialize cache
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as cf:
                cache = json.load(cf)
        else:
            cache = {}

        # Read queries
        queries = []
        with open("data/pubmed-queries.short", "r") as f:
            for line in f:
                query = line.strip()
                if query:
                    queries.append(query)

        # Search for PMIDs
        results = []
        i = 0
        for query in tqdm(queries, desc="Searching PMIDs"):
            if query in cache.keys():
                pmid = cache[query]
                results.append({self.bool_key: query, 'pmid': pmid})
                continue
            i += 1
            if i > 3000: break
            try:
                handle = Entrez.esearch(db="pubmed", term=query, retmax=1)
                record = Entrez.read(handle)
                handle.close()
            except Exception as e:
                print(f"Error searching '{query}': {e}")
                continue

            count = int(record.get("Count", 0))
            if count > 0:
                pmid = record["IdList"][0]
                cache[query] = pmid
                results.append({self.bool_key: query, 'pmid': pmid})

        # Save updated cache
        with open(CACHE_FILE, 'w') as cf:
            json.dump(cache, cf, indent=2)

        # Fetch summaries for all found PMIDs
        pmids = [r['pmid'] for r in results]
        if pmids:
            handle = Entrez.esummary(db="pubmed", id=','.join(pmids))
            summaries = list(Entrez.parse(handle))
            handle.close()

            # Merge summaries into results
            pmid_to_summary = {rec['Id']: rec for rec in summaries}
            for r in results:
                summary = pmid_to_summary.get(r['pmid'], {})
                r['nl_query'] = summary.get('Title', '')
                r['source'] = 'pubmed-query'

        # Save to JSONL
        with open("data/pubmed-queries.jsonl", 'w') as f:
            for rec in results:
                f.write(json.dumps(rec) + '\n')

        with open("data/raw.jsonl", "r") as f:
            x = []
            for line in tqdm(f):
                d = json.loads(line)
                if d["pubmed"] is not None \
                    and d["pubmed"].strip().startswith("(") \
                    and len(d["pubmed"].strip()) > 64 \
                        and "#" not in d["pubmed"].strip()[:8]:
                    x.append({self.nl_key: d["title"], self.bool_key: d["pubmed"], "source": "raw-jsonl"})

        if not os.path.exists("./data/training-logs.jsonl"):
            with open("./data/training-logs.jsonl", "w") as f:
                for item in self.process_searchrefiner_logs():
                    f.write(json.dumps(item) + "\n")

        with open("./data/training-logs.jsonl", "r") as f:
            for line in f:
                x.append(json.loads(line))

        with open("data/pubmed-queries.jsonl", "r") as f:
            for line in f:
                x.append(json.loads(line))

        with open("data/training.jsonl", "w") as f:
            for l in tqdm(x):
                l[self.bool_key] = l[self.bool_key].replace("\\t", " ")\
                    .replace("\\n", " ")\
                    .replace("\\r", " ")\
                    .replace("\\", "")\
                    .replace('\"', '"')\
                    .replace("\u201c", '"').replace("\u201d", '"')
                f.write(json.dumps(l) + "\n")


def process_TAR(path = os.path.join("tar", "2017-TAR"), verbose=False):
    bases = ['testing', 'training']
    bases = [os.path.join(path, base) for base in bases]

    def process_topic_file(path):
        """Read a topic file, extract title (first line) and query (rest),
        return dict with 'title' and 'query'."""
        with open(path, 'r', encoding='utf-8') as f:
            # strip trailing newline but keep blank lines for separation
            lines = [line.rstrip() for line in f]
        # drop any leading/trailing blank lines
        while lines and not lines[0].strip(): lines.pop(0)
        while lines and not lines[-1].strip(): lines.pop()
        if not lines: return None
        # first non-blank line is title
        title = lines[2].strip().removeprefix("Title: ")
        # remaining non-blank lines form the query
        query_lines = [ln.strip() for ln in lines[3:] if ln.strip()]
        query = ' '.join(query_lines).removeprefix("Query: ")
        query = query.split(" Pids:")[0]
        return {'nl_query': title, 'bool_query': query, 'source': 'TAR'}

    def iter_topic_dirs(base_dirs):
        """Yield paths to all files in each 'topics' directory under base_dirs."""
        for base in base_dirs:
            # in testing it's 'topics', in training it's 'topics_train'
            for sub in ('topics', 'topics_train'):
                dirpath = os.path.join(base, sub)
                if not os.path.isdir(dirpath):
                    continue
                for fname in os.listdir(dirpath):
                    fpath = os.path.join(dirpath, fname)
                    if os.path.isfile(fpath):
                        yield fpath

    data = []
    for topic_file in iter_topic_dirs(bases):
        rec = process_topic_file(topic_file)
        if rec is None: continue
        data.append(rec)
        if verbose: print(rec)
    with open('data/TAR_data.jsonl', 'w') as f:
        for rec in data:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

def paths_to_dataset(train_and_test: str|list[str], test_only: str|list[str] = None, split_perc: float = 0.1):
    def ensure_list(v):
        return [v] if isinstance(v, str) else v
    train_and_test = ensure_list(train_and_test)
    test_only = ensure_list(test_only)

    test_dfs = []
    train_dfs = []
    def sort(data, split: bool = True):
        for source, data in df.groupby("source"):
            if split:
                train_part, test_part = train_test_split(data, test_size=split_perc, random_state=42)
                train_dfs.append(train_part)
            test_dfs.append((source, test_part if split else data))

    for path in train_and_test:
        df = pd.read_json(path, lines=True)
        sort(df)

    if test_only is not None:
        for path in test_only:
            df = pd.read_json(path, lines=True)
            sort(df, False)

    train_df = pd.concat(train_dfs).reset_index(drop=True)
    # test_df = pd.concat(test_dfs).reset_index(drop=True)

    train_dataset = Dataset.from_pandas(train_df)
    # test_dataset = Dataset.from_pandas(test_df)
    test_datasets = {group: Dataset.from_pandas(df) for group, df in test_dfs}
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_datasets
    })
    return dataset_dict
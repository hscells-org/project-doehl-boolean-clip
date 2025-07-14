import json
import random
import os
from itertools import chain
from tqdm import tqdm
from Bio import Entrez
from transformers import AutoTokenizer
from deduplication import remove_similar_jaccard

Entrez.email = "simon.doehl@student.uni-tuebingen.de"

class PubmedQueries:
    def __init__(self):
        self.bool_key = "bool_query"
        self.nl_key = "nl_query"
        self.search_ref_high_w = 1
        self.search_ref_low_w = 0.5
        self.default_w = 0.1

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
                source = "pubmed-searchrefiner"
                quality = self.search_ref_high_w
                if x["num_ret"] == 0 or x["num_ret"] > 1_000_000:
                    quality = self.search_ref_low_w
                if len(x["pmids"]) == 0:
                    continue
                yield {
                    "pmid": random.choice(x["pmids"]),
                    self.bool_key: x["query"],
                    "source": source,
                    "quality": quality,
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
            del item["pmid"]
            l.append(item)
        return l

    def queries_to_short(self):
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

    def main(self, optional_steps=False):
        if optional_steps: self.queries_to_short()

        CACHE_FILE = "data/pubmed-cache.json"
        cache = {}
        # Load or initialize cache
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, 'r') as cf:
                    cache = json.load(cf)
        except: print("Couldn't load pubmed cache")

        # Read queries
        queries = []
        with open("data/pubmed-queries.short", "r") as f:
            for line in f:
                query = line.strip()
                if query:
                    queries.append(query)

        queries = remove_similar_jaccard(queries)

        # Search for PMIDs
        results = []
        new_cache = {}
        i = 0
        tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
        for query in tqdm(queries, desc="Searching PMIDs"):
            tokenized = ' '.join([str(t) for t in tokenizer.encode(query)])
            try:
                record = cache[tokenized]
            except:
                i += 1
                try:
                    handle = Entrez.esearch(db="pubmed", term=query, retmax=10)
                    record = Entrez.read(handle)
                    handle.close()
                except Exception as e:
                    print(f"Error searching '{query}': {e}")
                    continue

            count = int(record.get("Count", 0))
            if count > 0:
                pmid = record["IdList"][0]
                new_cache[tokenized] = record
                results.append({self.bool_key: query, 'pmid': pmid})
            if i % 100 == 0:
                # Save updated cache
                with open(CACHE_FILE, 'a') as cf:
                    json.dump(new_cache, cf, indent=2)

        # Save updated cache
        cache.update(new_cache)
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
                r['quality'] = self.default_w

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
                    x.append({self.nl_key: d["title"],
                              self.bool_key: d["pubmed"],
                              "source": "raw-jsonl",
                              "quality": self.default_w})

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
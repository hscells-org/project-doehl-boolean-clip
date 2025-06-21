import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import os
import json
from deduplication import remove_similar_jaccard

def paths_to_dataset(train_and_test: str|list[str],
                     split_perc: float = 0.1,
                     test_only_sources: list = [],
                     max_test_size: int = 3000):
    def ensure_list(v):
        return [v] if isinstance(v, str) else v
    train_and_test = ensure_list(train_and_test)

    test_dfs = []
    train_dfs = []
    def sort(data):
        for source, data in df.groupby("source"):
            if source not in test_only_sources:
                train_part, test_part = train_test_split(data, test_size=split_perc, random_state=42)
                train_part['quality'] = train_part['quality'] / train_part.shape[0]
                train_dfs.append(train_part)
                test_dfs.append((source, test_part[:max_test_size]))
            else:
                test_dfs.append((source, data[:max_test_size]))

    for path in train_and_test:
        df = pd.read_json(path, lines=True)
        sort(df)

    train_df = pd.concat(train_dfs).reset_index(drop=True)

    for i, (source, df) in enumerate(test_dfs):
        col = 'bool_query'
        deduped = remove_similar_jaccard(df[col])
        # remove similar
        df = df[df[col].isin(deduped)]
        # remove exact duplicates
        df = df.loc[~df[col].duplicated(keep='first')]
        test_dfs[i] = (source, df)

    train_dataset = Dataset.from_pandas(train_df)
    test_datasets = {group: Dataset.from_pandas(df) for group, df in test_dfs}

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_datasets
    })
    return dataset_dict


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
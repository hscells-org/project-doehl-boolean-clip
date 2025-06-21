import re
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

_norm_sets = None
_threshold = None

def init_worker(norm_sets, threshold):
    global _norm_sets, _threshold
    _norm_sets = norm_sets
    _threshold = threshold

def normalize(text):
    return set(re.findall(r'\w+', text.lower()))

def jaccard(A: set, B: set) -> float:
    if not A and not B: return 1.0
    return len(A & B) / len(A | B)

def _worker(i):
    Si = _norm_sets[i]
    sims = []
    for j, Sj in enumerate(_norm_sets):
        if j != i and jaccard(Si, Sj) >= _threshold: sims.append(j)
    return i, sims

def find_similar_jaccard(data, threshold=0.8):
    norm_sets = [normalize(q) for q in data]
    n = len(data)
    similar_dict = {}
    max_workers = cpu_count()//2

    with Pool(
        processes=max_workers,
        initializer=init_worker,
        initargs=(norm_sets, threshold)
    ) as pool:
        for i, sims in tqdm(
            pool.imap_unordered(_worker, range(n)),
            total=n,
            desc="Deduplicating",
        ):
            similar_dict[i] = sims
    return similar_dict

def remove_similar_jaccard(data, threshold=0.8):
    similar_dict = find_similar_jaccard(data, threshold)

    to_remove = set()
    unique = []
    data = list(data)
    for i in range(len(data)):
        if i in to_remove: continue
        unique.append(data[i])
        to_remove.update(similar_dict[i])
    return unique

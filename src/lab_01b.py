import numpy as np

tokens = np.array(["kot", "pies", "dom", ".", "samochod"])
logits = np.array([2.4, 1.7, 0.2, 1.1, -0.5], dtype=float)

def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exps = np.exp(shifted)
    return exps / exps.sum()

probs = softmax(logits)
for token, logit, propability in zip(tokens, logits, probs):
    print(f"{token!r:12} logit={logit:5.2f} p={propability:6.3f}")

print("suma:", probs.sum())
print("greedy:", tokens[np.argmax(probs)])

# -------------------------------------

def softmax_temp(values: np.ndarray, temp: float) -> np.ndarray:
    if temp <=0:
        raise ValueError("Blad!")
    return softmax(logits / temp)

for temperature in [0.2, 0.7, 1.0, 1.5]:
    probs = softmax_temp(logits, temperature)
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    print(temperature, np.round(probs, 3), "entropia=", round(entropy, 3))

# -------------------------------------

def filter_top_k(probs: np.ndarray, top_k: int) -> np.ndarray:
    filtered = probs.astype(float, copy=True)
    order = np.argsort(-filtered, kind="stable")
    keep = np.zeros(len(filtered), dtype=bool)
    keep[order[:top_k]] = True
    filtered[~keep] = 0.0
    filtered /= filtered.sum()
    return filtered

def filter_top_p(probs: np.ndarray, top_p: float = 1.0) -> np.ndarray:
    filtered = probs.astype(float, copy=True)
    order = np.argsort(-filtered, kind="stable")
    cumulative = np.cumsum(filtered[order])
    last = np.searchsorted(cumulative, top_p, side="left")
    keep = np.zeros(len(filtered), dtype=bool)
    keep[order[:last + 1]] = True
    filtered[~keep] = 0.0
    filtered /= filtered.sum()
    return filtered

def filter_top_k_top_p(probs: np.ndarray, top_k: int | None = None, top_p: float = 1.0) -> np.ndarray:
    if probs.ndim != 1 or len(probs) == 0:
        raise ValueError("Blad!")
    if np.any(probs < 0) or not np.isclose(probs.sum(), 1.0):
        raise ValueError("Blad!")
    if not 0 < top_p <= 1:
        raise ValueError("Blad!")

    filtered = probs.astype(float, copy=True)
    if(top_k) is not None:
        filtered = filter_top_k(probs, top_k)
    if(top_p) < 1.0:
        filtered = filter_top_p(probs, top_p)

    return filtered

rng = np.random.default_rng(42)
base = softmax_temp(logits, 0.8)

print("top-k:", np.round(filter_top_k_top_p(base, top_k=3), 3))
print("top-p:", np.round(filter_top_k_top_p(base, top_p=0.9), 3))
filtered = filter_top_k_top_p(base, top_k=3, top_p=0.9)
print("oba:", np.round(filtered, 3))

counts = {token: 0 for token in tokens}
for _ in range(1000):
    selected = rng.choice(tokens, p=filtered)
    counts[selected] += 1

print(counts)
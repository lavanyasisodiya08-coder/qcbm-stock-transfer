"""
Classical augmentation baselines - what the QCBM has to beat to justify
using a quantum model at all.
"""

import numpy as np
import pandas as pd
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer

import config
from src.qcbm_model import index_to_bigram, empirical_bigram_distribution, N_STATES


def bootstrap_augment(seq, n_synthetic, block_size=10, seed=None):
    """Block bootstrap from the SAME (target) sequence. No cross-stock
    information used - the 'no transfer' baseline."""
    rng = np.random.default_rng(seed)
    if len(seq) <= block_size:
        block_size = max(1, len(seq) // 2)
    out = []
    while len(out) < n_synthetic:
        start = rng.integers(0, len(seq) - block_size + 1)
        out.extend(seq[start:start + block_size])
    return out[:n_synthetic]


def fit_markov_transition(seq):
    labels = config.BUCKET_LABELS
    counts = pd.DataFrame(0, index=labels, columns=labels, dtype=float)
    for a, b in zip(seq[:-1], seq[1:]):
        counts.loc[a, b] += 1
    row_sums = counts.sum(axis=1)
    row_sums[row_sums == 0] = 1
    return counts.div(row_sums, axis=0)


def markov_cross_augment(source_seq, n_synthetic, start_bucket=None, seed=None):
    """Classical cross-stock transfer baseline: fit a Markov chain on the
    SOURCE (large-cap) sequence, generate from it. Direct classical
    analogue of what the QCBM does - the key quantum-vs-classical comparison."""
    rng = np.random.default_rng(seed)
    trans = fit_markov_transition(source_seq)
    current = start_bucket or rng.choice(config.BUCKET_LABELS)
    seq = [current]
    for _ in range(n_synthetic - 1):
        probs = trans.loc[current].values
        if probs.sum() == 0:
            probs = np.ones(len(config.BUCKET_LABELS)) / len(config.BUCKET_LABELS)
        nxt = rng.choice(config.BUCKET_LABELS, p=probs)
        seq.append(nxt)
        current = nxt
    return seq


def categorical_gan_augment(source_seq, n_synthetic, n_steps=300, seed=None):
    """Lightweight classical GAN over the 25 bigram categories, trained
    with autograd. A more expressive classical baseline than plain Markov."""
    rng = np.random.default_rng(seed)
    target_dist = pnp.array(empirical_bigram_distribution(source_seq))

    gen_logits = pnp.array(rng.normal(0, 0.1, N_STATES), requires_grad=True)
    disc_w = pnp.array(rng.normal(0, 0.1, N_STATES), requires_grad=True)
    disc_b = pnp.array(0.0, requires_grad=True)

    def softmax(x):
        e = pnp.exp(x - pnp.max(x))
        return e / pnp.sum(e)

    def sigmoid(x):
        return 1 / (1 + pnp.exp(-x))

    opt_g = AdamOptimizer(stepsize=0.05)
    opt_d = AdamOptimizer(stepsize=0.05)

    for step in range(n_steps):
        gen_dist = softmax(gen_logits)

        def disc_loss(dw, db):
            real_score = sigmoid(pnp.dot(dw, target_dist) + db)
            fake_score = sigmoid(pnp.dot(dw, gen_dist) + db)
            return -(pnp.log(real_score + 1e-10) + pnp.log(1 - fake_score + 1e-10))

        disc_w, disc_b = opt_d.step(disc_loss, disc_w, disc_b)

        def gen_loss(gl):
            gd = softmax(gl)
            fake_score = sigmoid(pnp.dot(disc_w, gd) + disc_b)
            return -pnp.log(fake_score + 1e-10)

        gen_logits = opt_g.step(gen_loss, gen_logits)

    final_dist = np.array(softmax(gen_logits))
    final_dist = final_dist / final_dist.sum()
    idxs = rng.choice(N_STATES, size=n_synthetic, p=final_dist)
    seq = []
    for idx in idxs:
        if idx < 25:
            _, t = index_to_bigram(idx)
        else:
            t = rng.choice(config.BUCKET_LABELS)
        seq.append(t)
    return seq

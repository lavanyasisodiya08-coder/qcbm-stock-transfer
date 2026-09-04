"""
Quantum Circuit Born Machine (QCBM) - learns the joint distribution over
(current_bucket, next_bucket) bigrams from a SOURCE stock (large-cap), then
samples synthetic bigram chains that become synthetic token sequences for
a TARGET stock (small-cap). This is the cross-stock quantum transfer piece.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

import config

N_QUBITS = 5
N_STATES = 2 ** N_QUBITS
N_BUCKETS = len(config.BUCKET_LABELS)


def bigram_index(from_label, to_label):
    labels = config.BUCKET_LABELS
    return labels.index(from_label) * N_BUCKETS + labels.index(to_label)


def index_to_bigram(idx):
    labels = config.BUCKET_LABELS
    return labels[idx // N_BUCKETS], labels[idx % N_BUCKETS]


def empirical_bigram_distribution(seq):
    dist = np.zeros(N_STATES)
    for a, b in zip(seq[:-1], seq[1:]):
        dist[bigram_index(a, b)] += 1
    total = dist.sum()
    if total > 0:
        dist = dist / total
    return dist


dev = qml.device("default.qubit", wires=N_QUBITS)


def circuit(params, n_layers):
    for l in range(n_layers):
        for q in range(N_QUBITS):
            qml.RY(params[l, q, 0], wires=q)
            qml.RZ(params[l, q, 1], wires=q)
        for q in range(N_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
    return qml.probs(wires=range(N_QUBITS))


qnode = qml.QNode(circuit, dev, interface="autograd")


def kl_divergence(p_target, p_model, eps=1e-10):
    p_target = pnp.clip(p_target, eps, 1.0)
    p_model = pnp.clip(p_model, eps, 1.0)
    return pnp.sum(p_target * pnp.log(p_target / p_model))


class QCBM:
    def __init__(self, n_layers=4, seed=None):
        self.n_layers = n_layers
        seed = seed if seed is not None else config.RANDOM_SEED
        rng = np.random.default_rng(seed)
        self.params = pnp.array(
            rng.uniform(0, 2 * np.pi, size=(n_layers, N_QUBITS, 2)),
            requires_grad=True,
        )
        self.target_dist = None
        self.loss_history = []

    def fit(self, source_seq, n_steps=150, lr=0.1, verbose=True):
        self.target_dist = pnp.array(empirical_bigram_distribution(source_seq))
        opt = qml.AdamOptimizer(stepsize=lr)

        def cost(params):
            model_dist = qnode(params, self.n_layers)
            return kl_divergence(self.target_dist, model_dist)

        for step in range(n_steps):
            self.params, loss = opt.step_and_cost(cost, self.params)
            self.loss_history.append(float(loss))
            if verbose and (step % 25 == 0 or step == n_steps - 1):
                print(f"  [QCBM] step {step:4d}  KL={loss:.5f}")
        return self

    def sampled_distribution(self):
        return np.array(qnode(self.params, self.n_layers))

    def generate_sequence(self, length, start_bucket=None, seed=None):
        rng = np.random.default_rng(seed)
        probs = self.sampled_distribution()
        table = np.zeros((N_BUCKETS, N_BUCKETS))
        for idx in range(N_BUCKETS * N_BUCKETS):
            f, t = index_to_bigram(idx)
            table[config.BUCKET_LABELS.index(f), config.BUCKET_LABELS.index(t)] = probs[idx]
        row_sums = table.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        table = table / row_sums

        current = start_bucket or rng.choice(config.BUCKET_LABELS)
        seq = [current]
        for _ in range(length - 1):
            row = table[config.BUCKET_LABELS.index(current)]
            nxt = rng.choice(config.BUCKET_LABELS, p=row)
            seq.append(nxt)
            current = nxt
        return seq

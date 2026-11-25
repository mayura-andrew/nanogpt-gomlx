# Pre-Series Primer: Foundations Labs (`00-foundations-labs`)

These four labs bridge first-principles mathematics and pure-Go loops with **GoMLX's** symbolic computation graphs and training abstractions. Complete them side-by-side before starting Module 1 (`01-setup/`).

| Lab | Concept | Pure Go (`scratch/`) | GoMLX (`gomlx/`) | NanoGPT Connection |
| :--- | :--- | :--- | :--- | :--- |
| **Lab 01** | Universal Computation & Logic Gates | 5-NAND Perceptron Half Adder | Symbolic `*graph.Node` wiring (`Mul`, `Add`, `Where`) | Causal masking (`graph.Where`) & residual streams (`graph.Add`) |
| **Lab 02** | Smooth Neurons & Gradient Descent | Sigmoid + manual chain rule on OR gate | `model.Store` variables, `layers.Dense`, & autodiff | Trainable embeddings (`wte`, `wpe`) & `Q`/`K`/`V` projections |
| **Lab 03** | Multi-Layer Perceptrons & Backprop | 2-layer MLP (`2 -> 2 -> 1`) solving XOR | Scoped `model.Scope` layers & automatic backprop | Stacked Transformer Feed-Forward Networks (`d_model -> 4*d_model -> d_model`) |
| **Lab 04** | Multi-Class Softmax & Cross-Entropy | ReLU + numerically stable Softmax + CE loop | `train.NewTrainer` + `SparseCategoricalCrossEntropyLogits` | Next-token prediction head & sequence training loop |

## Running the Labs

```bash
# Run any pure-Go baseline
go run ./00-foundations-labs/lab01-half-adder/scratch

# Run the GoMLX equivalent (pure-Go backend by default, or BACKEND=xla)
GOMLX_BACKEND=go go run ./00-foundations-labs/lab01-half-adder/gomlx
GOMLX_BACKEND=xla go run ./00-foundations-labs/lab01-half-adder/gomlx
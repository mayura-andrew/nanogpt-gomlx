### 1. `00-foundations-labs/lab01-half-adder/README.md`


# Lab 01: Universal Computation & The 5-NAND Half Adder

Before training neural networks with gradient descent, this lab demonstrates that simple threshold units (perceptrons) can perform universal logic—and contrasts imperative Go execution (`scratch/`) against symbolic computation graphs (`gomlx/`).

## 1. The NAND Perceptron

A single perceptron computes $z = w_1 x_1 + w_2 x_2 + b$ and fires `1` if $z > 0$ (else `0`). Setting $w_1 = -2$, $w_2 = -2$, and $b = 3$ implements an exact **NAND** gate:

| $x_1$ | $x_2$ | Weighted Sum ($-2x_1 - 2x_2 + 3$) | Output ($z > 0$) |
| :---: | :---: | :---: | :---: |
| 0 | 0 | $3$ | **1** |
| 0 | 1 | $1$ | **1** |
| 1 | 0 | $1$ | **1** |
| 1 | 1 | $-1$ | **0** |

Because NAND gates are functionally complete, wiring **5 NAND perceptrons** together computes binary addition (`Sum` and `Carry`):

```text
x1 ──┬──► [N1] ──a──┬──► [N2] ──b──┐
     │     ▲        │     ▲        ├──► [N4] ──► Sum
     └─────┼────────┘     │        │
           │              │        │
x2 ────────┴────────┬─────┼────────┼──► [N3] ──c──┘
                    │     │        ▲
                    └─────┴────────┘
                    │
                    └──► [N5 (a, a)] ──► Carry  (-2a - 2a + 3 = -4a + 3)

```

## 2. Side-by-Side Comparison: `scratch` vs. `gomlx`

| Concept | Pure Go (`scratch/main.go`) | GoMLX (`gomlx/main.go`) |
| --- | --- | --- |
| **Execution Model** | Eager: calculates `float64` values immediately on each call | Symbolic: records `*graph.Node` operations into a static blueprint compiled via `graph.MustNewExec` |
| **Thresholding** | Go `if sum > 0` branch | `graph.Where(graph.GreaterThan(z, 0), 1, 0)` (reused in Video 5 for causal attention masks) |
| **Wiring Layers** | Passing return values between structs | Chaining `*graph.Node` outputs into downstream graph ops |

## 3. Running This Lab

```bash
# Run pure-Go implementation
go run ./00-foundations-labs/lab01-half-adder/scratch

# Run GoMLX implementation (pure-Go 'go' backend or OpenXLA 'xla' backend)
GOMLX_BACKEND=go go run ./00-foundations-labs/lab01-half-adder/gomlx
GOMLX_BACKEND=xla go run ./00-foundations-labs/lab01-half-adder/gomlx

```

### Expected Output

```text
x1 x2 | Sum Carry
-----------------
 0  0 |  0    0
 0  1 |  1    0
 1  0 |  1    0
 1  1 |  0    1

```

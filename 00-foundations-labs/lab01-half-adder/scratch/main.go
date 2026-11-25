package main

import "fmt"

type Perceptron struct {
	Weights []float64
	Bias    float64
}

func (p Perceptron) Forward(inputs []float64) float64 {
	sum := p.Bias
	for i := range inputs {
		sum += p.Weights[i] * inputs[i]
	}
	if sum > 0 {
		return 1.0
	}
	return 0.0
}

func NewNAND() Perceptron {
	return Perceptron{Weights: []float64{-2, -2}, Bias: 3}
}

// HalfAdder wires 5 NAND perceptrons together
func HalfAdder(x1, x2 float64) (sum, carry float64) {
	n1, n2, n3, n4, n5 := NewNAND(), NewNAND(), NewNAND(), NewNAND(), NewNAND()

	a := n1.Forward([]float64{x1, x2})
	b := n2.Forward([]float64{x1, a})
	c := n3.Forward([]float64{x2, a})

	sum = n4.Forward([]float64{b, c})
	// Feeding 'a' into both inputs computes (-2)*a + (-2)*a + 3 = (-4)*a + 3
	carry = n5.Forward([]float64{a, a})

	return sum, carry
}

func main() {
	inputs := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	fmt.Println("x1 x2 | sum carry")
	fmt.Println("-----------------")
	for _, in := range inputs {
		s, c := HalfAdder(in[0], in[1])
		fmt.Printf("%.0f %.0f |  %.0f    %.0f\n", in[0], in[1], s, c)
	}
}

package main

import (
	"fmt"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/core/graph"
)

func nandNode(x1, x2 *Node) *Node {
	g := x1.Graph()
	w1 := Const(g, float32(-2.0))
	w2 := Const(g, float32(-2.0))
	b := Const(g, float32(3.0))

	z := Add(Add(Mul(x1, w1), Mul(x2, w2)), b)
	// Heaviside step function: 1.0 if z > 0 else 0.0
	return Where(
		GreaterThan(z, ScalarZero(g, z.DType())),
		ScalarOne(g, z.DType()),
		ScalarZero(g, z.DType()),
	)
}

func main() {
	backend := compute.MustNew()
	defer backend.Finalize()

	halfAdderExec := MustNewExec(backend, func(x1, x2 *Node) (*Node, *Node) {
		a := nandNode(x1, x2)
		b := nandNode(x1, a)
		c := nandNode(x2, a)

		sum := nandNode(b, c)
		carry := nandNode(a, a)

		return sum, carry
	})

	inputs := [][2]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	fmt.Println("x1 x2 | Sum Carry (GoMLX)")
	fmt.Println("-------------------------")
	for _, in := range inputs {
		sumT, carryT := halfAdderExec.MustCall2(in[0], in[1])
		fmt.Printf(" %.0f  %.0f |  %.0f    %.0f\n",
			in[0], in[1], sumT.Value().(float32), carryT.Value().(float32))
	}
}

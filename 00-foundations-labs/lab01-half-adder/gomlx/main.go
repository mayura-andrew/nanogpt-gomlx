package main

import (
	"fmt"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
)

func nandNode(g *graph.Graph, x1, x2 *graph.Node) *graph.Node {
	w1 := graph.Const(g, float32(-2.0))
	w2 := graph.Const(g, float32(-2.0))
	b := graph.Const(g, float32(3.0))

	z := graph.Add(graph.Add(graph.Mul(x1, w1), graph.Mul(x2, w2)), b)
	// Heaviside step function: 1.0 if z > 0 else 0.0
	return graph.Where(
		graph.GreaterThan(z, graph.ScalarZero(g, z.DType())),
		graph.ScalarOne(g, z.DType()),
		graph.ScalarZero(g, z.DType()),
	)
}

func main() {
	backend := compute.MustNew()
	defer backend.Finalize()

	halfAdderExec := graph.MustNewExec(backend, func(x1, x2 *graph.Node) (*graph.Node, *graph.Node) {
		g := x1.Graph()
		a := nandNode(g, x1, x2)
		b := nandNode(g, x1, a)
		c := nandNode(g, x2, a)

		sum := nandNode(g, b, c)
		carry := nandNode(g, a, a)

		return sum, carry
	})

	inputs := [][2]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	fmt.Println("x1 x2 | Sum Carry (GoMLX)")
	fmt.Println("-------------------------")
	for _, in := range inputs {
		res := halfAdderExec.MustCall(tensors.FromValue(in[0]), tensors.FromValue(in[1]))
		fmt.Printf(" %.0f  %.0f |  %.0f    %.0f\n",
			in[0], in[1], res[0].Value().(float32), res[1].Value().(float32))
	}

}

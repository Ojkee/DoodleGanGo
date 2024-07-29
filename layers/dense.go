package layers

import "gonum.org/v1/gonum/mat"

type DenseLayer struct {
	weights mat.Dense
}

func (layer *DenseLayer) Forward(input mat.Vector) mat.Dense {
	return *mat.NewDense(1, 1, []float64{2})
}

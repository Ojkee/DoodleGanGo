package layers

import "gonum.org/v1/gonum/mat"

type DenseLayer struct {
	weights mat.Dense

	SavedDataVec
}

func (layer *DenseLayer) Forward(input mat.VecDense) mat.VecDense {
	layer.lastOutput.MulVec(&layer.weights, &input)
	return layer.lastOutput
}

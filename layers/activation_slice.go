package layers

import "math"

type Softmax struct {
	SavedData
}

func NewSoftmax() Softmax {
	return Softmax{}
}

func (layer *Softmax) Forward(input []float64) []float64 {
	expSums := 0.0
	for i := range input {
		expSums += math.Exp(input[i])
	}
	for i := range input {
		input[i] = math.Exp(input[i]) / expSums
	}
	return input
}

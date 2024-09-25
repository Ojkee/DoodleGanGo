package layers

import "gonum.org/v1/gonum/mat"

type Layer interface {
	Forward(input *mat.VecDense) *mat.VecDense
	Backward(inGrads *mat.VecDense) *mat.VecDense

	ApplyGrads(learningRate *float64, dWeightsGrads *mat.Dense, dBiasGrad *mat.VecDense)
	GetOutWeightsGrads() *mat.Dense
	GetOutBiasGrads() *mat.VecDense
}

type SavedDataVec struct {
	lastInput  mat.VecDense
	lastOutput mat.VecDense
}

type SavedGradsVec struct {
	lastInGrads  mat.VecDense
	lastOutGrads mat.VecDense
}

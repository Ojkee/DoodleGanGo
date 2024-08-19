package layers

import "gonum.org/v1/gonum/mat"

type Layer interface {
	Forward(input *mat.VecDense) *mat.VecDense
}

type SavedDataVec struct {
	lastInput  mat.VecDense
	lastOutput mat.VecDense
}

type SavedGradsVec struct {
	lastInGrads  mat.VecDense
	lastOutGrads mat.VecDense
}

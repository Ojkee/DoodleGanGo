package layers

import "gonum.org/v1/gonum/mat"

type Layer interface {
	Forward(input *mat.VecDense) *mat.VecDense
}

type SavedDataVec struct {
	lastInput     mat.VecDense
	lastOutput    mat.VecDense
	lastGradients mat.VecDense
}

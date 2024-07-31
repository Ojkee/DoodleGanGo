package layers

import "gonum.org/v1/gonum/mat"

type Layer interface {
	Forward(input interface{}) interface{}
}

type SavedDataMat struct {
	lastInput  []mat.Dense
	lastOutput []mat.Dense
}

type SavedDataVec struct {
	lastInput  mat.VecDense
	lastOutput mat.VecDense
}

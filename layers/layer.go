package layers

import "gonum.org/v1/gonum/mat"

type Layer interface {
	Forward(input interface{}) interface{}
}

type SavedMatData struct {
	lastInput  []mat.Dense
	lastOutput []mat.Dense
}

type SavedData struct {
	lastInput  []float64
	lastOutput []float64
}

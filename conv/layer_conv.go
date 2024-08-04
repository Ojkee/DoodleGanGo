package conv

import "gonum.org/v1/gonum/mat"

type ConvLayer interface {
	Forward(*[]mat.Dense) *[]mat.Dense
}

type SavedDataMat struct {
	lastInput  []mat.Dense
	lastOutput []mat.Dense
}

type MatSize struct {
	height int
	width  int
}

type Stride struct {
	horizontal int
	vertical   int
}

type Padding struct {
	up    int
	right int
	down  int
	left  int
}

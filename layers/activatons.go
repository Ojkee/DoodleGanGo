package layers

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type lambda func(i, j int, v float64) float64

type ReLU struct {
	SavedMatData
	lambda
}

type LeakyReLU struct {
	alpha float64
	SavedMatData
	lambda
}

type ELU struct {
	alpha float64
	SavedMatData
	lambda
}

type Sigmoid struct {
	SavedMatData
	lambda
}

type Tanh struct {
	SavedMatData
	lambda
}

func ApplyOnInput(
	act func(i, j int, v float64) float64,
	input []mat.Dense,
	dest *[]mat.Dense,
) {
	for i := range len(input) {
		(*dest)[i].Apply(
			act,
			&input[i],
		)
	}
}

func NewReLU() ReLU {
	act := func(i, j int, v float64) float64 {
		return max(0, v)
	}
	return ReLU{lambda: act}
}

func (layer *ReLU) Forward(input []mat.Dense) []mat.Dense {
	layer.lastInput = input
	layer.lastOutput = make([]mat.Dense, len(input))
	ApplyOnInput(layer.lambda, input, &layer.lastOutput)
	return layer.lastOutput
}

func NewLeakyReLU(alpha float64) LeakyReLU {
	if alpha < 0 || alpha > 1 {
		panic("Parameter alpha in LeakyReLU must be in range [0, 1]")
	}
	act := func(i, j int, v float64) float64 { return max(v*alpha, v) }
	return LeakyReLU{
		alpha:  alpha,
		lambda: act,
	}
}

func (layer *LeakyReLU) Forward(input []mat.Dense) []mat.Dense {
	layer.lastInput = input
	layer.lastOutput = make([]mat.Dense, len(input))
	ApplyOnInput(layer.lambda, input, &layer.lastOutput)
	return layer.lastOutput
}

func NewELU(alpha ...float64) ELU {
	param := 1.0
	if len(alpha) > 0 {
		param = alpha[0]
	}
	act := func(i, j int, v float64) float64 {
		if v >= 0 {
			return v
		}
		return param * (math.Exp(v) - 1)
	}
	return ELU{
		alpha:  param,
		lambda: act,
	}
}

func (layer *ELU) Forward(input []mat.Dense) []mat.Dense {
	layer.lastInput = input
	layer.lastOutput = make([]mat.Dense, len(input))
	ApplyOnInput(layer.lambda, input, &layer.lastOutput)
	return layer.lastOutput
}

func NewSigmoid() Sigmoid {
	act := func(i, j int, v float64) float64 {
		return 1 / (1 + math.Exp(-v))
	}
	return Sigmoid{
		lambda: act,
	}
}

func (layer *Sigmoid) Forward(input []mat.Dense) []mat.Dense {
	layer.lastInput = input
	layer.lastOutput = make([]mat.Dense, len(input))
	ApplyOnInput(layer.lambda, input, &layer.lastOutput)
	return layer.lastOutput
}

func NewTanh() Tanh {
	act := func(i, j int, v float64) float64 {
		return math.Tanh(v)
	}
	return Tanh{
		lambda: act,
	}
}

func (layer *Tanh) Forward(input []mat.Dense) []mat.Dense {
	layer.lastInput = input
	layer.lastOutput = make([]mat.Dense, len(input))
	ApplyOnInput(layer.lambda, input, &layer.lastOutput)
	return layer.lastOutput
}

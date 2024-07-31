package layers

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type Vlambda func(v float64) float64

type Softmax struct {
	SavedDataVec
}

type VReLU struct {
	SavedDataVec
	Vlambda
}

type VLeakyReLU struct {
	alpha float64
	SavedDataVec
	Vlambda
}

type VELU struct {
	alpha float64
	SavedDataVec
	Vlambda
}

type VSigmoid struct {
	SavedDataVec
	Vlambda
}

type VTanh struct {
	SavedDataVec
	Vlambda
}

func NewSoftmax() Softmax {
	return Softmax{}
}

func (layer *Softmax) Forward(input mat.VecDense) mat.VecDense {
	expSums := 0.0
	for i := range input.Len() {
		exp := math.Exp(input.At(i, 0))
		expSums += exp
		input.RawVector().Data[i] = exp
	}
	layer.lastOutput.ScaleVec(1/expSums, &input)
	return layer.lastOutput
}

func ApplyOnInputVec(
	act func(v float64) float64,
	input mat.VecDense,
	dest *mat.VecDense,
) {
	for i := range input.Len() {
		dest.RawVector().Data[i] = act(input.RawVector().Data[i])
	}
	dest = &input
}

func NewVReLU() VReLU {
	act := func(v float64) float64 {
		return max(0, v)
	}
	return VReLU{Vlambda: act}
}

func (layer *VReLU) Forward(input mat.VecDense) mat.VecDense {
	layer.lastInput = input
	layer.lastOutput = *mat.NewVecDense(input.Len(), nil)
	ApplyOnInputVec(layer.Vlambda, input, &layer.lastOutput)
	return layer.lastOutput
}

func NewVLeakyReLU(alpha float64) VLeakyReLU {
	if alpha < 0 || alpha > 1 {
		panic("Parameter alpha in LeakyReLU must be in range [0, 1]")
	}
	act := func(v float64) float64 { return max(v*alpha, v) }
	return VLeakyReLU{
		alpha:   alpha,
		Vlambda: act,
	}
}

func (layer *VLeakyReLU) Forward(input mat.VecDense) mat.VecDense {
	layer.lastInput = input
	layer.lastOutput = *mat.NewVecDense(input.Len(), nil)
	ApplyOnInputVec(layer.Vlambda, input, &layer.lastOutput)
	return layer.lastOutput
}

func NewVELU(alpha ...float64) VELU {
	param := 1.0
	if len(alpha) > 0 {
		param = alpha[0]
	}
	act := func(v float64) float64 {
		if v >= 0 {
			return v
		}
		return param * (math.Exp(v) - 1)
	}
	return VELU{
		alpha:   param,
		Vlambda: act,
	}
}

func (layer *VELU) Forward(input mat.VecDense) mat.VecDense {
	layer.lastInput = input
	layer.lastOutput = *mat.NewVecDense(input.Len(), nil)
	ApplyOnInputVec(layer.Vlambda, input, &layer.lastOutput)
	return layer.lastOutput
}

func NewVSigmoid() VSigmoid {
	act := func(v float64) float64 {
		return 1 / (1 + math.Exp(-v))
	}
	return VSigmoid{
		Vlambda: act,
	}
}

func (layer *VSigmoid) Forward(input mat.VecDense) mat.VecDense {
	layer.lastInput = input
	layer.lastOutput = *mat.NewVecDense(input.Len(), nil)
	ApplyOnInputVec(layer.Vlambda, input, &layer.lastOutput)
	return layer.lastOutput
}

func NewVTanh() VTanh {
	act := func(v float64) float64 {
		return math.Tanh(v)
	}
	return VTanh{
		Vlambda: act,
	}
}

func (layer *VTanh) Forward(input mat.VecDense) mat.VecDense {
	layer.lastInput = input
	layer.lastOutput = *mat.NewVecDense(input.Len(), nil)
	ApplyOnInputVec(layer.Vlambda, input, &layer.lastOutput)
	return layer.lastOutput
}

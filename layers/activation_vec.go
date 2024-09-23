package layers

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type (
	Vlambda      func(v float64) float64
	VlambdaPrime func(v float64) float64
)

type VActivationsLambdas struct {
	Vlambda
	VlambdaPrime
}

type Softmax struct {
	SavedDataVec
}

type VReLU struct {
	SavedDataVec
	VActivationsLambdas
}

type VLeakyReLU struct {
	alpha float64
	SavedDataVec
	VActivationsLambdas
}

type VELU struct {
	alpha float64
	SavedDataVec
	VActivationsLambdas
}

type VSigmoid struct {
	SavedDataVec
	VActivationsLambdas
}

type VTanh struct {
	SavedDataVec
	VActivationsLambdas
}

func ApplyOnInputVec(
	act func(v float64) float64,
	input, dest *mat.VecDense,
) {
	for i := range input.Len() {
		dest.RawVector().Data[i] = act(input.AtVec(i))
	}
	// dest = &input
}

func BackwardApply(
	actPrime func(v float64) float64,
	lastInput, inGrads *mat.VecDense,
) mat.VecDense {
	result := mat.NewVecDense(inGrads.Len(), nil)
	ApplyOnInputVec(actPrime, lastInput, result)
	result.MulElemVec(result, inGrads.SliceVec(0, inGrads.Len()))
	return *result
}

func NewSoftmax() Softmax {
	return Softmax{}
}

func (layer *Softmax) Forward(input *mat.VecDense) *mat.VecDense {
	expSums := 0.0
	for i := range input.Len() {
		exp := math.Exp(input.AtVec(i))
		expSums += exp
		input.RawVector().Data[i] = exp
	}
	layer.lastOutput.ScaleVec(1./expSums, input)
	return &layer.lastOutput
}

func NewVReLU() VReLU {
	act := func(v float64) float64 {
		return max(0, v)
	}
	actPrime := func(v float64) float64 {
		if v > 0 {
			return 1.0
		}
		return 0
	}
	return VReLU{
		VActivationsLambdas: VActivationsLambdas{
			Vlambda:      act,
			VlambdaPrime: actPrime,
		},
	}
}

func (layer *VReLU) Forward(input *mat.VecDense) *mat.VecDense {
	layer.lastInput = *input
	layer.lastOutput = *mat.NewVecDense(input.Len(), nil)
	ApplyOnInputVec(layer.Vlambda, input, &layer.lastOutput)
	return &layer.lastOutput
}

func (layer *VReLU) Backward(inGrads *mat.VecDense) *mat.VecDense {
	retVal := BackwardApply(layer.VlambdaPrime, &layer.lastInput, inGrads)
	return &retVal
}

func NewVLeakyReLU(alpha float64) VLeakyReLU {
	if alpha < 0 || alpha > 1 {
		panic("Parameter alpha in LeakyReLU must be in range [0, 1]")
	}
	act := func(v float64) float64 { return max(v*alpha, v) }
	actPrime := func(v float64) float64 {
		if v > 0.0 {
			return 1.0
		}
		return alpha
	}
	return VLeakyReLU{
		alpha: alpha,
		VActivationsLambdas: VActivationsLambdas{
			Vlambda:      act,
			VlambdaPrime: actPrime,
		},
	}
}

func (layer *VLeakyReLU) Forward(input *mat.VecDense) *mat.VecDense {
	layer.lastInput = *input
	layer.lastOutput = *mat.NewVecDense(input.Len(), nil)
	ApplyOnInputVec(layer.Vlambda, input, &layer.lastOutput)
	return &layer.lastOutput
}

func (layer *VLeakyReLU) Backward(inGrads *mat.VecDense) *mat.VecDense {
	retVal := BackwardApply(layer.VlambdaPrime, &layer.lastInput, inGrads)
	return &retVal
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
	actPrime := func(v float64) float64 {
		if v > 0.0 {
			return 1.0
		}
		return param * math.Exp(v)
	}
	return VELU{
		alpha: param,
		VActivationsLambdas: VActivationsLambdas{
			Vlambda:      act,
			VlambdaPrime: actPrime,
		},
	}
}

func (layer *VELU) Forward(input *mat.VecDense) *mat.VecDense {
	layer.lastInput = *input
	layer.lastOutput = *mat.NewVecDense(input.Len(), nil)
	ApplyOnInputVec(layer.Vlambda, input, &layer.lastOutput)
	return &layer.lastOutput
}

func (layer *VELU) Backward(inGrads *mat.VecDense) *mat.VecDense {
	retVal := BackwardApply(layer.VlambdaPrime, &layer.lastInput, inGrads)
	return &retVal
}

func NewVSigmoid() VSigmoid {
	act := func(v float64) float64 {
		return 1 / (1 + math.Exp(-v))
	}
	actPrime := func(v float64) float64 {
		return act(v) * (1.0 - act(v))
	}
	return VSigmoid{
		VActivationsLambdas: VActivationsLambdas{
			Vlambda:      act,
			VlambdaPrime: actPrime,
		},
	}
}

func (layer *VSigmoid) Forward(input *mat.VecDense) *mat.VecDense {
	layer.lastInput = *input
	layer.lastOutput = *mat.NewVecDense(input.Len(), nil)
	ApplyOnInputVec(layer.Vlambda, input, &layer.lastOutput)
	return &layer.lastOutput
}

func (layer *VSigmoid) Backward(inGrads *mat.VecDense) *mat.VecDense {
	retVal := BackwardApply(layer.VlambdaPrime, &layer.lastInput, inGrads)
	return &retVal
}

func NewVTanh() VTanh {
	act := func(v float64) float64 {
		return math.Tanh(v)
	}
	actPrime := func(v float64) float64 {
		return 1.0 - math.Pow(math.Tanh(v), 2)
	}
	return VTanh{
		VActivationsLambdas: VActivationsLambdas{
			Vlambda:      act,
			VlambdaPrime: actPrime,
		},
	}
}

func (layer *VTanh) Forward(input *mat.VecDense) *mat.VecDense {
	layer.lastInput = *input
	layer.lastOutput = *mat.NewVecDense(input.Len(), nil)
	ApplyOnInputVec(layer.Vlambda, input, &layer.lastOutput)
	return &layer.lastOutput
}

func (layer *VTanh) Backward(inGrads *mat.VecDense) *mat.VecDense {
	retVal := BackwardApply(layer.VlambdaPrime, &layer.lastInput, inGrads)
	return &retVal
}

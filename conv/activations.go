package conv

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type (
	lambda      func(i, j int, v float64) float64
	lambdaPrime func(i, j int, v float64) float64
)

type ActivationsLambdas struct {
	lambda
	lambdaPrime
}

type ReLU struct {
	SavedDataMat
	ActivationsLambdas
}

type LeakyReLU struct {
	alpha float64
	SavedDataMat
	ActivationsLambdas
}

type ELU struct {
	alpha float64
	SavedDataMat
	ActivationsLambdas
}

type Sigmoid struct {
	SavedDataMat
	ActivationsLambdas
}

type Tanh struct {
	SavedDataMat
	ActivationsLambdas
}

func ApplyOnInputMatDense(
	act func(i, j int, v float64) float64,
	source, dest *[]mat.Dense,
) {
	for i := range len(*source) {
		(*dest)[i].Apply(
			act,
			&(*source)[i],
		)
	}
}

func BackwardApply(
	actPrime func(i, j int, v float64) float64,
	lastInput []mat.Dense,
	inGrads *[]mat.Dense,
) []mat.Dense {
	result := make([]mat.Dense, len(lastInput))
	for i := range len(lastInput) {
		result[i].Apply(
			actPrime,
			&lastInput[i],
		)
		result[i].MulElem(&result[i], &(*inGrads)[i])
	}
	return result
}

func NewReLU() ReLU {
	act := func(i, j int, v float64) float64 {
		return max(0, v)
	}
	actPrime := func(i, j int, v float64) float64 {
		if v > 0 {
			return 1.0
		}
		return 0
	}
	return ReLU{
		ActivationsLambdas: ActivationsLambdas{
			lambda:      act,
			lambdaPrime: actPrime,
		},
	}
}

func (layer *ReLU) Forward(input *[]mat.Dense) *[]mat.Dense {
	layer.lastInput = *input
	layer.lastOutput = make([]mat.Dense, len(*input))
	ApplyOnInputMatDense(layer.lambda, input, &layer.lastOutput)
	return &layer.lastOutput
}

func (layer *ReLU) Backward(inGrads *[]mat.Dense) *[]mat.Dense {
	retVal := BackwardApply(layer.lambdaPrime, layer.lastInput, inGrads)
	return &retVal
}

func (layer *ReLU) DeflatOutGrads() *[]mat.Dense {
	return nil
}

func (layer *ReLU) GetBiasGrads() *[]float64 {
	return nil
}

func (layer *ReLU) ApplyGrads(
	learningRate *float64,
	dWeightsGrads *[]mat.Dense,
	dBiasGrad *[]float64,
) {
	return
}

func NewLeakyReLU(alpha float64) LeakyReLU {
	if alpha < 0 || alpha > 1 {
		panic("Parameter alpha in LeakyReLU must be in range [0, 1]")
	}
	act := func(i, j int, v float64) float64 { return max(v*alpha, v) }
	actPrime := func(i, j int, v float64) float64 {
		if v > 0.0 {
			return 1.0
		}
		return alpha
	}
	return LeakyReLU{
		alpha: alpha,
		ActivationsLambdas: ActivationsLambdas{
			lambda:      act,
			lambdaPrime: actPrime,
		},
	}
}

func (layer *LeakyReLU) Forward(input *[]mat.Dense) *[]mat.Dense {
	layer.lastInput = *input
	layer.lastOutput = make([]mat.Dense, len(*input))
	ApplyOnInputMatDense(layer.lambda, input, &layer.lastOutput)
	return &layer.lastOutput
}

func (layer *LeakyReLU) Backward(inGrads *[]mat.Dense) *[]mat.Dense {
	retVal := BackwardApply(layer.lambdaPrime, layer.lastInput, inGrads)
	return &retVal
}

func (layer *LeakyReLU) DeflatOutGrads() *[]mat.Dense {
	return nil
}

func (layer *LeakyReLU) GetBiasGrads() *[]float64 {
	return nil
}

func (layer *LeakyReLU) ApplyGrads(
	learningRate *float64,
	dWeightsGrads *[]mat.Dense,
	dBiasGrad *[]float64,
) {
	return
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
	actPrime := func(i, j int, v float64) float64 {
		if v > 0.0 {
			return 1.0
		}
		return param * math.Exp(v)
	}
	return ELU{
		alpha: param,
		ActivationsLambdas: ActivationsLambdas{
			lambda:      act,
			lambdaPrime: actPrime,
		},
	}
}

func (layer *ELU) Forward(input *[]mat.Dense) *[]mat.Dense {
	layer.lastInput = *input
	layer.lastOutput = make([]mat.Dense, len(*input))
	ApplyOnInputMatDense(layer.lambda, input, &layer.lastOutput)
	return &layer.lastOutput
}

func (layer *ELU) Backward(inGrads *[]mat.Dense) *[]mat.Dense {
	reVal := BackwardApply(layer.lambdaPrime, layer.lastInput, inGrads)
	return &reVal
}

func (layer *ELU) DeflatOutGrads() *[]mat.Dense {
	return nil
}

func (layer *ELU) GetBiasGrads() *[]float64 {
	return nil
}

func (layer *ELU) ApplyGrads(
	learningRate *float64,
	dWeightsGrads *[]mat.Dense,
	dBiasGrad *[]float64,
) {
	return
}

func NewSigmoid() Sigmoid {
	act := func(i, j int, v float64) float64 {
		return 1 / (1 + math.Exp(-v))
	}
	actPrime := func(i, j int, v float64) float64 {
		return act(i, j, v) * (1.0 - act(i, j, v))
	}
	return Sigmoid{
		ActivationsLambdas: ActivationsLambdas{
			lambda:      act,
			lambdaPrime: actPrime,
		},
	}
}

func (layer *Sigmoid) Forward(input *[]mat.Dense) *[]mat.Dense {
	layer.lastInput = *input
	layer.lastOutput = make([]mat.Dense, len(*input))
	ApplyOnInputMatDense(layer.lambda, input, &layer.lastOutput)
	return &layer.lastOutput
}

func (layer *Sigmoid) Backward(inGrads *[]mat.Dense) *[]mat.Dense {
	retVal := BackwardApply(layer.lambdaPrime, layer.lastInput, inGrads)
	return &retVal
}

func (layer *Sigmoid) DeflatOutGrads() *[]mat.Dense {
	return nil
}

func (layer *Sigmoid) GetBiasGrads() *[]float64 {
	return nil
}

func (layer *Sigmoid) ApplyGrads(
	learningRate *float64,
	dWeightsGrads *[]mat.Dense,
	dBiasGrad *[]float64,
) {
	return
}

func NewTanh() Tanh {
	act := func(i, j int, v float64) float64 {
		return math.Tanh(v)
	}
	actPrime := func(i, j int, v float64) float64 {
		return 1.0 - math.Pow(math.Tanh(v), 2)
	}
	return Tanh{
		ActivationsLambdas: ActivationsLambdas{
			lambda:      act,
			lambdaPrime: actPrime,
		},
	}
}

func (layer *Tanh) Forward(input *[]mat.Dense) *[]mat.Dense {
	layer.lastInput = *input
	layer.lastOutput = make([]mat.Dense, len(*input))
	ApplyOnInputMatDense(layer.lambda, input, &layer.lastOutput)
	return &layer.lastOutput
}

func (layer *Tanh) Backward(inGrads *[]mat.Dense) *[]mat.Dense {
	retVal := BackwardApply(layer.lambdaPrime, layer.lastInput, inGrads)
	return &retVal
}

func (layer *Tanh) DeflatOutGrads() *[]mat.Dense {
	return nil
}

func (layer *Tanh) GetBiasGrads() *[]float64 {
	return nil
}

func (layer *Tanh) ApplyGrads(
	learningRate *float64,
	dWeightsGrads *[]mat.Dense,
	dBiasGrad *[]float64,
) {
	return
}

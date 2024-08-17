package layers

import (
	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type DenseLayer struct {
	nInputs  int
	nNeurons int
	weights  mat.Dense

	bias mat.VecDense
	SavedDataVec
}

func NewDenseLayer(nInputs, nNeurons int) *DenseLayer {
	if nInputs < 0 || nNeurons < 0 {
		mess := fmt.Sprintf(
			"Number of inputs ans number of neurons must be positive,\n\thave: %d, %d",
			nInputs,
			nNeurons,
		)
		panic(mess)
	}
	bias := *mat.NewVecDense(nNeurons, nil)
	return &DenseLayer{
		nInputs:  nInputs,
		nNeurons: nNeurons,
		bias:     bias,
	}
}

func (layer *DenseLayer) InitFilterRandom(minRange, maxRange float64) {
	if maxRange < minRange {
		panic("minRange can't be greater than maxRange")
	}
	nWeights := layer.nInputs * layer.nNeurons
	randWeights := make([]float64, nWeights)
	for i := range nWeights {
		randWeights[i] = rand.Float64()*(maxRange-minRange) + minRange
	}
	layer.weights = *mat.NewDense(
		layer.nInputs,
		layer.nNeurons,
		randWeights,
	)
}

func (layer *DenseLayer) LoadWeights(source *[]float64) {
	if len(*source) != layer.nInputs*layer.nNeurons {
		mess := fmt.Sprintf(
			"Source length and dimentions doesn't match: %d * %d != %d",
			layer.nNeurons,
			layer.nInputs,
			len(*source),
		)
		panic(mess)
	}
	layer.weights = *mat.NewDense(
		layer.nNeurons,
		layer.nInputs,
		*source,
	)
}

func (layer *DenseLayer) LoadBias(bias float64) {
	biasSlice := make([]float64, layer.nNeurons)
	for i := range layer.nNeurons {
		biasSlice[i] = bias
	}
	layer.bias = *mat.NewVecDense(layer.nNeurons, biasSlice)
}

func (layer *DenseLayer) Forward(input mat.VecDense) *mat.VecDense {
	layer.lastInput = input
	layer.lastOutput.MulVec(&layer.weights, &input)
	layer.lastOutput.AddVec(&layer.lastOutput, &layer.bias)
	return &layer.lastOutput
}

func (layer *DenseLayer) Backward(nextGradients mat.VecDense) *mat.Dense {
	return nil
}

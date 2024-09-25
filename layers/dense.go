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
	bias     mat.VecDense

	SavedDataVec
	SavedGradsVec
	weightGrads mat.Dense
}

func NewDenseLayer(nInputs, nNeurons int) DenseLayer {
	if nInputs < 0 || nNeurons < 0 {
		mess := fmt.Sprintf(
			"Number of inputs ans number of neurons must be positive,\n\thave: %d, %d",
			nInputs,
			nNeurons,
		)
		panic(mess)
	}
	bias := *mat.NewVecDense(nNeurons, nil)
	return DenseLayer{
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

func (layer *DenseLayer) LoadBias(bias *[]float64) {
	if len(*bias) != layer.nNeurons {
		mess := fmt.Sprintf(
			"Bias length (%d) and number of neurons (%d) must be the same\n",
			len(*bias),
			layer.nNeurons,
		)
		panic(mess)
	}
	layer.bias = *mat.NewVecDense(layer.nNeurons, *bias)
}

func (layer *DenseLayer) Forward(input *mat.VecDense) *mat.VecDense {
	layer.lastInput = *input
	layer.lastOutput.MulVec(&layer.weights, input)
	layer.lastOutput.AddVec(&layer.lastOutput, &layer.bias)
	return &layer.lastOutput
}

func (layer *DenseLayer) Backward(inGrads *mat.VecDense) *mat.VecDense {
	layer.lastInGrads = *inGrads
	layer.weightGrads.Mul(inGrads, layer.lastInput.T())

	var result mat.VecDense
	result.MulVec(layer.weights.T(), inGrads)
	layer.lastOutGrads = result
	return &layer.lastOutGrads
}

func (layer *DenseLayer) GetOutWeightsGrads() *mat.Dense {
	return &layer.weightGrads
}

func (layer *DenseLayer) GetOutBiasGrads() *mat.VecDense {
	return &layer.lastInGrads
}

func (layer *DenseLayer) ApplyGrads(
	learningRate *float64,
	dWeightsGrads *mat.Dense,
	dBiasGrad *mat.VecDense,
) {
	var scaledWeightGrads mat.Dense
	scaledWeightGrads.Scale(*learningRate, &*dWeightsGrads)
	layer.weights.Sub(&layer.weights, &scaledWeightGrads)

	var scaledBiasGrads mat.VecDense
	scaledBiasGrads.ScaleVec(*learningRate, dBiasGrad)
	layer.bias.SubVec(&layer.bias, &scaledBiasGrads)
}

func (layer *DenseLayer) GetWeightsData() []float64 {
	return layer.weights.RawMatrix().Data
}

func (layer *DenseLayer) GetWeights() *mat.Dense {
	return &layer.weights
}

func (layer *DenseLayer) GetBiasData() []float64 {
	return layer.bias.RawVector().Data
}

func (layer *DenseLayer) GetBias() *mat.VecDense {
	return &layer.bias
}

package layers

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type Conv2D struct {
	kernelSize      [2]int
	numberOfFilters int
	inputSize       [2]int
	inputChannels   int
	padding         [4]int // N E S W // TODO
	stride          [2]int // TODO
	filters         []mat.Dense
	filtersRaw      []mat.Dense // TODO

	SavedMatData
}

func NewConv2D(kernelSize [2]int, numberOfFilters int, inputSize [2]int, inputChannels int) Conv2D {
	return Conv2D{
		kernelSize:      kernelSize,
		numberOfFilters: numberOfFilters,
		inputSize:       inputSize,
		inputChannels:   inputChannels,
		padding:         [4]int{0, 0, 0, 0},
		stride:          [2]int{1, 1},
	}
}

func (layer *Conv2D) NumChannels() int {
	return layer.numberOfFilters * layer.inputChannels
}

func (layer *Conv2D) NumPixels() int {
	return layer.kernelSize[0] * layer.kernelSize[1]
}

func (layer *Conv2D) InitFilterRandom(minRange, maxRange float64) {
	if maxRange < minRange {
		panic("minRange can't be greater than maxRange")
	}
	numChannels := layer.NumChannels()
	numPixels := layer.NumPixels()
	newFilter := make([]mat.Dense, numChannels)
	matValues := make([]float64, numPixels)
	for i := range numChannels {
		for j := range numPixels {
			ranVal := rand.Float64()*(maxRange-minRange) + minRange
			matValues[j] = ranVal
		}
		matValuesCopy := append([]float64{}, matValues...)
		newFilter[i] = layer.PrepareFilterToConv(
			mat.NewDense(layer.kernelSize[0], layer.kernelSize[1], matValuesCopy),
		)
	}
	layer.filters = newFilter
}

func (layer *Conv2D) LoadFilter(source []float64) {
	numChannels := layer.NumChannels()
	numPixels := layer.NumPixels()
	if len(source) != numChannels*numPixels {
		panic("Source length and dimentions doesn't match")
	}
	layer.filters = make([]mat.Dense, numChannels)
	for i := range numChannels {
		matValues := source[i*numPixels : i*numPixels+numPixels]
		layer.filters[i] = layer.PrepareFilterToConv(
			mat.NewDense(layer.kernelSize[0], layer.kernelSize[1], matValues),
		)
	}
}

func (layer *Conv2D) OutDims() (int, int) {
	outHeight := layer.inputSize[0] - layer.kernelSize[0] + 1
	outWidth := layer.inputSize[1] - layer.kernelSize[1] + 1
	return outHeight, outWidth
}

func (layer *Conv2D) PrepareFilterToConv(source *mat.Dense) mat.Dense {
	outHeight, outWidth := layer.OutDims()
	inputFlatDim := layer.inputSize[0] * layer.inputSize[1]
	convValues := make([]float64, outHeight*outWidth*inputFlatDim)

	rowOffset := 0
	for range outHeight {
		for j := range outWidth {
			for ki := range layer.kernelSize[0] {
				for kj := range layer.kernelSize[1] {
					c := rowOffset + j + kj + ki*layer.inputSize[1]
					convValues[c] = source.At(ki, kj)
				}
			}
			rowOffset += layer.inputSize[0] * layer.inputSize[1]
		}
		rowOffset += layer.inputSize[1]
	}
	return *mat.NewDense(outHeight*outWidth, inputFlatDim, convValues)
}

func (layer *Conv2D) PrintFilter(precision int) {
	PrintMatArray(&layer.filters, precision)
}

func (layer *Conv2D) ArrayToConv2DInput(source []float64) []mat.Dense {
	result := make([]mat.Dense, layer.inputChannels)
	numPixels := layer.NumPixels()
	s := 0
	for i := range layer.inputChannels {
		result[i] = *mat.NewDense(layer.inputSize[0], layer.inputSize[1], source[s:s+numPixels])
		s += numPixels
	}
	return result
}

func (layer *Conv2D) Forward(input []mat.Dense) []mat.Dense {
	outHeight, outWidth := layer.OutDims()
	inputFlatDim := layer.inputSize[0] * layer.inputSize[1]

	layer.lastInput = input
	layer.lastOutput = make([]mat.Dense, 0)

	for f := range layer.numberOfFilters {
		currentConvolved := *mat.NewDense(outHeight*outWidth, 1, nil)
		for i := range layer.inputChannels {
			flatInput := *mat.NewVecDense(inputFlatDim, FlattenMat(&input[i]))
			currFilter := &layer.filters[f*layer.inputChannels+i]
			var cm mat.VecDense
			cm.MulVec(currFilter, &flatInput)
			currentConvolved.Add(&currentConvolved, &cm)
		}
		layer.lastOutput = append(layer.lastOutput, currentConvolved)
	}

	return layer.lastOutput
}

func (layer *Conv2D) DeflatOutput() []mat.Dense {
	outHeight, outWidth := layer.OutDims()
	result := make([]mat.Dense, 0)
	for i := range layer.lastOutput {
		reshaped := mat.NewDense(outHeight, outWidth, layer.lastOutput[i].RawMatrix().Data)
		result = append(result, *reshaped)
	}
	return result
}

func (layer *Conv2D) FlatOutput() []float64 {
	var result []float64
	for i := range layer.numberOfFilters {
		result = append(result, FlattenMat(&layer.lastOutput[i])...)
	}
	return result
}

package conv

import (
	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/functools"
)

type Conv2D struct {
	ConvType
	kernelSize      MatSize
	numberOfFilters int
	inputChannels   int
	padding         Padding
	stride          Stride

	filters    []mat.Dense
	filtersRaw []mat.Dense // TODO
}

func NewConv2D(
	kernelSize [2]int,
	numberOfFilters int,
	inputSize [2]int,
	inputChannels int,
	stride [2]int,
	padding [4]int, // N, E, S, W
) Conv2D {
	outHeight := (inputSize[0]-kernelSize[0]+
		padding[0]+padding[2])/stride[0] + 1
	outWidth := (inputSize[1]-kernelSize[1]+
		padding[1]+padding[3])/stride[1] + 1
	return Conv2D{
		ConvType: ConvType{
			inputSize:  MatSize{inputSize[0], inputSize[1]},
			outputSize: MatSize{outHeight, outWidth},
		},
		kernelSize:      MatSize{kernelSize[0], kernelSize[1]},
		numberOfFilters: numberOfFilters,
		inputChannels:   inputChannels,
		padding:         Padding{padding[0], padding[1], padding[2], padding[3]},
		stride:          Stride{stride[0], stride[1]},
	}
}

func (layer *Conv2D) NumChannels() int {
	return layer.numberOfFilters * layer.inputChannels
}

func (layer *Conv2D) NumPixelsKernel() int {
	return layer.kernelSize.height * layer.kernelSize.width
}

func (layer *Conv2D) NumPixelsInput() int {
	return layer.inputSize.height * layer.inputSize.width
}

func (layer *Conv2D) GetOutputSize() (int, int) {
	return layer.outputSize.height, layer.outputSize.width
}

func (layer *Conv2D) GetFlatInputDim() int {
	return (layer.inputSize.height +
		layer.padding.up +
		layer.padding.down) *
		(layer.inputSize.width +
			layer.padding.left +
			layer.padding.right)
}

func (layer *Conv2D) InitFilterRandom(minRange, maxRange float64) {
	if maxRange < minRange {
		panic("minRange can't be greater than maxRange")
	}
	numChannels := layer.NumChannels()
	numPixelsKernel := layer.NumPixelsKernel()
	newFilter := make([]mat.Dense, numChannels)
	matValues := make([]float64, numPixelsKernel)
	for i := range numChannels {
		for j := range numPixelsKernel {
			ranVal := rand.Float64()*(maxRange-minRange) + minRange
			matValues[j] = ranVal
		}
		matValuesCopy := append([]float64{}, matValues...)
		newFilter[i] = layer.PrepareFilterToConv(
			mat.NewDense(layer.kernelSize.height, layer.kernelSize.width, matValuesCopy),
		)
	}
	layer.filters = newFilter
}

func (layer *Conv2D) LoadFilter(source *[]float64) {
	numChannels := layer.NumChannels()
	numPixelsKernel := layer.NumPixelsKernel()
	if len(*source) != numChannels*numPixelsKernel {
		mess := fmt.Sprintf(
			"Source length and dimentions doesn't match: %d * %d * %d * %d != %d",
			layer.numberOfFilters,
			layer.inputChannels,
			layer.kernelSize.height,
			layer.kernelSize.width,
			len(*source),
		)
		panic(mess)
	}
	layer.filters = make([]mat.Dense, numChannels)
	for i := range numChannels {
		matValues := (*source)[i*numPixelsKernel : i*numPixelsKernel+numPixelsKernel]
		layer.filters[i] = layer.PrepareFilterToConv(
			mat.NewDense(layer.kernelSize.height, layer.kernelSize.width, matValues),
		)
	}
}

func (layer *Conv2D) PrepareFilterToConv(source *mat.Dense) mat.Dense {
	inputFlatDim := layer.GetFlatInputDim()
	convValues := make([]float64, layer.outputSize.height*layer.outputSize.width*inputFlatDim)
	rowOffset := 0
	for range layer.outputSize.height {
		for j := range layer.outputSize.width {
			for ki := range layer.kernelSize.height {
				for kj := range layer.kernelSize.width {
					c := rowOffset + j*layer.stride.horizontal + kj + ki*layer.inputSize.width
					convValues[c] = source.At(ki, kj)
				}
			}
			rowOffset += layer.inputSize.height * layer.inputSize.width
		}
		rowOffset += layer.inputSize.width * layer.stride.vertical
	}
	return *mat.NewDense(layer.outputSize.height*layer.outputSize.width, inputFlatDim, convValues)
}

func (layer *Conv2D) PrintFilter(precision int) {
	functools.PrintMatArray(&layer.filters, precision)
}

func (layer *Conv2D) ArrayToConv2DInput(source []float64) []mat.Dense {
	result := make([]mat.Dense, layer.inputChannels)
	numPixelsInput := layer.NumPixelsInput()
	s := 0
	for i := range layer.inputChannels {
		result[i] = *mat.NewDense(layer.inputSize.height, layer.inputSize.width, source[s:s+numPixelsInput])
		s += numPixelsInput
	}
	return result
}

func (layer *Conv2D) Forward(input *[]mat.Dense) *[]mat.Dense {
	inputFlatDim := layer.inputSize.height * layer.inputSize.width

	layer.lastInput = *input
	layer.lastOutput = make([]mat.Dense, layer.numberOfFilters)

	for f := range layer.numberOfFilters {
		currentConvolved := *mat.NewDense(layer.outputSize.height*layer.outputSize.width, 1, nil)
		for i := range layer.inputChannels {
			flatInput := *mat.NewVecDense(inputFlatDim, functools.FlattenMat(&(*input)[i]))
			currFilter := &layer.filters[f*layer.inputChannels+i]
			var cm mat.VecDense
			cm.MulVec(currFilter, &flatInput)
			currentConvolved.Add(&currentConvolved, &cm)
		}
		layer.lastOutput[f] = currentConvolved
	}
	return &layer.lastOutput
}

func (layer *Conv2D) DeflatOutput() *[]mat.Dense {
	return GetDeflatOutput(&layer.ConvType)
}

func (layer *Conv2D) FlatOutput() *[]float64 {
	return GetFlatOutput(&layer.ConvType)
}

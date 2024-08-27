package conv

import (
	"fmt"
	"math/rand"
	"slices"

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

	filters []mat.Dense
	bias    []float64

	SavedGrads
	filterGrads []mat.Dense
	biasGrads   []float64
}

func NewConv2D(
	kernelSize [2]int,
	numberOfFilters int,
	inputSize [2]int,
	inputChannels int,
	stride [2]int,
	padding [4]int, // N, E, S, W
) Conv2D {
	if kernelSize[0] > inputSize[0] || kernelSize[1] > inputSize[1] {
		mess := fmt.Sprintf(
			"Kernel size (%d x %d) can't be greater than input size (%d x %d)",
			kernelSize[0], kernelSize[1],
			inputSize[0], inputSize[1],
		)
		panic(mess)
	}
	if numberOfFilters < 1 || inputChannels < 1 {
		mess := fmt.Sprintf(
			"Number of filters (%d) and number of channels (%d) must be positive",
			numberOfFilters, inputChannels,
		)
		panic(mess)
	}
	outputSize := MatSize{
		height: (inputSize[0]-kernelSize[0]+
			padding[0]+padding[2])/stride[0] + 1,
		width: (inputSize[1]-kernelSize[1]+
			padding[1]+padding[3])/stride[1] + 1,
	}
	return Conv2D{
		ConvType: ConvType{
			inputSize:  MatSize{inputSize[0], inputSize[1]},
			outputSize: outputSize,
		},
		kernelSize:      MatSize{kernelSize[0], kernelSize[1]},
		numberOfFilters: numberOfFilters,
		inputChannels:   inputChannels,
		padding: Padding{
			up:    padding[0],
			right: padding[1],
			down:  padding[2],
			left:  padding[3],
		},
		stride: Stride{
			vertical:   stride[0],
			horizontal: stride[1],
		},
		bias: make([]float64, numberOfFilters),
	}
}

func (layer *Conv2D) PrintFilter(precision int) {
	functools.PrintMatArray(&layer.filters, precision)
}

func MakeBiasMat(v float64, outSize MatSize) *mat.Dense {
	return mat.NewDense(
		outSize.FlatDim(),
		1,
		functools.RepeatSlice(v, outSize.FlatDim()),
	)
}

func MakeBias(biases []float64, outputSize MatSize, numberOfFilters int) []mat.Dense {
	if len(biases) != numberOfFilters {
		mess := fmt.Sprintf(
			"Number of biases (%d) doesn't mach number of filters (%d) ",
			len(biases),
			numberOfFilters,
		)
		panic(mess)
	}
	resultBias := make([]mat.Dense, numberOfFilters)
	for i := range numberOfFilters {
		resultBias[i] = *MakeBiasMat(biases[i], outputSize)
	}
	return resultBias
}

func (layer *Conv2D) LoadBias(biases *[]float64) {
	layer.bias = *biases
}

func (layer *Conv2D) NumChannels() int {
	return layer.numberOfFilters * layer.inputChannels
}

func GetPaddedInputSize(inputSize MatSize, padding Padding) MatSize {
	return MatSize{
		inputSize.height + padding.up + padding.down,
		inputSize.width + padding.left + padding.right,
	}
}

func (layer *Conv2D) GetFlatInputDim() int {
	horizontal := layer.inputSize.height +
		layer.padding.up +
		layer.padding.down
	vertical := layer.inputSize.width +
		layer.padding.left +
		layer.padding.right
	return horizontal * vertical
}

func (layer *Conv2D) InitFilterRandom(minRange, maxRange float64) {
	if maxRange < minRange {
		panic("minRange can't be greater than maxRange")
	}
	numChannels := layer.NumChannels()
	numPixelsKernel := layer.kernelSize.FlatDim()
	newFilter := make([]mat.Dense, numChannels)
	matValues := make([]float64, numPixelsKernel)

	for i := range numChannels {
		for j := range numPixelsKernel {
			ranVal := rand.Float64()*(maxRange-minRange) + minRange
			matValues[j] = ranVal
		}
		layer.filters[i] = *mat.NewDense(
			layer.kernelSize.height,
			layer.kernelSize.width,
			slices.Clone(matValues),
		)
	}
	layer.filters = newFilter
}

func (layer *Conv2D) LoadFilter(source *[]float64) {
	numChannels := layer.NumChannels()
	numPixelsKernel := layer.kernelSize.FlatDim()
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
		layer.filters[i] = *mat.NewDense(
			layer.kernelSize.height,
			layer.kernelSize.width,
			matValues,
		)
	}
}

func PrepareFilterToConv(
	source *mat.Dense,
	inputFlatDim int,
	paddedSize, outputSize, kernelSize MatSize,
	stride Stride,
) mat.Dense {
	convValues := make([]float64, outputSize.height*outputSize.width*inputFlatDim)
	rowOffset := 0
	for range outputSize.height {
		for j := range outputSize.width {
			for ki := range kernelSize.height {
				for kj := range kernelSize.width {
					c := rowOffset + j*stride.horizontal + kj + ki*paddedSize.width
					convValues[c] = source.At(ki, kj)
				}
			}
			rowOffset += paddedSize.height * paddedSize.width
		}
		rowOffset += paddedSize.width * stride.vertical
	}
	return *mat.NewDense(outputSize.height*outputSize.width, inputFlatDim, convValues)
}

func (layer *Conv2D) ArrayToConv2DInput(source []float64) []mat.Dense {
	result := make([]mat.Dense, layer.inputChannels)
	numPixelsInput := layer.inputSize.FlatDim()
	s := 0
	for i := range layer.inputChannels {
		result[i] = *mat.NewDense(
			layer.inputSize.height,
			layer.inputSize.width,
			source[s:s+numPixelsInput],
		)
		s += numPixelsInput
	}
	return result
}

func PreparedFlatInput(input *mat.Dense, inputSize MatSize, padding Padding) *mat.VecDense {
	paddedSize := GetPaddedInputSize(inputSize, padding)
	inputFlatDim := paddedSize.FlatDim()
	result := make([]float64, inputFlatDim)
	currentOffset := padding.up * GetPaddedInputSize(inputSize, padding).width
	for i := range inputSize.height {
		currentOffset += padding.left
		for j := range inputSize.width {
			result[currentOffset] = input.At(i, j)
			currentOffset += 1
		}
		currentOffset += padding.right
	}
	return mat.NewVecDense(inputFlatDim, result)
}

func Convolve(flatInput *mat.VecDense, kernel *mat.Dense) mat.VecDense {
	var cm mat.VecDense
	cm.MulVec(kernel, flatInput)
	return cm
}

func AddBias(dest *mat.Dense, b float64) {
	dest.Apply(
		func(i, j int, v float64) float64 {
			return v + b
		},
		dest,
	)
}

func (layer *Conv2D) Forward(input *[]mat.Dense) *[]mat.Dense {
	layer.lastInput = *input
	layer.lastOutput = make([]mat.Dense, layer.numberOfFilters)
	for f := range layer.numberOfFilters {
		currentConvolved := *mat.NewDense(layer.outputSize.FlatDim(), 1, nil)
		for i := range layer.inputChannels {
			kernelMat := PrepareFilterToConv(
				&layer.filters[f*layer.inputChannels+i],
				layer.GetFlatInputDim(),
				GetPaddedInputSize(layer.inputSize, layer.padding),
				layer.outputSize,
				layer.kernelSize,
				layer.stride,
			)
			cm := Convolve(
				PreparedFlatInput(&(*input)[i], layer.inputSize, layer.padding),
				&kernelMat,
			)
			currentConvolved.Add(&currentConvolved, &cm)
		}
		AddBias(&currentConvolved, layer.bias[f])
		layer.lastOutput[f] = currentConvolved
	}
	return &layer.lastOutput
}

func (layer *Conv2D) DeflatOutput() *[]mat.Dense {
	return Deflat(layer.lastOutput, layer.outputSize, layer.numberOfFilters)
}

func (layer *Conv2D) FlatOutput() *[]float64 {
	return GetFlatOutput(&layer.ConvType)
}

func RotateMatHalfPi(m mat.Dense) mat.Dense {
	rows, cols := m.Dims()
	rotatedRows := mat.NewDense(rows, cols, nil)
	rotatedRowsCols := mat.NewDense(rows, cols, nil)
	for i := range rows {
		rotatedRows.SetRow(rows-i-1, m.RawRowView(i))
	}
	for j := range cols {
		rawCol := mat.NewVecDense(rows, nil)
		rawCol.CopyVec(rotatedRows.ColView(j))
		rotatedRowsCols.SetCol(cols-j-1, rawCol.RawVector().Data)
	}
	return *rotatedRowsCols
}

func (layer *Conv2D) Backward(inGrads *[]mat.Dense) *[]mat.Dense {
	layer.lastInGrads = *inGrads

	layer.filterGrads = make([]mat.Dense, layer.NumChannels())
	layer.biasGrads = make([]float64, layer.numberOfFilters)
	for f := range layer.numberOfFilters {
		gradKernel := PrepareFilterToConv(
			&(*inGrads)[f],
			layer.GetFlatInputDim(),
			GetPaddedInputSize(layer.inputSize, layer.padding),
			layer.kernelSize,
			layer.outputSize,
			layer.stride,
		)
		for i := range layer.inputChannels {
			flatInput := PreparedFlatInput(
				&layer.lastInput[i],
				layer.inputSize,
				layer.padding,
			)
			cGrads := Convolve(flatInput, &gradKernel)
			layer.filterGrads[f*layer.inputChannels+i] = *mat.NewDense(
				layer.kernelSize.height,
				layer.kernelSize.width,
				cGrads.RawVector().Data,
			)
		}
		layer.biasGrads[f] = mat.Sum(&(*inGrads)[f])
	}

	layer.lastOutGrads = make([]mat.Dense, layer.inputChannels)
	for i := range layer.inputChannels {
		layer.lastOutGrads[i] = *mat.NewDense(layer.inputSize.FlatDim(), 1, nil)
	}

	for f := range layer.numberOfFilters {
		paddedGrads := PreparedFlatInput(&(*inGrads)[f], layer.outputSize, Padding{
			up:    layer.kernelSize.height - 1,
			right: layer.kernelSize.width - 1,
			down:  layer.kernelSize.height - 1,
			left:  layer.kernelSize.width - 1,
		})
		for i := range layer.inputChannels {
			rotatedKernel := RotateMatHalfPi(layer.filters[f*layer.inputChannels+i])
			pHeight, pWidth := (*inGrads)[f].Dims()
			pHeight += 2 * (layer.kernelSize.height - 1)
			pWidth += 2 * (layer.kernelSize.width - 1)
			inputGradKernel := PrepareFilterToConv(
				&rotatedKernel,
				paddedGrads.Len(),
				*NewMatSize(pHeight, pWidth),
				layer.inputSize,
				layer.kernelSize,
				layer.stride,
			)
			outGradsConvolved := Convolve(paddedGrads, &inputGradKernel)
			layer.lastOutGrads[i].Add(&layer.lastOutGrads[i], &outGradsConvolved)
		}
	}
	return &layer.lastOutGrads
}

func (layer *Conv2D) BackwardFromFlat(inGrads *[]float64) *[]mat.Dense {
	return nil
}

func (layer *Conv2D) DeflatOutGrads() *[]mat.Dense {
	return Deflat(layer.lastOutGrads, layer.inputSize, layer.inputChannels)
}

func (layer *Conv2D) GetFilterGrads() *[]mat.Dense {
	return &layer.filterGrads
}

func (layer *Conv2D) GetBiasGrads() *[]float64 {
	return &layer.biasGrads
}

func (layer *Conv2D) ApplyGrads(learningRate float64) {
	for f := range layer.NumChannels() {
		var scaledGrads mat.Dense
		scaledGrads.Scale(learningRate, &layer.filterGrads[f])
		gradsKernelMatSize := PrepareFilterToConv(
			&scaledGrads,
			layer.inputSize.FlatDim(),
			GetPaddedInputSize(layer.inputSize, layer.padding),
			layer.outputSize,
			layer.kernelSize,
			layer.stride,
		)
		layer.filters[f].Sub(
			&layer.filters[f],
			&gradsKernelMatSize,
		)
	}
	for b := range layer.numberOfFilters {
		layer.bias[b] -= learningRate * layer.biasGrads[b]
	}
}

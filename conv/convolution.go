package conv

import (
	"fmt"
	"math/rand"
	"slices"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/functools"
)

/*

   Sources for backpropagation:

   https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf
   https://hideyukiinada.github.io/cnn_backprop_strides2.html
   https://youtu.be/Pn7RK7tofPg?si=ICatCW-zOZKZk7au

*/

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

	convCache
}

type convCache struct {
	paddedInputSize       MatSize
	gradSize              MatSize
	dilatedGradSize       MatSize
	interweavedGradSize   MatSize
	paddedDilatedGradSize MatSize
	skippedRows           int
	skippedCols           int
}

func NewConv2D(
	kernelSize [2]int,
	numberOfFilters int,
	inputSize [2]int,
	inputChannels int,
	stride [2]int,
	padding [4]int, // N, E, S, W
) Conv2D {
	if kernelSize[0] > inputSize[0]+padding[0]+padding[2] ||
		kernelSize[1] > inputSize[1]+padding[1]+padding[3] {
		mess := fmt.Sprintf(
			"NewConv2D fail:\n\tKernel size (%d x %d) can't be greater than input size (%d x %d)",
			kernelSize[0], kernelSize[1],
			inputSize[0], inputSize[1],
		)
		panic(mess)
	}
	if numberOfFilters < 1 || inputChannels < 1 {
		mess := fmt.Sprintf(
			"NewConv2D fail:\n\tNumber of filters (%d) and number of channels (%d) must be positive",
			numberOfFilters,
			inputChannels,
		)
		panic(mess)
	}
	inputSize_ := MatSize{inputSize[0], inputSize[1]}
	outputSize_ := MatSize{
		height: (inputSize[0]-kernelSize[0]+
			padding[0]+padding[2])/stride[0] + 1,
		width: (inputSize[1]-kernelSize[1]+
			padding[1]+padding[3])/stride[1] + 1,
	}
	padding_ := Padding{
		up:    padding[0],
		right: padding[1],
		down:  padding[2],
		left:  padding[3],
	}
	stride_ := Stride{
		vertical:   stride[0],
		horizontal: stride[1],
	}

	skippedRows_ := skippedRowsColsPad(
		inputSize[0],
		kernelSize[0],
		stride[0],
		padding[0]+padding[2],
	)
	skippedCols_ := skippedRowsColsPad(
		inputSize[1],
		kernelSize[1],
		stride[1],
		padding[1]+padding[3],
	)
	dilatedGradSize_ := *dilatedSize(&outputSize_, &stride_)
	paddedDilatedGradSize_ := MatSize{
		height: dilatedGradSize_.height + 2*(kernelSize[0]-1) + skippedRows_,
		width:  dilatedGradSize_.width + 2*(kernelSize[1]-1) + skippedCols_,
	}

	interweavedGradSize_ := MatSize{
		height: dilatedGradSize_.height + skippedRows_,
		width:  dilatedGradSize_.width + skippedCols_,
	}
	return Conv2D{
		ConvType: ConvType{
			inputSize:  inputSize_,
			outputSize: outputSize_,
		},
		kernelSize:      MatSize{kernelSize[0], kernelSize[1]},
		numberOfFilters: numberOfFilters,
		inputChannels:   inputChannels,
		padding:         padding_,
		stride:          stride_,
		bias:            make([]float64, numberOfFilters),
		convCache: convCache{
			paddedInputSize:       getPaddedInputSize(inputSize_, padding_),
			gradSize:              outputSize_,
			dilatedGradSize:       dilatedGradSize_,
			interweavedGradSize:   interweavedGradSize_,
			paddedDilatedGradSize: paddedDilatedGradSize_,
			skippedRows:           skippedRows_,
			skippedCols:           skippedCols_,
		},
	}
}

func (layer *Conv2D) GetKernelSize() (int, int) {
	return layer.kernelSize.height, layer.kernelSize.width
}

func (layer *Conv2D) GetOutputSize() (int, int) {
	return layer.outputSize.height, layer.outputSize.width
}

func (layer *Conv2D) PrintFilter(precision int) {
	functools.PrintMatArray(&layer.filters, precision)
}

func (layer *Conv2D) LoadBias(biases *[]float64) {
	if len(*biases) != layer.numberOfFilters {
		mess := fmt.Sprintf(
			"Bias load fail:\n\tdimention of bias to load (%d) doesn't match number filters (%d)",
			len(*biases),
			layer.numberOfFilters,
		)
		panic(mess)
	}
	layer.bias = *biases
}

func (layer *Conv2D) InitFilterRandom(minRange, maxRange float64) {
	if maxRange < minRange {
		panic("Filter random initialization fail:\n\tminRange can't be greater than maxRange")
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
			"Load filter fail:\n\tSource length and dimentions doesn't match: %d * %d * %d * %d != %d",
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

func (layer *Conv2D) Forward(input *[]mat.Dense) *[]mat.Dense {
	layer.lastInput = *input
	layer.lastOutput = make([]mat.Dense, layer.numberOfFilters)
	for f := range layer.numberOfFilters {
		currentConvolved := *mat.NewDense(layer.outputSize.FlatDim(), 1, nil)
		for i := range layer.inputChannels {
			kernelMat := prepareFilterToConv(
				&layer.filters[f*layer.inputChannels+i],
				layer.paddedInputSize.FlatDim(),
				layer.paddedInputSize,
				layer.outputSize,
				layer.kernelSize,
				layer.stride,
			)
			cm := convolve(
				preparedFlatInput(&(*input)[i], layer.inputSize, layer.padding),
				&kernelMat,
			)
			currentConvolved.Add(&currentConvolved, &cm)
		}
		addBias(&currentConvolved, layer.bias[f])
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

func (layer *Conv2D) Backward(inGrads *[]mat.Dense) *[]mat.Dense {
	layer.lastInGrads = *inGrads
	layer.calcKernelBiasGrads(inGrads)
	layer.calcOutGrads(inGrads)
	return &layer.lastOutGrads
}

func (layer *Conv2D) BackwardFromFlat(inGrads *[]float64) *[]mat.Dense {
	inGradsMat := make([]mat.Dense, layer.numberOfFilters)
	pixelsInKernel := layer.kernelSize.FlatDim()
	for i := range layer.numberOfFilters {
		inGradsMat[i] = *mat.NewDense(
			layer.gradSize.height,
			layer.gradSize.width,
			(*inGrads)[i*pixelsInKernel:(i+1)*pixelsInKernel],
		)
	}
	return layer.Backward(&inGradsMat)
}

func (layer *Conv2D) BackwardFromVec(inGrads *mat.VecDense) *[]mat.Dense {
	inGradsData := inGrads.RawVector().Data
	return layer.BackwardFromFlat(&inGradsData)
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

func (layer *Conv2D) ApplyGrads(
	learningRate *float64,
	dWeightsGrads *[]mat.Dense,
	dBiasGrad *[]float64,
) {
	for b := range layer.numberOfFilters {
		layer.bias[b] -= *learningRate * (*dBiasGrad)[b]
	}
	for f := range layer.NumChannels() {
		var scaledGrads mat.Dense
		scaledGrads.Scale(*learningRate, &(*dWeightsGrads)[f])
		layer.filters[f].Sub(&layer.filters[f], &scaledGrads)
	}
}

func (layer *Conv2D) GetFilter() *[]mat.Dense {
	return &layer.filters
}

func (layer *Conv2D) GetBias() *[]float64 {
	return &layer.bias
}

func (layer *Conv2D) NumChannels() int {
	return layer.numberOfFilters * layer.inputChannels
}

func (layer *Conv2D) NumFilters() int {
	return layer.numberOfFilters
}

func getPaddedInputSize(inputSize MatSize, padding Padding) MatSize {
	return MatSize{
		inputSize.height + padding.up + padding.down,
		inputSize.width + padding.left + padding.right,
	}
}

func prepareFilterToConv(
	source *mat.Dense,
	paddedFlatInputDim int,
	paddedSize, outputSize, kernelSize MatSize,
	stride Stride,
) mat.Dense {
	convValues := make(
		[]float64,
		outputSize.FlatDim()*paddedFlatInputDim,
	)
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
	return *mat.NewDense(outputSize.FlatDim(), paddedFlatInputDim, convValues)
}

func preparedFlatInput(
	input *mat.Dense,
	inputSize MatSize,
	padding Padding,
) *mat.VecDense {
	paddedSize := getPaddedInputSize(inputSize, padding)
	inputFlatDim := paddedSize.FlatDim()
	result := make([]float64, inputFlatDim)
	currentOffset := padding.up * getPaddedInputSize(inputSize, padding).width
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

func convolve(flatInput *mat.VecDense, kernel *mat.Dense) mat.VecDense {
	var cm mat.VecDense
	cm.MulVec(kernel, flatInput)
	return cm
}

func addBias(dest *mat.Dense, b float64) {
	dest.Apply(
		func(i, j int, v float64) float64 {
			return v + b
		},
		dest,
	)
}

func rotateMatHalfPi(m mat.Dense) mat.Dense {
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

func addPadding(source *mat.Dense, padding Padding) *mat.Dense {
	n, m := source.Dims()
	paddedSize := getPaddedInputSize(MatSize{height: n, width: m}, padding)
	retData := make([]float64, paddedSize.FlatDim())
	offset := paddedSize.width * padding.up
	for i := range n {
		offset += padding.left
		for j := range m {
			retData[offset] = source.At(i, j)
			offset++
		}
		offset += padding.right
	}
	return mat.NewDense(paddedSize.height, paddedSize.width, retData)
}

func (layer *Conv2D) calcKernelBiasGrads(inGrads *[]mat.Dense) {
	layer.filterGrads = make([]mat.Dense, layer.NumChannels())
	layer.biasGrads = make([]float64, layer.numberOfFilters)

	for f := range layer.numberOfFilters {
		dilatedGrad := dilate(&(*inGrads)[f], &layer.dilatedGradSize, &layer.stride)
		paddedDilatedGrad := addPadding(dilatedGrad, Padding{
			up:    0,
			right: layer.skippedCols,
			down:  layer.skippedRows,
			left:  0,
		})
		kernelGrad := prepareFilterToConv(
			paddedDilatedGrad,
			layer.paddedInputSize.FlatDim(),
			layer.paddedInputSize,
			layer.kernelSize,
			layer.interweavedGradSize,
			Stride{
				vertical:   1,
				horizontal: 1,
			},
		)
		for i := range layer.inputChannels {
			flatInput := preparedFlatInput(
				&layer.lastInput[i],
				layer.inputSize,
				layer.padding,
			)
			convolvedKernelGrad := convolve(flatInput, &kernelGrad)
			layer.filterGrads[f*layer.inputChannels+i] = *mat.NewDense(
				layer.kernelSize.height,
				layer.kernelSize.width,
				convolvedKernelGrad.RawVector().Data,
			)
		}
		layer.biasGrads[f] = mat.Sum(&(*inGrads)[f])
	}
}

func dilatedSize(sourceSize *MatSize, stride *Stride) *MatSize {
	return &MatSize{
		height: stride.vertical*(sourceSize.height-1) + 1,
		width:  stride.horizontal*(sourceSize.width-1) + 1,
	}
}

func dilate(source *mat.Dense, dilatedSize *MatSize, stride *Stride) *mat.Dense {
	n, m := source.Dims()
	result := make([]float64, dilatedSize.FlatDim())
	pos := 0
	for i := range n {
		for j := range m {
			result[pos] = source.At(i, j)
			pos += stride.horizontal
		}
		if stride.vertical > 1 {
			pos -= 1
		}
		pos += dilatedSize.width * (stride.vertical - 1)
	}
	return mat.NewDense(dilatedSize.height, dilatedSize.width, result)
}

func cropMat(source mat.Dense, cropPadding Padding) mat.Dense {
	n, m := source.Dims()
	retHeight := n - cropPadding.up - cropPadding.down
	retWidth := m - cropPadding.left - cropPadding.right
	retVal := mat.NewDense(retHeight, retWidth, nil)
	retVal.Add(retVal, source.Slice(
		cropPadding.up,
		cropPadding.up+retHeight,
		cropPadding.left,
		cropPadding.left+retWidth,
	))
	return *retVal
}

func skippedRowsColsPad(inputSize, kernelSize, stride, sumPad int) int {
	retVal := inputSize + sumPad
	for retVal > kernelSize {
		retVal -= stride
	}
	if retVal == kernelSize {
		return 0
	}
	return retVal
}

func (layer *Conv2D) calcOutGrads(inGrads *[]mat.Dense) {
	layer.lastOutGrads = make([]mat.Dense, layer.inputChannels)
	for i := range layer.inputChannels {
		layer.lastOutGrads[i] = *mat.NewDense(layer.inputSize.height, layer.inputSize.width, nil)
	}

	for f := range layer.numberOfFilters {
		dilated := dilate(&(*inGrads)[f], &layer.dilatedGradSize, &layer.stride)
		paddedGradsFlat := preparedFlatInput(
			dilated,
			layer.dilatedGradSize,
			Padding{
				up:    layer.kernelSize.height - 1,
				right: layer.kernelSize.width - 1 + layer.skippedCols,
				down:  layer.kernelSize.height - 1 + layer.skippedRows,
				left:  layer.kernelSize.width - 1,
			},
		)
		for i := range layer.inputChannels {
			rotatedKernel := rotateMatHalfPi(layer.filters[f*layer.inputChannels+i])
			rotatedKernelPrep := prepareFilterToConv(
				&rotatedKernel,
				paddedGradsFlat.Len(),
				layer.paddedDilatedGradSize,
				layer.paddedInputSize,
				layer.kernelSize,
				Stride{horizontal: 1, vertical: 1},
			)
			outGradsConvolved := convolve(paddedGradsFlat, &rotatedKernelPrep)
			cropped := cropMat(
				*mat.NewDense(
					layer.paddedInputSize.height,
					layer.paddedInputSize.width,
					outGradsConvolved.RawVector().Data,
				),
				layer.padding,
			)
			layer.lastOutGrads[i].Add(&layer.lastOutGrads[i], &cropped)
		}
	}
}

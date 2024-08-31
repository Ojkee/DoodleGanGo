package conv

import "gonum.org/v1/gonum/mat"

type ConvLayer interface {
	Forward(*[]mat.Dense) *[]mat.Dense
}

type ConvType struct {
	SavedDataMat
	inputSize  MatSize
	outputSize MatSize
}

type SavedDataMat struct {
	lastInput  []mat.Dense
	lastOutput []mat.Dense
}

type MatSize struct {
	height int
	width  int
}

func NewMatSize(height, width int) *MatSize {
	return &MatSize{
		height: height,
		width:  width,
	}
}

func AddMatSizes(a *MatSize, b *MatSize) *MatSize {
	return NewMatSize(a.height+b.height, a.width+b.width)
}

type Stride struct {
	horizontal int
	vertical   int
}

type Padding struct {
	up    int
	right int
	down  int
	left  int
}

func NewPadding(up, right, down, left int) Padding {
	return Padding{
		up:    up,
		right: right,
		down:  down,
		left:  left,
	}
}

type SavedGrads struct {
	lastInGrads  []mat.Dense
	lastOutGrads []mat.Dense
}

func GetFlatOutput(layer *ConvType) *[]float64 {
	channelSize := layer.outputSize.height * layer.outputSize.width
	result := make([]float64, channelSize*len(layer.lastOutput))
	for i := range layer.lastOutput {
		for j := range channelSize {
			result[i*channelSize+j] = layer.lastOutput[i].RawMatrix().Data[j]
		}
	}
	return &result
}

func Deflat(source []mat.Dense, size MatSize, numFilter int) *[]mat.Dense {
	result := make([]mat.Dense, numFilter)
	for i := range numFilter {
		reshaped := mat.NewDense(
			size.height,
			size.width,
			source[i].RawMatrix().Data,
		)
		result[i] = *reshaped
	}
	return &result
}

func GetDeflatOutput(layer *ConvType) *[]mat.Dense {
	return Deflat(layer.lastOutput, layer.outputSize, len(layer.lastOutput))
}

func (size *MatSize) FlatDim() int {
	return size.height * size.width
}

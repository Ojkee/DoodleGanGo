package conv

import (
	"fmt"
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestPreparePaddingInput_1(t *testing.T) {
	prepared := preparedFlatInput(
		mat.NewDense(2, 2, []float64{
			1, 2,
			3, 4,
		}),
		*NewMatSize(2, 2),
		NewPadding(1, 1, 0, 0),
	)
	target := []float64{
		0, 0, 0,
		1, 2, 0,
		3, 4, 0,
	}
	if !reflect.DeepEqual(target, prepared.RawVector().Data) {
		t.Fail()
		fmt.Println(target)
		fmt.Println(prepared.RawVector().Data)
	}
}

func TestPreparePaddingInput_2(t *testing.T) {
	prepared := preparedFlatInput(
		mat.NewDense(2, 2, []float64{
			1, 2,
			3, 4,
		}),
		*NewMatSize(2, 2),
		NewPadding(1, 1, 1, 1),
	)
	target := []float64{
		0, 0, 0, 0,
		0, 1, 2, 0,
		0, 3, 4, 0,
		0, 0, 0, 0,
	}
	if !reflect.DeepEqual(target, prepared.RawVector().Data) {
		t.Fail()
		fmt.Println(target)
		fmt.Println(prepared.RawVector().Data)
	}
}

func TestPreparePaddingInput_3(t *testing.T) {
	prepared := preparedFlatInput(
		mat.NewDense(3, 3, []float64{
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
		}),
		*NewMatSize(3, 3),
		NewPadding(2, 2, 0, 1),
	)
	target := []float64{
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 1, 2, 3, 0, 0,
		0, 4, 5, 6, 0, 0,
		0, 7, 8, 9, 0, 0,
	}
	if !reflect.DeepEqual(target, prepared.RawVector().Data) {
		t.Fail()
		fmt.Println(target)
		fmt.Println(prepared.RawVector().Data)
	}
}

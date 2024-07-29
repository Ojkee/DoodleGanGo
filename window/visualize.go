package window

import (
	"log"

	rl "github.com/gen2brain/raylib-go/raylib"

	"DoodleGan/preprocess"
)

func RunVisualizeLoop() {
	dataSize := 100
	stride := 784
	var trainSetRatio float32 = 0.8

	trainSet, _, err := preprocess.GetSplitData(
		"Tornado.npy",
		dataSize,
		stride,
		trainSetRatio,
	)
	if err != nil {
		log.Fatal("Couldn't load data")
	}

	var SCALE int32 = 20
	var IMG_HEIGHT int32 = 28
	var IMG_WIDTH int32 = 28
	rl.InitWindow(IMG_HEIGHT*SCALE, IMG_WIDTH*SCALE, "DoodleGen")
	defer rl.CloseWindow()

	rl.SetTargetFPS(60)
	var imageIndex int
	for !rl.WindowShouldClose() {
		if rl.IsKeyPressed(rl.KeySpace) && imageIndex < len(trainSet)-1 {
			imageIndex += 1
		} else if rl.IsKeyPressed(rl.KeyBackspace) && imageIndex > 0 {
			imageIndex -= 1
		}

		rl.BeginDrawing()
		rl.ClearBackground(rl.RayWhite)
		DrawImage(trainSet[imageIndex], SCALE)
		rl.EndDrawing()
	}
}

func DrawImage(imageArray []uint8, SCALE int32) {
	for i := range 784 {
		c := rl.Color{
			R: 255 - imageArray[i],
			G: 255 - imageArray[i],
			B: 255 - imageArray[i],
			A: 255,
		}
		x := int32(i%28) * SCALE
		y := int32(i/28) * SCALE
		rl.DrawRectangle(x, y, SCALE, SCALE, c)
	}
}

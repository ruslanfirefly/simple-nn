package main

import (
	"ff"
	"fmt"
	"github.com/petar/GoMNIST"
)

func createFloatArr(b []byte) []float64 {
	var res []float64

	for _, v := range b {
		res = append(res, float64(v))
	}
	return res
}
func createAnsArr(l GoMNIST.Label) []float64 {
	res := []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	res[int(l)] = float64(1)
	return res
}

func main() {
	var (
		patterns  [][][]float64
		patterns1 [][][]float64
		ff        ff.FeedForward
	)
	fmt.Println("hello world")
	train, test, err := GoMNIST.Load("./src/github.com/petar/GoMNIST/data")
	if err != nil {
	}
	fmt.Println(train.Count())
	fmt.Println(test.Count())
	// get train patters
	for i := 0; i < 600; i++ {
		image, label := train.Get(i)
		patterns = append(patterns, [][]float64{createFloatArr(image), createAnsArr(label)})
	}
	ff.Init(784, 10, []int{350, 150, 70, 30 ,15}, 0.5)
	ff.Train(patterns, 1000)

	for i := 0; i < 10000; i++ {
		image1, label1 := test.Get(i)
		patterns1 = append(patterns1, [][]float64{createFloatArr(image1), createAnsArr(label1)})
	}
	ff.Test(patterns1)
}
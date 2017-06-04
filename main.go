package main

import (
	"ff"
	"fmt"
	"github.com/petar/GoMNIST"
	"multiNN"
)

func createFloatArr(b []byte) []float64 {
	var res []float64

	for _, v := range b {
		res = append(res, float64(float64(v)/1000))
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
		train_patterns [][][]float64
		test_patterns  [][][]float64
		Mnn            multiNN.MultiNN
	)
	digit := int(5)
	fmt.Println("Start programm for recognition bumbers")
	train, test, err := GoMNIST.Load("./src/github.com/petar/GoMNIST/data")
	if err != nil {
	}
	fmt.Println("Total trail collection: ", train.Count())
	fmt.Println("Total test collection: ", test.Count())
	//get train patters
	for i := 0; i < train.Count(); i++ {
		image, label := train.Get(i)
		if int(label) == digit {
			train_patterns = append(train_patterns, [][]float64{createFloatArr(image), createAnsArr(label)})
		}
	}
	fmt.Println(len(train_patterns))
	for i := 0; i < test.Count(); i++ {
		image1, label1 := test.Get(i)
		if int(label1) == digit {
			test_patterns = append(test_patterns, [][]float64{createFloatArr(image1), createAnsArr(label1)})
		}
	}
	fmt.Printf("Total trail collection for digit %d:  %d \n", digit, len(train_patterns))
	fmt.Printf("Total trail collection for digit %d:  %d \n", digit, len(test_patterns))

	var ff, ff2 ff.FeedForward
	ff.Init(784, 50, 30)
	Mnn.NN = append(Mnn.NN, &ff)

	ff2.Init(30, 15, 10)
	Mnn.NN = append(Mnn.NN, &ff2)

	fmt.Println("Start train")
	Mnn.Train(train_patterns, 500, 0.9, 0.1, true)
	Mnn.Test(test_patterns)
}

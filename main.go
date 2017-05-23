package main

import (
	"ff"
	"fmt"
	"github.com/petar/GoMNIST"
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

type MultiNN struct {
	nn []*ff.FeedForward
}

func (ff *MultiNN) Update(patern []float64) []float64 {
	res := patern
	for _, v := range ff.nn {
		res = v.Predict(res)
	}
	return res
}

func (ff *MultiNN) BackProp(target []float64, lRate, mFactor float64) float64 {
	t := target
	sum := float64(0)
	for i := len(ff.nn) - 1; i >= 0; i-- {
		err, deltas := ff.nn[i].BackPropagate(t, lRate, mFactor)
		if i > 0 {
			t = nil
			for k, v := range deltas {
				t = append(t, v+ff.nn[i-1].OutputActivations[k])
			}
		}
		if i == len(ff.nn)-1 {
			sum += err
		} else {
			sum += 0
		}
	}
	//e,_:=ff.nn[len(ff.nn)-1].BackPropagate(t, lRate, mFactor)
	//sum += e
	return sum
}

func (nn *MultiNN) Train(patterns [][][]float64, iterations int, lRate, mFactor float64, debug bool) []float64 {
	errors := make([]float64, iterations)

	for i := 0; i < iterations; i++ {
		var e float64
		for _, p := range patterns {
			nn.Update(p[0])

			tmp := nn.BackProp(p[1], lRate, mFactor)
			e += tmp
		}

		errors[i] = e

		if debug && i%10 == 0 {
			fmt.Println(i, e)
		}
	}

	return errors
}

func (nn *MultiNN) Test(patterns [][][]float64) {
	correct := 0
	err := 0
	for _, p := range patterns {
		key := 0
		res := float64(0)
		key1 := 0
		res1 := float64(0)
		for k, v := range nn.Update(p[0]) {
			if v > float64(res) {
				res = v
				key = k
			}
		}
		for k, v := range p[1] {
			if v >= float64(res1) {
				res1 = v
				key1 = k
			}
		}
		if key == key1 {
			correct++
		} else {
			err++
		}

	}
	fmt.Println("correct: ", correct, "error: ", err)
}

func main() {
	var (
		patterns  [][][]float64
		patterns1 [][][]float64
		Mnn       MultiNN
	)
	fmt.Println("hello world")
	train, test, err := GoMNIST.Load("./src/github.com/petar/GoMNIST/data")
	if err != nil {
	}
	fmt.Println(train.Count())
	fmt.Println(test.Count())
	//get train patters
	for i := 0; i < 600; i++ {
		image, label := train.Get(i)
		patterns = append(patterns, [][]float64{createFloatArr(image), createAnsArr(label)})
	}
	//patterns = [][][]float64{
	//	{{0, 0, 0, 0, 0}, {0, 1}},
	//	{{0, 0, 0, 0, 1}, {1, 0}},
	//	{{0, 1, 0, 0, 1}, {0, 1}},
	//	{{1, 1, 0, 0, 1}, {0, 1}},
	//	{{1, 1, 0, 0, 0}, {0, 1}},
	//	{{1, 0, 0, 0, 0}, {1, 0}},
	//	{{1, 0, 0, 0, 1}, {1, 0}},
	//	{{0, 1, 0, 0, 0}, {0, 1}},
	//}
	//ff.Init(784, 10, []int{10, 5}, 0.5)
	//ff.Init(5, 2, []int{3, 2}, 0.05)
	//fmt.Println(ff.Weights)
	//ff.Train(patterns, 100, false)
	//fmt.Println(ff.Weights)
	for i := 0; i < 10000; i++ {
		image1, label1 := test.Get(i)
		patterns1 = append(patterns1, [][]float64{createFloatArr(image1), createAnsArr(label1)})
	}
	var ff, ff2 ff.FeedForward
	ff.Init(784, 50, 30)
	Mnn.nn = append(Mnn.nn, &ff)

	ff2.Init(30, 15, 10)
	Mnn.nn = append(Mnn.nn, &ff2)

	fmt.Println("Start train")
	Mnn.Train(patterns, 500, 0.9, 0.1, true)
	Mnn.Test(patterns1)
}

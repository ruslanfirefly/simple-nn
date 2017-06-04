package main

import (
	"fmt"
	"github.com/MaxHalford/gago"
	"github.com/petar/GoMNIST"
	"math"
	"math/rand"
	"multiNN"
	"time"
)

var (
	Mnn            multiNN.MultiNN
	train_patterns [][][]float64
	test_patterns  [][][]float64
	index          int
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

type Vector []float64

func (X Vector) Evaluate() float64 {
	var Mnn1 multiNN.MultiNN
	Mnn1.Init([][]int{{784, 50, 10}})
	Mnn1.AllWeightsNetwork = []float64{}
	Mnn1.AllWeightsNetwork = append(Mnn1.AllWeightsNetwork, []float64(X)...)
	Mnn1.SetAllWeightsNetwork()
	result := Mnn1.Update(train_patterns[index][0])
	e := float64(0)
	for i := 0; i < len(train_patterns[index][1]); i++ {
		e += 0.5 * math.Pow(train_patterns[index][1][i]-result[i], 2)
	}
	return e
}

func (X Vector) Mutate(rng *rand.Rand) {
	gago.MutNormalFloat64(X, 0.8, rng)
}

func (X Vector) Crossover(Y gago.Genome, rng *rand.Rand) (gago.Genome, gago.Genome) {
	var o1, o2 = gago.CrossGNXFloat64(X, Y.(Vector), 2, rng) // Returns two float64 slices
	return Vector(o1), Vector(o2)
}

func MakeVector(rng *rand.Rand) gago.Genome {
	return Vector(gago.InitUnifFloat64(len(Mnn.AllWeightsNetwork), 0, 1, rng))

}

func init() {
	Mnn.Init([][]int{{784, 50, 10}})
	Mnn.GetAllWeightsNetwork()
}

func main() {
	backPropagation := false
	digit := int(5)
	fmt.Println("Start programm for recognition numbers. For number: ", digit)
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
		if len(train_patterns) == 2 {
			break
		}
	}
	for i := 0; i < test.Count(); i++ {
		image1, label1 := test.Get(i)
		if int(label1) == digit {
			test_patterns = append(test_patterns, [][]float64{createFloatArr(image1), createAnsArr(label1)})
		}
	}

	fmt.Printf("Total trail collection for digit %d:  %d \n", digit, len(train_patterns))
	fmt.Printf("Total test collection for digit %d:  %d \n", digit, len(test_patterns))

	if backPropagation {
		fmt.Println("Start train")
		startTime := time.Now()
		Mnn.Train(train_patterns, 11, 0.9, 0.1, true)
		endTime := time.Now().Sub(startTime)
		fmt.Printf("Time for education %f \n", endTime.Seconds())
		Mnn.Test(test_patterns)
	} else {
		fmt.Println("test")
		var ga = gago.Generational(MakeVector)
		ga.NPops = 1
		ga.Initialize()
		fmt.Printf("Population: %d \n", ga.PopSize)
		fmt.Printf("Best fitness at generation 0: %e\n", ga.Best.Fitness)
		for i := 0; i < 12; i++ {
			for index = 0; index < len(train_patterns); index++ {
				ga.Enhance()
			}
			fmt.Printf("Best fitness at generation %d: %e\n", i, ga.Best.Fitness)
		}
		fmt.Printf("Best fitness at final generation : %e\n", ga.Best.Fitness)
		Mnn.AllWeightsNetwork = []float64{}
		Mnn.AllWeightsNetwork = append(Mnn.AllWeightsNetwork, []float64(ga.Best.Genome.(Vector))...)
		Mnn.SetAllWeightsNetwork()
		Mnn.Test(test_patterns)
	}

}

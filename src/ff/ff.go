package ff

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
)

type FeedForward struct {
	InputNodes   int
	OutputNodes  int
	LayerNumbers int
	Layers       []int
	Weights      []float64
	wMatrix      []mat64.Dense
	LRate        float64
}

func applySigm(i, j int, v float64) float64 {
	return activateFunction(v)
}

func (ff *FeedForward) Init(inputNodes, OutputNodes int, layers []int, lRate float64) {
	ff.InputNodes = inputNodes + 1 //+ bias
	ff.OutputNodes = OutputNodes
	fl := ff.InputNodes
	ff.Layers = layers
	ff.LRate = lRate
	sum := 0
	for _, v := range ff.Layers {
		sum = sum + (fl * v)
		fl = v
	}
	sum = sum + (fl * ff.OutputNodes)
	ff.Weights = make([]float64, sum)
	for i := 0; i < len(ff.Weights); i++ {
		ff.Weights[i] = random(-1, 1)
	}
}
func (ff *FeedForward) Predict(pattern []float64) []float64 {
	var result_itog mat64.Dense
	updatedPattern := append(pattern, 1)
	initMat := mat64.NewDense(1, len(updatedPattern), updatedPattern)
	fl := ff.InputNodes
	split := 0
	for _, v := range ff.Layers {
		var result mat64.Dense
		secondMatrix := mat64.NewDense(fl, v, ff.Weights[split:(split+(fl*v))])
		result.Mul(initMat, secondMatrix)
		result.Apply(applySigm, &result)
		*initMat = result
		fl = v
		split += fl * v
	}
	secondMatrix := mat64.NewDense(fl, ff.OutputNodes, ff.Weights[split:(split+(fl*ff.OutputNodes))])
	result_itog.Mul(initMat, secondMatrix)
	result_itog.Apply(applySigm, &result_itog)
	return result_itog.RawRowView(0)
}

func (ff *FeedForward) predict_for_train(pattern []float64) {
	var result_itog mat64.Dense
	updatedPattern := append(pattern, 1)
	initMat := mat64.NewDense(1, len(updatedPattern), updatedPattern)
	ff.wMatrix = append(ff.wMatrix, *initMat)
	fl := ff.InputNodes
	split := 0
	for _, v := range ff.Layers {
		var result mat64.Dense
		secondMatrix := mat64.NewDense(fl, v, ff.Weights[split:(split+(fl*v))])
		result.Mul(initMat, secondMatrix)
		result.Apply(applySigm, &result)
		*initMat = result
		fl = v
		split += fl * v
		ff.wMatrix = append(ff.wMatrix, *secondMatrix)
		ff.wMatrix = append(ff.wMatrix, result)
	}
	secondMatrix := mat64.NewDense(fl, ff.OutputNodes, ff.Weights[split:(split+(fl*ff.OutputNodes))])
	result_itog.Mul(initMat, secondMatrix)
	result_itog.Apply(applySigm, &result_itog)

	ff.wMatrix = append(ff.wMatrix, *secondMatrix)
	ff.wMatrix = append(ff.wMatrix, result_itog)
}

func (ff *FeedForward) backPropagation(targets []float64) {
	var delta_w []float64
	for cnt := len(ff.wMatrix) - 2; cnt >= 1; cnt -= 2 {
		var err_lay []float64
		res := ff.wMatrix[cnt+1].RawRowView(0)
		for k := 0; k < len(res); k++ {
			if len(ff.wMatrix)-2 == cnt {
				err_lay = append(err_lay, res[k]-targets[k])
			} else {
				err_lay = append(err_lay, res[k]*delta_w[k])
			}
		}
		delta_w = nil
		ff.wMatrix[cnt].Apply(
			func(i, j int, v float64) float64 {
				prev_res := ff.wMatrix[cnt-1].RawRowView(0)
				gradient_lay := dActivateFunction(res[j])
				delta_w = append(delta_w, err_lay[j]*gradient_lay)
				return v - (prev_res[j] * delta_w[j] * ff.LRate)
			}, &ff.wMatrix[cnt])
	}
	var newW []float64
	for i := 1; i < len(ff.wMatrix); i += 2 {
		r, _ := ff.wMatrix[i].Caps()
		for k := 0; k < r; k++ {
			row := ff.wMatrix[i].RawRowView(k)
			newW = append(newW, row...)
		}
	}
	ff.Weights = newW
}

func (ff *FeedForward) Train(patterns [][][]float64, epohe int) {
	for i := 0; i < epohe; i++ {
		for _, v := range patterns {
			ff.predict_for_train(v[0])
			ff.backPropagation(v[1])
			ff.wMatrix = nil
		}
	}

}

func (nn *FeedForward) Test(patterns [][][]float64) {
	correct := 0
	err := 0
	for _, p := range patterns {
		key := 0
		res := float64(0)
		key1 := 0
		res1 := float64(0)
		for k, v := range nn.Predict(p[0]) {
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

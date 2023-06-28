use super::matrix::Matrix;
use rand::prelude::*;

#[derive(Clone, Debug)]
pub struct NeuralNetwork {
    size: usize,
    weight: Vec<Matrix>,
    bias: Vec<Matrix>,
    activation: Vec<Matrix>,
}

impl NeuralNetwork {
    pub fn new(data: &[usize]) -> Self {
        let size = data.len();
        assert!(size > 0);

        let mut nn = NeuralNetwork {
            size: size - 1,
            weight: Vec::with_capacity(size),
            bias: Vec::with_capacity(size),
            activation: Vec::with_capacity(size),
        };

        nn.activation.push(Matrix::new(1, data[0]));
        for i in 1..size {
            nn.weight.push(Matrix::new(data[i - 1], data[i]));
            nn.bias.push(Matrix::new(1, data[i]));
            nn.activation.push(Matrix::new(1, data[i]));
        }

        nn
    }

    pub fn new_random(data: &[usize]) -> Self {
        let size = data.len();
        // let mut rng = StdRng::seed_from_u64(1);
        let mut rng = rand::thread_rng();

        assert!(size > 0);

        let mut nn = NeuralNetwork {
            size: size - 1,
            weight: Vec::with_capacity(size),
            bias: Vec::with_capacity(size),
            activation: Vec::with_capacity(size),
        };

        nn.activation.push(Matrix::from_iter(
            1,
            data[0],
            std::iter::repeat_with(|| rng.gen_range(0.0..1.0)),
        ));

        for i in 1..size {
            nn.weight.push(Matrix::from_iter(
                data[i - 1],
                data[i],
                std::iter::repeat_with(|| rng.gen_range(0.0..1.0)),
            ));
            nn.bias.push(Matrix::from_iter(
                1,
                data[i],
                std::iter::repeat_with(|| rng.gen_range(0.0..1.0)),
            ));
            nn.activation.push(Matrix::from_iter(
                1,
                data[i],
                std::iter::repeat_with(|| rng.gen_range(0.0..1.0)),
            ));
        }

        nn
    }

    pub fn forward(&mut self) {
        for i in 0..self.size {
            self.activation[i + 1] =
                (&(&self.activation[i] * &self.weight[i]) + &self.bias[i]).sigmoid();
        }
    }

    pub fn cost(&mut self, input: &Matrix, output: &Matrix) -> f64 {
        assert!(input.rows() == output.rows());
        assert!(output.cols() == self.activation[self.size].cols());

        let mut result = 0.0;
        let n = input.rows();

        for i in 0..input.rows() {
            let x = input.get_row_matrix(i).unwrap();
            let y = output.get_row_matrix(i).unwrap();

            self.activation[0] = x;
            self.forward();

            for j in 0..output.cols() {
                let d = self.activation[self.size].get(0, j).unwrap() - y.get(0, j).unwrap();
                result += d * d;
            }
        }
        result / n as f64
    }

    pub fn finite_diff(&mut self, gradient: &mut Self, eps: &f64, input: &Matrix, output: &Matrix) {
        let c = self.cost(input, output);

        for i in 0..self.weight.len() {
            for row in 0..self.weight[i].rows() {
                for col in 0..self.weight[i].cols() {
                    let temp = *self.weight[i].get(row, col).unwrap();
                    self.weight[i].set(row, col, temp + eps);
                    gradient.weight[i].set(row, col, (self.cost(input, output) - c) / eps);
                    self.weight[i].set(row, col, temp);
                }
            }

            for row in 0..self.bias[i].rows() {
                for col in 0..self.bias[i].cols() {
                    let temp = *self.bias[i].get(row, col).unwrap();
                    self.bias[i].set(row, col, temp + eps);
                    gradient.bias[i].set(row, col, (self.cost(input, output) - c) / eps);
                    self.bias[i].set(row, col, temp);
                }
            }
        }
    }

    pub fn learn(&mut self, gradient: &mut Self, rate: &f64) {
        for i in 0..self.size {
            for row in 0..self.weight[i].rows() {
                for col in 0..self.weight[i].cols() {
                    let mut temp = *self.weight[i].get(row, col).unwrap();
                    temp -= rate * gradient.weight[i].get(row, col).unwrap();
                    self.weight[i].set(row, col, temp);
                }
            }
        }
    }

    pub fn test(&mut self, input: &Matrix) -> Matrix {
        self.activation[0] = input.clone();
        self.forward();
        self.activation[self.size].clone()
    }
}

impl std::fmt::Display for NeuralNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "size: {}", self.size)?;
        for i in 0..self.size {
            writeln!(f, "weight[{}]:\n{}", i, self.weight[i])?;
            writeln!(f, "bias[{}]:\n{}", i, self.bias[i])?;
            writeln!(f, "activation[{}]:\n{}", i, self.activation[i])?;
        }
        writeln!(
            f,
            "activation[{}]:\n{}",
            self.size, self.activation[self.size]
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let nn = NeuralNetwork::new(&[2, 3, 1]);
        assert_eq!(nn.size, 2);
        assert_eq!(nn.weight.len(), 2);
        assert_eq!(nn.bias.len(), 2);
        assert_eq!(nn.activation.len(), 3);
        assert_eq!(nn.activation[0], Matrix::new(1, 2));
        assert_eq!(nn.activation[1], Matrix::new(1, 3));
        assert_eq!(nn.activation[2], Matrix::new(1, 1));
        assert_eq!(nn.weight[0], Matrix::new(2, 3));
        assert_eq!(nn.weight[1], Matrix::new(3, 1));
        assert_eq!(nn.bias[0], Matrix::new(1, 3));
        assert_eq!(nn.bias[1], Matrix::new(1, 1));
    }

    #[test]
    fn test_new_random() {
        let nn = NeuralNetwork::new_random(&[2, 3, 1]);
        assert_eq!(nn.size, 2);
        assert_eq!(nn.weight.len(), 2);
        assert_eq!(nn.bias.len(), 2);
        assert_eq!(nn.activation.len(), 3);
        assert_eq!(nn.activation[1].rows(), 1);
        assert_eq!(nn.activation[1].cols(), 3);
        assert_eq!(nn.activation[2].rows(), 1);
        assert_eq!(nn.activation[2].cols(), 1);
        assert!(nn.weight[0].iter().any(|&x| x != 0.0));
        assert!(nn.weight[1].iter().any(|&x| x != 0.0));
        assert!(nn.bias[0].iter().any(|&x| x != 0.0));
        assert!(nn.bias[1].iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_forward() {
        let mut nn = NeuralNetwork::new(&[2, 3, 1]);
        nn.weight[0] = Matrix::from_iter(2, 3, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        nn.weight[1] = Matrix::from_iter(3, 1, vec![0.7, 0.8, 0.9]);

        nn.bias[0] = Matrix::from_iter(1, 3, vec![0.1, 0.2, 0.3]);
        nn.bias[1] = Matrix::from_iter(1, 1, vec![0.4]);
        nn.activation[0] = Matrix::from_iter(1, 2, vec![0.5, 0.6]);

        nn.forward();

        assert_eq!(
            nn.activation[1],
            Matrix::from_iter(
                1,
                3,
                vec![0.5962826992967879, 0.6456563062257954, 0.6921095043017882]
            )
        );
        assert_eq!(
            nn.activation[2],
            Matrix::from_iter(1, 1, vec![0.8761885526812198])
        );
    }

    #[test]
    fn test_cost() {
        let mut nn = NeuralNetwork::new(&[2, 3, 1]);
        nn.weight[0] = Matrix::from_iter(2, 3, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        nn.weight[1] = Matrix::from_iter(3, 1, vec![0.7, 0.8, 0.9]);
        nn.bias[0] = Matrix::from_iter(1, 3, vec![0.1, 0.2, 0.3]);
        nn.bias[1] = Matrix::from_iter(1, 1, vec![0.4]);

        let input = Matrix::from_iter(2, 2, vec![0.1, 0.2, 0.3, 0.4]);
        let output = Matrix::from_iter(2, 1, vec![0.5, 0.6]);

        let cost = nn.cost(&input, &output);

        assert_eq!(cost, 0.0997167628239912);
    }

    #[test]
    fn test_finite_diff() {
        let mut nn = NeuralNetwork::new(&[2, 3, 1]);
        nn.weight[0] = Matrix::from_iter(2, 3, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        nn.weight[1] = Matrix::from_iter(3, 1, vec![0.7, 0.8, 0.9]);
        nn.bias[0] = Matrix::from_iter(1, 3, vec![0.1, 0.2, 0.3]);
        nn.bias[1] = Matrix::from_iter(1, 1, vec![0.4]);

        let input = Matrix::from_iter(2, 2, vec![0.1, 0.2, 0.3, 0.4]);
        let output = Matrix::from_iter(2, 1, vec![0.5, 0.6]);

        let mut gradient = NeuralNetwork::new(&[2, 3, 1]);
        let eps = 0.0001;
        nn.finite_diff(&mut gradient, &eps, &input, &output);

        assert_eq!(
            gradient.weight[0],
            Matrix::from_iter(
                2,
                3,
                vec![
                    0.002338431382142847,
                    0.0026023290392029885,
                    0.0028172072125132175,
                    0.0036226984256870765,
                    0.004037243530596868,
                    0.004379096906204083
                ]
            )
        );
        assert_eq!(
            gradient.weight[1],
            Matrix::from_iter(
                3,
                1,
                vec![
                    0.04148976830056772,
                    0.044146948279810694,
                    0.046728225587544525
                ]
            )
        );
        assert_eq!(
            gradient.bias[0],
            Matrix::from_iter(
                1,
                3,
                vec![
                    0.012842612617247617,
                    0.014349051058182294,
                    0.01561876383046612
                ]
            )
        );
        assert_eq!(
            gradient.bias[1],
            Matrix::from_iter(1, 1, vec![0.07441508547284537])
        );
    }

    #[test]
    fn test_learn() {
        let mut nn = NeuralNetwork::new(&[2, 3, 1]);
        nn.weight[0] = Matrix::from_iter(2, 3, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        nn.weight[1] = Matrix::from_iter(3, 1, vec![0.7, 0.8, 0.9]);
        nn.bias[0] = Matrix::from_iter(1, 3, vec![0.1, 0.2, 0.3]);
        nn.bias[1] = Matrix::from_iter(1, 1, vec![0.4]);
        nn.activation[0] = Matrix::from_iter(1, 2, vec![0.5, 0.6]);

        let mut gradient = NeuralNetwork::new(&[2, 3, 1]);
        gradient.weight[0] = Matrix::from_iter(2, 3, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        gradient.weight[1] = Matrix::from_iter(3, 1, vec![0.7, 0.8, 0.9]);
        gradient.bias[0] = Matrix::from_iter(1, 3, vec![0.1, 0.2, 0.3]);
        gradient.bias[1] = Matrix::from_iter(1, 1, vec![0.4]);

        nn.learn(&mut gradient, &0.1);

        assert_eq!(
            nn.weight[0],
            Matrix::from_iter(2, 3, vec![0.090, 0.180, 0.270, 0.360, 0.450, 0.540])
        );
        assert_eq!(
            nn.weight[1],
            Matrix::from_iter(3, 1, vec![0.630, 0.720, 0.810])
        );
        assert_eq!(nn.bias[0], Matrix::from_iter(1, 3, vec![0.1, 0.2, 0.3]));
        assert_eq!(nn.bias[1], Matrix::from_iter(1, 1, vec![0.4]));
    }

    #[test]
    fn test_nn_test() {
        let mut nn = NeuralNetwork::new(&[2, 3, 1]);
        nn.weight[0] = Matrix::from_iter(2, 3, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        nn.weight[1] = Matrix::from_iter(3, 1, vec![0.7, 0.8, 0.9]);
        nn.bias[0] = Matrix::from_iter(1, 3, vec![0.1, 0.2, 0.3]);
        nn.bias[1] = Matrix::from_iter(1, 1, vec![0.4]);

        let input = Matrix::from_iter(1, 2, vec![0.5, 0.6]);
        let output = nn.test(&input);

        assert_eq!(output, Matrix::from_iter(1, 1, vec![0.8761885526812198]));
    }
}

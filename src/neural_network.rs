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

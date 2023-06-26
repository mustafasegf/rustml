
use std::fmt::Display;
use std::ops::{Add, Sub};

use num_traits::NumRef;

use super::Matrix;

impl<T: NumRef + Clone + Display> Add for Matrix<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let data = self
            .data
            .into_iter()
            .zip(rhs.data.into_iter())
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();

        Self {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl<'a: 'b, 'b, T: NumRef + Clone + Display> Add for &'a Matrix<T>
where
    &'a T: Add<&'b T, Output = T>,
{
    type Output = Matrix<T>;

    fn add(self, rhs: &'b Matrix<T>) -> Self::Output {
        assert!(self.rows == rhs.rows);
        assert!(self.cols == rhs.cols);

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.iter().zip(rhs.iter()).map(|(a, b)| a + b).collect(),
        }
    }
}

impl<'a: 'b, 'b, T: NumRef + Clone + Display> Sub for &'a Matrix<T>
where
    &'a T: Sub<&'b T, Output = T>,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: &'b Matrix<T>) -> Self::Output {
        assert!(self.rows == rhs.rows);
        assert!(self.cols == rhs.cols);

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.iter().zip(rhs.iter()).map(|(a, b)| a - b).collect(),
        }
    }
}

impl<T: NumRef + Clone + Display> Sub for Matrix<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let data = self
            .data
            .into_iter()
            .zip(rhs.data.into_iter())
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();

        Self {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }
}
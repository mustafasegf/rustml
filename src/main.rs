use std::{fmt::Display};
use std::ops::{Add, Sub, Deref};

use num_traits::NumRef;
fn main() {
    let arr1 = Matrix::from_iter(3, 3, 0..9);
    let arr2 = Matrix::from_iter(3, 3, 10..);
    println!("{}", arr1);
    let arr3 = arr1 + arr2;
    println!("{}", arr3);
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd)]
pub struct Matrix<T: NumRef + Clone + Display> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: NumRef + Clone + Display> Matrix<T> {
    pub fn new(row: usize, col: usize) -> Self {
        Self::from_iter(row, col, std::iter::repeat(T::zero()))
    }

    pub fn from_iter<I>(row: usize, col: usize, iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let data = iter.into_iter().take(col * row).collect::<Vec<_>>();
        assert_eq!(data.len(), row * col);
        Self {
            data,
            rows: row,
            cols: col,
        }
    }
}

impl<T: NumRef + Clone + Display> Deref for Matrix<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

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

impl<T: NumRef + Clone + Display> std::fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in 0..self.rows {
            for col in 0..self.cols {
                write!(f, "{} ", self.data[row * self.cols + col])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

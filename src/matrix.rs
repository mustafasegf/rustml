mod ops;

use std::fmt::Display;
use std::ops::Deref;
use num_traits::NumRef;

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

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
}

impl<T: NumRef + Clone + Display> Deref for Matrix<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.data
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
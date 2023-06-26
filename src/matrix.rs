mod ops;

use num_traits::{NumAssignRef, NumRef};
use std::fmt::Display;
use std::ops::Deref;

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd)]
pub struct Matrix<T: NumRef + NumAssignRef + Copy + Display> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: NumRef + NumAssignRef + Copy + Display> Matrix<T> {
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

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        match row < self.rows && col < self.cols {
            true => Some(&self.data[col + row * self.cols]),
            false => None,
        }
    }

    pub fn get_row(&self, row: usize) -> Option<impl Iterator<Item = &T>> {
        match row < self.rows {
            true => Some(self.data[row * self.cols..(row + 1) * self.cols].iter()),
            false => None,
        }
    }

    pub fn get_col(&self, col: usize) -> Option<impl Iterator<Item = &T>> {
        match col < self.cols {
            true => Some((0..self.rows).map(move |row| self.get(row, col).unwrap())),
            false => None,
        }
    }
}

impl<T: NumRef + NumAssignRef + Copy + Display> Deref for Matrix<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T: NumRef + NumAssignRef + Copy + Display> std::fmt::Display for Matrix<T> {
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

mod ops;

// use rand::prelude::*;
use std::ops::Deref;

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}
impl Matrix {
    pub fn new(row: usize, col: usize) -> Self {
        Self::from_iter(row, col, std::iter::repeat(0.0))
    }

    pub fn from_iter<I>(row: usize, col: usize, iter: I) -> Self
    where
        I: IntoIterator<Item = f64>,
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

    pub fn get(&self, row: usize, col: usize) -> Option<&f64> {
        self.data.get(col + row * self.cols)
        // match row < self.rows && col < self.cols {
        //     true => Some(&self.data[col + row * self.cols]),
        //     false => None,
        // }
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut f64> {
        self.data.get_mut(col + row * self.cols)
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Option<f64> {
        let old = self.data.get(col + row * self.cols).map(|x| *x);
        self.data[col + row * self.cols] = value;
        old
    }

    pub fn get_row(&self, row: usize) -> Option<impl Iterator<Item = &f64>> {
        match row < self.rows {
            true => Some(self.data[row * self.cols..(row + 1) * self.cols].iter()),
            false => None,
        }
    }

    pub fn get_col(&self, col: usize) -> Option<impl Iterator<Item = &f64>> {
        match col < self.cols {
            true => Some((0..self.rows).map(move |row| self.get(row, col).unwrap())),
            false => None,
        }
    }

    pub fn get_row_matrix(&self, row: usize) -> Option<Self> {
        match self.get_row(row) {
            None => None,
            Some(data) => Some(Self {
                rows: 1,
                cols: self.cols,
                data: data.map(|x| *x).collect(),
            }),
        }
    }

    pub fn get_col_matrix(&self, row: usize) -> Option<Self> {
        match self.get_col(row) {
            None => None,
            Some(data) => Some(Self {
                rows: self.rows,
                cols: 1,
                data: data.map(|x| *x).collect(),
            }),
        }
    }

    

    pub fn sigmoid(&self) -> Self {
        Self::from_iter(
            self.rows,
            self.cols,
            self.iter().map(|x| 1.0 / (1.0 + (-*x).exp())),
        )
    }
}

impl Deref for Matrix {
    type Target = Vec<f64>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::fmt::Display for Matrix {
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

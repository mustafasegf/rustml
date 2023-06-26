use std::fmt::Display;
use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};

use num_traits::{NumAssignRef, NumRef};

use super::Matrix;

impl<T: NumRef + NumAssignRef + Copy + Display> Add for Matrix<T> {
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

impl<'a: 'b, 'b, T: NumRef + NumAssignRef + Copy + Display> Add for &'a Matrix<T>
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

impl<'a: 'b, 'b, T: NumRef + NumAssignRef + Copy + Display> Sub for &'a Matrix<T>
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

impl<T: NumRef + NumAssignRef + Copy + Display> Sub for Matrix<T> {
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

impl<T: NumRef + NumAssignRef + Copy + Display> AddAssign for Matrix<T> {
    fn add_assign(&mut self, rhs: Self) {
        assert!(self.rows == rhs.rows);
        assert!(self.cols == rhs.cols);

        self.data
            .iter_mut()
            .zip(rhs.data.into_iter())
            .for_each(|(a, b)| *a += b);
    }
}

impl<'a, T: NumRef + NumAssignRef + Copy + Display> AddAssign<&'a Matrix<T>> for Matrix<T> {
    fn add_assign(&mut self, rhs: &'a Self) {
        assert!(self.rows == rhs.rows);
        assert!(self.cols == rhs.cols);

        self.data
            .iter_mut()
            .zip(rhs.iter())
            .for_each(|(a, b)| *a += b);
    }
}

impl<T: NumRef + NumAssignRef + Copy + Display> SubAssign for Matrix<T> {
    fn sub_assign(&mut self, rhs: Self) {
        assert!(self.rows == rhs.rows);
        assert!(self.cols == rhs.cols);

        self.data
            .iter_mut()
            .zip(rhs.data.into_iter())
            .for_each(|(a, b)| *a -= b);
    }
}

impl<'a, T: NumRef + NumAssignRef + Copy + Display> SubAssign<&'a Matrix<T>> for Matrix<T> {
    fn sub_assign(&mut self, rhs: &'a Self) {
        assert!(self.rows == rhs.rows);
        assert!(self.cols == rhs.cols);

        self.data
            .iter_mut()
            .zip(rhs.iter())
            .for_each(|(a, b)| *a -= b);
    }
}

impl<T: NumRef + NumAssignRef + Copy + Display> Mul for Matrix<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        assert!(self.cols == rhs.rows);

        let mut data = Vec::with_capacity(self.rows * rhs.cols);

        for row in 0..self.rows {
            for col in 0..rhs.cols {
                let row_iter = self.get_row(row).unwrap();
                let col_iter = rhs.get_col(col).unwrap();

                let acc = row_iter
                    .zip(col_iter)
                    .fold(T::zero(), |acc, (a, b)| acc + (*a * *b));

                data.push(acc);
            }
        }

        Self {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

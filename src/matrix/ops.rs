use super::Matrix;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

impl Add for Matrix {
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

impl<'a: 'b, 'b> Add for &'a Matrix
where
    &'a f64: Add<&'b f64, Output = f64>,
{
    type Output = Matrix;

    fn add(self, rhs: &'b Matrix) -> Self::Output {
        assert!(self.rows == rhs.rows);
        assert!(self.cols == rhs.cols);

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.iter().zip(rhs.iter()).map(|(a, b)| a + b).collect(),
        }
    }
}

impl AddAssign for Matrix {
    fn add_assign(&mut self, rhs: Self) {
        assert!(self.rows == rhs.rows);
        assert!(self.cols == rhs.cols);

        self.data
            .iter_mut()
            .zip(rhs.data.into_iter())
            .for_each(|(a, b)| *a += b);
    }
}

impl<'a> AddAssign<&'a Matrix> for Matrix {
    fn add_assign(&mut self, rhs: &'a Self) {
        assert!(self.rows == rhs.rows);
        assert!(self.cols == rhs.cols);

        self.data
            .iter_mut()
            .zip(rhs.iter())
            .for_each(|(a, b)| *a += b);
    }
}

impl Sub for Matrix {
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

impl<'a: 'b, 'b> Sub for &'a Matrix
where
    &'a f64: Sub<&'b f64, Output = f64>,
{
    type Output = Matrix;

    fn sub(self, rhs: &'b Matrix) -> Self::Output {
        assert!(self.rows == rhs.rows);
        assert!(self.cols == rhs.cols);

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.iter().zip(rhs.iter()).map(|(a, b)| a - b).collect(),
        }
    }
}

impl SubAssign for Matrix {
    fn sub_assign(&mut self, rhs: Self) {
        assert!(self.rows == rhs.rows);
        assert!(self.cols == rhs.cols);

        self.data
            .iter_mut()
            .zip(rhs.data.into_iter())
            .for_each(|(a, b)| *a -= b);
    }
}

impl<'a> SubAssign<&'a Matrix> for Matrix {
    fn sub_assign(&mut self, rhs: &'a Self) {
        assert!(self.rows == rhs.rows);
        assert!(self.cols == rhs.cols);

        self.data
            .iter_mut()
            .zip(rhs.iter())
            .for_each(|(a, b)| *a -= b);
    }
}

impl Mul for Matrix {
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
                    .fold(0.0, |acc, (a, b)| acc + (*a * *b));

                data.push(acc);
            }
        }

        Self {
            data,
            rows: self.rows,
            cols: rhs.cols,
        }
    }
}

impl<'a: 'b, 'b> Mul for &'a Matrix
where
    &'a f64: Mul<&'b f64, Output = f64>,
{
    type Output = Matrix;

    fn mul(self, rhs: &'b Matrix) -> Self::Output {
        assert!(self.cols == rhs.rows);

        let mut data = Vec::with_capacity(self.rows * rhs.cols);

        for row in 0..self.rows {
            for col in 0..rhs.cols {
                let row_iter = self.get_row(row).unwrap();
                let col_iter = rhs.get_col(col).unwrap();

                let acc = row_iter
                    .zip(col_iter)
                    .fold(0.0, |acc, (a, b)| acc + (*a * *b));

                data.push(acc);
            }
        }

        Matrix {
            rows: self.rows,
            cols: rhs.cols,
            data,
        }
    }
}

impl MulAssign<Matrix> for Matrix {
    fn mul_assign(&mut self, rhs: Self) {
        assert!(self.cols == rhs.rows);

        let mut data = Vec::with_capacity(self.rows * rhs.cols);

        for row in 0..self.rows {
            for col in 0..rhs.cols {
                let row_iter = self.get_row(row).unwrap();
                let col_iter = rhs.get_col(col).unwrap();

                let acc = row_iter
                    .zip(col_iter)
                    .fold(0.0, |acc, (a, b)| acc + (*a * *b));

                data.push(acc);
            }
        }

        self.data = data;
        self.cols = rhs.cols;
    }
}

impl<'a: 'b, 'b> MulAssign<&'a Matrix> for Matrix
where
    &'a f64: Mul<&'b f64, Output = f64>,
{
    fn mul_assign(&mut self, rhs: &'b Matrix) {
        assert!(self.cols == rhs.rows);

        let mut data = Vec::with_capacity(self.rows * rhs.cols);

        for row in 0..self.rows {
            for col in 0..rhs.cols {
                let row_iter = self.get_row(row).unwrap();
                let col_iter = rhs.get_col(col).unwrap();

                let acc = row_iter
                    .zip(col_iter)
                    .fold(0.0, |acc, (a, b)| acc + (*a * *b));

                data.push(acc);
            }
        }

        self.data = data;
        self.cols = rhs.cols;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_add() {
        let a = Matrix::from_iter(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_iter(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let expected = Matrix::from_iter(2, 2, vec![6.0, 8.0, 10.0, 12.0]);
        assert_eq!(a + b, expected);
    }

    #[test]
    fn test_add_ref() {
        let a = Matrix::from_iter(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_iter(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let expected = Matrix::from_iter(2, 2, vec![6.0, 8.0, 10.0, 12.0]);
        assert_eq!(&a + &b, expected);
    }

    #[test]
    fn test_add_assign() {
        let mut a = Matrix::from_iter(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_iter(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let expected = Matrix::from_iter(2, 2, vec![6.0, 8.0, 10.0, 12.0]);
        a += b;
        assert_eq!(a, expected);
    }

    #[test]
    fn test_add_assign_ref() {
        let mut a = Matrix::from_iter(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_iter(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let expected = Matrix::from_iter(2, 2, vec![6.0, 8.0, 10.0, 12.0]);
        a += &b;
        assert_eq!(a, expected);
    }

    #[test]
    fn test_sub() {
        let a = Matrix::from_iter(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_iter(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let expected = Matrix::from_iter(2, 2, vec![-4.0, -4.0, -4.0, -4.0]);
        assert_eq!(a - b, expected);
    }

    #[test]
    fn test_sub_ref() {
        let a = Matrix::from_iter(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_iter(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let expected = Matrix::from_iter(2, 2, vec![-4.0, -4.0, -4.0, -4.0]);
        assert_eq!(&a - &b, expected);
    }

    #[test]
    fn test_sub_assign() {
        let mut a = Matrix::from_iter(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_iter(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let expected = Matrix::from_iter(2, 2, vec![-4.0, -4.0, -4.0, -4.0]);
        a -= b;
        assert_eq!(a, expected);
    }

    #[test]
    fn test_sub_assign_ref() {
        let mut a = Matrix::from_iter(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_iter(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let expected = Matrix::from_iter(2, 2, vec![-4.0, -4.0, -4.0, -4.0]);
        a -= &b;
        assert_eq!(a, expected);
    }

    #[test]
    fn test_mul() {
        let a = Matrix::from_iter(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::from_iter(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let expected = Matrix::from_iter(2, 2, vec![58.0, 64.0, 139.0, 154.0]);
        assert_eq!(a * b, expected);
    }

    #[test]
    fn test_mul_ref() {
        let a = Matrix::from_iter(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::from_iter(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let expected = Matrix::from_iter(2, 2, vec![58.0, 64.0, 139.0, 154.0]);
        assert_eq!(&a * &b, expected);
    }

    #[test]
    fn test_mul_assign() {
        let mut a = Matrix::from_iter(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::from_iter(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let expected = Matrix::from_iter(2, 2, vec![58.0, 64.0, 139.0, 154.0]);
        a *= b;
        assert_eq!(a, expected);
    }

    #[test]
    fn test_mul_assign_ref() {
        let mut a = Matrix::from_iter(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::from_iter(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let expected = Matrix::from_iter(2, 2, vec![58.0, 64.0, 139.0, 154.0]);
        a *= &b;
        assert_eq!(a, expected);
    }
}

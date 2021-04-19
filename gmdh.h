#pragma once
#include <time.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>

#include "mlpack/core.hpp"
#include "mlpack/methods/linear_regression/linear_regression.hpp"
#include "mlpack/core/cv/metrics/r2_score.hpp"
#include "mlpack/core/cv/metrics/accuracy.hpp"
#include "mlpack/core/cv/metrics/precision.hpp"
#include "mlpack/core/cv/metrics/recall.hpp"
#include "mlpack/core/cv/metrics/f1.hpp"
#include "mlpack/methods/kmeans/kmeans.hpp"

using namespace arma;

using criterion = double (*)(rowvec&, rowvec&, int);

criterion set (int task_type);
double max_corr (mat& buf, rowvec& pred, int buf_val);
void update_min_accu (rowvec& r2, int* ind_min, int *i0, int *i1, int buf_val, int step);
void init (mat& a, rowvec& b, int M, bool t);
void phi (rowvec& fuzzy, rowvec& x,
              double z1, double z2, double a, double b, int N);
void transform (rowvec& pred);
void confusion_matrix (rowvec& pred, rowvec& y,
                           double *tp, double *fp, double *fn, double *tn);
double r2_score (rowvec& pred, rowvec& y, int n);
double accuracy (rowvec& pred, rowvec& y, int n);

class GMDH
{
private:
  int Q;                       /* Размер буфера */
  double C;                    /* Порог корреляции */
  int I;                       /* Количество итераций */
  int K;                       /* Число отрезков разбиения в функции нечеткой принадлежности */
  rowvec col_nums;             /* Номера всех вошедших в буферы столбцов */
  rowvec best_cols;            /* Номера всех различных вошедших в буферы столбцов */
public:
  GMDH (int Q = 100, double C = 0.9, int I = 10, int K = 3);
  ~GMDH ();
  void fit (mat X, Row <size_t> y, double lambda, int task_type);
  void get_best_cols ();
  void predict (mat a, Row <size_t> y, double lambda, int task_type);
};

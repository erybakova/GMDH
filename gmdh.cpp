#include "gmdh.h"

using namespace mlpack::regression;
using namespace mlpack::cv;

criterion set (int task_type)
{
  switch (task_type)
  {
    case 0:
      return r2_score;
    case 1:
      return accuracy;
    default:
      return 0;
  }
}

double max_corr (mat& buf, rowvec& pred, int buf_val)
{
  double max = 0., curr;

  for (int i = 0; i < buf_val; i++)
  {
      curr = norm_dot(conv_to<vec>::from(buf.row(i)), conv_to<vec>::from(pred));
      max = (curr > max ? curr : max);
  }

  return max;
}

void update_min_accu (rowvec& r2, int* ind_min, int *i0, int *i1, int buf_val, int step)
{
  double min = 1.;

  for (int i = 0, j = 0; i < buf_val; i += 1, j += step)
  {
    if (r2(i) < min)
    {
      min = r2(i);
      *ind_min = i;
      *i0 = j;
      *i1 = j + 1;
    }
  }
}

void init (mat& a, rowvec& b, int M, bool t)
{
  if (t)
  {
    for (int i = 0; i < M; i++)
      b(i) = a.row(i).max();
  }
  else
  {
    for (int i = 0; i < M; i++)
      b(i) = a.row(i).min();
  }
}

void phi (rowvec& fuzzy, rowvec& x,
              double z1, double z2, double a, double b, int N)
{
  for (int j = 0; j < N; j++)
  {
    if (x(j) < z1)
      fuzzy(j) = 0.;

    else if (x(j) > z2)
      fuzzy(j) = b - a;

    /* если константный столбец, чтобы не делить на 0 */
    else if (z2 <= z1 && z2 >= z1)
      fuzzy(j) = x(j);

    else
      fuzzy(j) = ((b - a) / (z2 - z1)) * (x(j) - z1);
  }
}

void transform (rowvec& pred)
{
  for (int i = 0; i < pred.n_elem; i++)
  {
    if (pred[i] <= 0.5)
      pred[i] = 0;
    else
      pred[i] = 1;
  }
}

double r2_score (rowvec& pred, rowvec& y, int n)
{
  double pr = 0., mn = 0.;
  double y_mn = mean (y), curr;

  for (int i = 0; i < n; i++)
  {
    curr = pred[i] - y[i];
    pr += curr * curr;

    curr = y_mn - y[i];
    mn += curr * curr;
  }

  return 1. - pr / mn;
}

double accuracy (rowvec& pred, rowvec& y, int n)
{
  double total = 0.;

  for (int i = 0; i < n; i++)
  {
    if ((y[i] == 0 && pred[i] == 0)
          || (y[i] == 1 && pred[i] == 1))
      total += 1;
  }

  return total / n;
}

void confusion_matrix (rowvec& pred, rowvec& y,
                           double *tp, double *fp, double *fn, double *tn)
{
  for (int i = 0; i < y.n_elem; i++)
  {
    if (y[i] == 1)
    {
      if (pred[i] == 1)
        *tp += 1;
      else
        *fn += 1;
    }
    else if (pred[i] == 1)
      *fp += 1;
    else
      *tn += 1;
  }
}

GMDH:: GMDH (int Q, double C, int I, int K)
{
  this -> Q = Q;
  this -> C = C;
  this -> I = I;
  this -> K = K;
}

GMDH:: ~GMDH()
{}

void GMDH::fit (mat a, Row <size_t> y, double lambda, int task_type)
{
  int buf_val = 0, curr_buf_val = 0, col_val = 0, col_val_1;
  int i, j, k, m, l, i0, i1, ind_min;
  int M = a.n_rows, N = a.n_cols;
  double min_accu = 0., curr_accu = 0., accu = 0., best_accu = 0.;
  double curr_min, curr_max, z1, z2, h, mn, mc = 0.;

  rowvec pred (N);
  mat buf (Q, N);
  mat curr_buf (Q, N);
  rowvec r2 (Q);
  mat X_train (2, N);
  mat best_X_train (2, N);
  rowvec min (M);
  rowvec max (M);
  rowvec fuzzy (N);
  rowvec curr_row (N);

  criterion cr = set (task_type);

  /* Центрирование матрицы */
  mn = mean (mean (a));
  a -= mn;

  /* Нормирование матрицы */
  a = normalise(a, 2, 1);

  init (a, min, M, 0);
  init (a, max, M, 1);

  /* Первая селекция */
  for (i = 0; i < M; i++)
  {
    X_train.row(0) = a.row(i);

    for (j = i + 1; j < M; j++)
    {
      X_train.row(1) = a.row(j);

      LinearRegression lr (X_train, conv_to<rowvec>::from(y), lambda);
      lr.Predict (X_train, pred);

      /* Если буфер еще пуст, то
       * добавляем столбец в буфер */
      if (buf_val == 0)
      {
        buf.row(0) = pred;
        col_nums.insert_cols (0, 1);
        col_nums.insert_cols (1, 1);
        col_nums(0) = i;
        col_nums(1) = j;

        rowvec g = conv_to<rowvec>::from (y);
        min_accu =  cr (pred, g, N);
        ind_min = 0;
        i0 = 0; i1 = 1;

        r2(0) = min_accu;
        buf_val += 1;
        col_val += 2;
      }
      else
      {
        mc = max_corr (buf, pred, buf_val);
        rowvec g = conv_to<rowvec>::from (y);
        curr_accu = cr (pred, g, N);

        /* Если буфер не заполнен полностью и все попарные корреляции < C, то
         * добавляем столбец в буфер */
        if (buf_val < Q
             && mc < C)
        {
          buf.row(buf_val) = pred;
          col_nums.insert_cols (col_val, 1);
          col_nums.insert_cols (col_val + 1, 1);
          col_nums(col_val) = i;
          col_nums(col_val + 1) = j;

          if (min_accu > curr_accu)
          {
            min_accu = curr_accu;
            ind_min = buf_val;
            i0 = col_val; i1 = col_val + 1;
          }

          r2(buf_val) = curr_accu;
          buf_val += 1;
          col_val += 2;
        }
        /* Если буфер заполнен полностью и все попарные корреляции < C, то возможно столбец
         * надо добавить вместо какого-то из уже имеющихся, для этого считаем критерий качества */
        else if (buf_val == Q
                  && mc < C
                    && curr_accu > min_accu)
        {
          buf.row(ind_min) = pred;
          col_nums(i0) = i;
          col_nums(i1) = j;
          r2(ind_min) = curr_accu;

          update_min_accu (r2, &ind_min, &i0, &i1, buf_val, 2);
        }
      }
    }
  }

  col_val_1 = col_val;

  /* Остальные селекции */
  for (k = 1; k < I; k++)
  {
    for (i = 0; i < M; i++)
    {
      curr_min = min(i);
      curr_max = max(i);
      h = (curr_max - curr_min) / K;

      for (j = 0; j < buf.n_rows; j++)
      {
        if (h <= 0 && h >= 0)
        {
          best_X_train.row(0) = a.row(i);
          best_X_train.row(1) = buf.row(j);
        }
        else
        {
          X_train.row(1) = buf.row(j);

          for (m = 0; m < K; m++)
          {
            z1 = curr_min + m * h;

            for (l = m + 1; l < K; l++)
            {
              z2 = curr_min + l * h;

              curr_row = a.row(i);

              phi (fuzzy, curr_row, z1, z2, curr_min, curr_max, N);

              /* Перенормировка вектора */
              fuzzy = normalise (fuzzy, 2);

              X_train.row(0) = fuzzy;
              LinearRegression lr (X_train, conv_to<rowvec>::from(y), lambda);
              lr.Predict (X_train, pred);
              rowvec g = conv_to<rowvec>::from (y);
              accu = cr (pred, g, N);

              /* Не с чем сравнить - первая фазификация, либо
               * остальные фазификации при условии лучшего критерия качества */
              if ((m == 0 && l == 1)
                    || (accu > best_accu))
              {
                best_accu = accu;
                best_X_train = X_train;
              }
            }
          }
        }

        LinearRegression lr (best_X_train, conv_to<rowvec>::from(y), lambda);
        lr.Predict (best_X_train, pred);

        /* Если текущий буфер еще пуст, то
         * добавляем столбец в текущий буфер */
        if (curr_buf_val == 0)
        {
          curr_buf.row(0) = pred;
          col_nums.insert_cols (col_val, 1);
          col_nums(col_val) = i;

          rowvec g = conv_to<rowvec>::from (y);
          min_accu = cr (pred, g, N);
          ind_min = 0;
          i0 = col_val;

          r2(0) = min_accu;
          curr_buf_val += 1;
          col_val += 1;
        }
        else
        {
          mc = max_corr (curr_buf, pred, curr_buf_val);
          rowvec g = conv_to<rowvec>::from (y);
          curr_accu = cr (pred, g, N);

          /* Если буфер не заполнен полностью и все попарные корреляции < C, то
           * добавляем столбец в буфер */
          if (curr_buf_val < Q
                && mc < C)
          {
            curr_buf.row (curr_buf_val) = pred;
            col_nums.insert_cols (col_val, 1);
            col_nums (col_val) = i;

            if (min_accu > curr_accu)
            {
              min_accu = curr_accu;
              ind_min = curr_buf_val;
              i0 = col_val;
            }

            r2 (curr_buf_val) = curr_accu;
            curr_buf_val += 1;
            col_val += 1;
          }
          /* Если буфер заполнен полностью и все попарные корреляции < C, то возможно
           * столбец надо добавить вместо какого-то из уже имеющихся, для этого считаем критерий качества */
          else if (curr_buf_val == Q
                     && mc < C
                       && curr_accu > min_accu)
          {
            curr_buf.row (ind_min) = pred;
            col_nums (i0) = i;
            r2 (ind_min) = curr_accu;

            int prev = i0;
            update_min_accu (r2, &ind_min, &i0, &i1, curr_buf_val, 1);

            /* Возможно, критерий качества уже стал константным,
             * тогда i0 не нужно обновлять */
            if (i0 != prev)
              i0 += col_val_1;
          }
        }
      }
    }
    buf = curr_buf;
  }
}

void GMDH:: get_best_cols ()
{
  int j = 0;

  for (int i = 0; i < col_nums.n_elem; i++)
  {
    if (best_cols.n_elem == 0
          || conv_to<uvec>:: from (find (best_cols == col_nums(i))).n_elem == 0)
    {
      best_cols.insert_cols (j, 1);
      best_cols (j) = col_nums (i);
      j++;
    }
  }

  best_cols = sort (best_cols);
  //best_cols.print ("Best cols:");
  fprintf (stdout, "Best_cols:  %d\n", (int) best_cols.n_elem);
}

void GMDH:: predict (mat a, Row <size_t> y, double lambda, int task_type)
{
  mat X (best_cols.n_elem, a.n_cols);
  rowvec predictions (a.n_cols);
  rowvec rvy = conv_to<rowvec>::from (y);

  for (int i = 0; i < best_cols.n_elem; i++)
    X.row (i) = a.row (best_cols (i));

  LinearRegression lr (X, conv_to<rowvec>::from (y), lambda);
  lr.Predict (X, predictions);

  if (task_type == 0)
  {
    double r2 = r2_score (predictions, rvy, a.n_cols);

    fprintf (stdout, "R2-score  = %.5f\n", r2);
  }
  else if (task_type == 1)
  {
    double tp = 0., fp = 0., fn = 0., tn = 0.;
    transform (predictions);
    confusion_matrix (predictions, rvy, &tp, &fp, &fn, &tn);

    double accuracy  = (tp + tn) / (tp + fp + tn + fn);
    double precision = tp / (tp + fp);
    double recall    = tp / (tp + fn);
    double fscore    = 2. * precision * recall / (precision + recall);

    fprintf (stdout, "Accuracy  = %.5f\n", accuracy);
    fprintf (stdout, "Precision = %.5f\n", precision);
    fprintf (stdout, "Recall    = %.5f\n", recall);
    fprintf (stdout, "F-score   = %.5f\n", fscore);
  }
}

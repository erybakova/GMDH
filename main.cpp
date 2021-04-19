#include "gmdh.h"

using namespace mlpack::regression;
using namespace mlpack::kmeans;
using namespace mlpack::tree;
using namespace mlpack::cv;
using namespace mlpack;
using namespace arma;

int main (int argc, char *argv[])
{
  int Q, I, K, task_type;
  double C, lambda;
  char *descriptors, *dataset;

  if (argc != 9 || (Q = atoi(argv[1])) <= 0 || (C = atof(argv[2])) <= 0 || (I = atoi(argv[3])) <= 0
        || (K = atoi(argv[4])) <= 0 || (lambda = atof(argv[5])) < 0 || (task_type = atoi(argv[8])) < 0)
  {
    fprintf (stderr, "Usage: ./a.out Q C I K lambda descriptors.csv dataset.csv <task type>\n"
                     "Note:  <task type> press 0 for regression\n"
                     "                   press 1 for classification\n");
    return -1;
  }

  descriptors = argv[6];
  dataset = argv[7];

  fprintf (stdout, "Loading %s\n", dataset);
  fprintf (stdout, "Q = %d C = %.5f I = %d K = %d lambda = %.5f\n", Q, C, I, K, lambda);

  Mat <double> d;
  Mat <double> a;
  Mat <double> a1;
  Row <size_t> b;
  Row <size_t> b1;

  if (!(mlpack::data::Load (descriptors, d)))
  {
    fprintf (stderr, "Error loading descriptors!\n");
    return -1;
  }

  if (!(mlpack::data::Load (dataset, a)))
  {
    fprintf (stderr, "Error loading dataset!\n");
    return -1;
  }

  d.shed_col (0);
  a.shed_col (0);
  b = conv_to <Row <size_t>>::from (a.row (a.n_rows - 1));
  a.shed_row (a.n_rows - 1);

  //printf ("m = %d, n = %d\n", (int)a.n_rows, (int)a.n_cols);

  int n_clusters = 3;
  Row <size_t> clusters;
  Mat <double> centroids;
  KMeans <> k;
  k.Cluster (d, n_clusters, clusters, centroids);
  //printf ("len(clusters) = %d\n", (int) clusters.n_elem);
  //clusters.print ("Clusters:");

  double t0 = clock();

  for (int i = 0; i < n_clusters; i++)
  {
    double t = clock();
    int l = 0;

    fprintf (stdout, "Cluster %d ", i);

    for (int j = 0; j < clusters.n_elem; j++)
    {
      if (clusters[j] == i)
      {
        a1.insert_cols (l, a.col(j));
        b1.insert_cols (l, 1);
        b1[l] = b[j];
        l++;
      }
    }

    fprintf (stdout, "(%d points)\n", (int) a1.n_cols);

    GMDH model (Q, C, I, K);
    model.fit (a1, b1, lambda, task_type);
    model.get_best_cols ();
    model.predict (a1, b1, lambda, task_type);

    t = (clock() - t) / CLOCKS_PER_SEC;
    fprintf (stdout, "Elapsed:    %.5fs\n", t);

    a1.clear ();
    b1.clear ();
  }

  t0 = (clock() - t0) / CLOCKS_PER_SEC;
  fprintf (stdout, "Total time: %.5fs\n", t0);

  fprintf (stdout, "Ready %s\n\n", dataset);

  return 0;
}

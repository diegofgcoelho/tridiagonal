//#include <iostream>
//#include <cstdio>
//#include <cstdlib>
//
//#include <gsl/gsl_spblas.h>
//#include <gsl/gsl_spmatrix.h>
//
//
//int main()
//{
//	gsl_spmatrix *A = gsl_spmatrix_alloc(5, 4);
//
//	std::cout << "Hey, I print on the screen!"<<std::endl;
//	return 0;
//}
//
//void* syl_kac_matrix(unsigned int n, gsl_spmatrix* A){
//
//}

//#include <stdio.h>
//#include <stdlib.h>
//
//#include <gsl/gsl_spmatrix.h>
//
//int
//main()
//{
//  gsl_spmatrix *A = gsl_spmatrix_alloc(5, 4); /* triplet format */
//  gsl_spmatrix *B, *C;
//  size_t i, j;
//
//  /* build the sparse matrix */
//  gsl_spmatrix_set(A, 0, 2, 3.1);
//  gsl_spmatrix_set(A, 0, 3, 4.6);
//  gsl_spmatrix_set(A, 1, 0, 1.0);
//  gsl_spmatrix_set(A, 1, 2, 7.2);
//  gsl_spmatrix_set(A, 3, 0, 2.1);
//  gsl_spmatrix_set(A, 3, 1, 2.9);
//  gsl_spmatrix_set(A, 3, 3, 8.5);
//  gsl_spmatrix_set(A, 4, 0, 4.1);
//
//  printf("printing all matrix elements:\n");
//  for (i = 0; i < 5; ++i)
//    for (j = 0; j < 4; ++j)
//      printf("A(%zu,%zu) = %g\n", i, j,
//             gsl_spmatrix_get(A, i, j));
//
//  /* print out elements in triplet format */
//  printf("matrix in triplet format (i,j,Aij):\n");
//  gsl_spmatrix_fprintf(stdout, A, "%.1f");
//
//  /* convert to compressed column format */
//  printf("\n\n before converting to ccs\n\n");
//  B = gsl_spmatrix_ccs(A);
//
//  printf("matrix in compressed column format:\n");
//  printf("i = [ ");
//  for (i = 0; i < B->nz; ++i)
//    printf("%zu, ", B->i[i]);
//  printf("]\n");
//
//  printf("p = [ ");
//  for (i = 0; i < B->size2 + 1; ++i)
//    printf("%zu, ", B->p[i]);
//  printf("]\n");
//
//  printf("d = [ ");
//  for (i = 0; i < B->nz; ++i)
//    printf("%g, ", B->data[i]);
//  printf("]\n");
//
//  /* convert to compressed row format */
//  C = gsl_spmatrix_crs(A);
//
//  printf("matrix in compressed row format:\n");
//  printf("i = [ ");
//  for (i = 0; i < C->nz; ++i)
//    printf("%zu, ", C->i[i]);
//  printf("]\n");
//
//  printf("p = [ ");
//  for (i = 0; i < C->size1 + 1; ++i)
//    printf("%zu, ", C->p[i]);
//  printf("]\n");
//
//  printf("d = [ ");
//  for (i = 0; i < C->nz; ++i)
//    printf("%g, ", C->data[i]);
//  printf("]\n");
//
//  gsl_spmatrix_free(A);
//  gsl_spmatrix_free(B);
//  gsl_spmatrix_free(C);
//
//  return 0;
//}


#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_statistics.h>

int main2 (void)
{
  /* declare variables */
  int i,j,rows,cols;
  gsl_matrix *m,*mm;
  gsl_vector *v;
  gsl_vector_view vv;
  FILE *f;

  rows=4;cols=3;
  m = gsl_matrix_alloc(rows,cols); /* create a matrix */

  /* initialise a matrix */
  for (i=0;i<rows;i++)
    {
      for (j=0;j<cols;j++)
	{
	  gsl_matrix_set(m,i,j,10.0*i+j);
	}
    }

  /* print matrix the hard way */
  printf("Matrix m\n");
  for (i=0;i<rows;i++)
    {
      for (j=0;j<cols;j++)
	{
	  printf("%f ",gsl_matrix_get(m,i,j));
	}
      printf("\n");
    }
  printf("\n");

  //insert the matrix into a file
  f=fopen("gsl_test_matvec.txt","w");
  gsl_matrix_fprintf(f,m,"%f");
  fclose(f);

  /* read in a matrix from a file */
  mm=gsl_matrix_alloc(rows,cols);
  f=fopen("gsl_test_matvec.txt","r");
  gsl_matrix_fscanf(f,mm);
  fclose(f);

  /* print matrix the easy way */
  printf("Matrix mm\n");
  gsl_matrix_fprintf(stdout,mm,"%f");

  /* put column means into a vector */
  v=gsl_vector_alloc(cols);
  for (i=0;i<cols;i++)
    {
      vv=gsl_matrix_column(mm,i);
      printf("\nCol %d\n",i);
      gsl_vector_fprintf(stdout,&vv.vector,"%f");
      gsl_vector_set(v,i,gsl_stats_mean(vv.vector.data,vv.vector.stride,vv.vector.size));
    }
  /* print column means */
  printf("\nColumn means\n");
  gsl_vector_fprintf(stdout,v,"%f");
  /* print overall mean */
  printf("\nOverall mean\n");
  printf("%f\n",gsl_stats_mean(v->data,v->stride,v->size));

  /* release memory */
  gsl_matrix_free(m);
  gsl_matrix_free(mm);
  gsl_vector_free(v);
  return(0);
}




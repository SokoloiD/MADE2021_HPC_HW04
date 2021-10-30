#include <stdio.h>
#include <cblas.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>




#define MATRIX_IDX(n, i, j) j*n + i
#define MATRIX_ELEMENT(A, m, n, i, j) A[ MATRIX_IDX(m, i, j) ]
const float CONN_PROB=0.2;

void initRandAdjMatrix(double* A, int m, int n){
   srand(time(NULL));   
   int threshold = RAND_MAX * CONN_PROB;  
   double element = 1.0;
   for (int j = 0; j < n; j++){
      for (int i = j; i < m; i++){
         
         int r = rand(); 
         if (r < threshold)
            element = 1;
         else
            element = 0;

         MATRIX_ELEMENT(A, m, n, i, j) = element;
         MATRIX_ELEMENT(A, m, n, j, i) = element;
         element *= 0.9;
      }
   }
}


void addMatrix(double *target, double *sourceA, double *sourceB, int n){

   int nSq = n * n;
   #pragma omp parallel for private(i) shared(target, sourceA, sourceB)
   for(int i = 0; i < nSq; i++){
      target[i] = sourceA[i] + sourceB[i];

   }
}


void copyMatrix(double *target, double *sourceA, int n){

   int nSq = n * n;

   #pragma omp parallel for private(i) shared(target, sourceA)
   for(int i = 0; i < nSq; i++){
      target[i] = sourceA[i];

   }
}

void makeEMatrix(double *target, int n){

   int nSq = n * n, step = n + 1;

   #pragma omp parallel for private(i) shared(target)
   for(int i = 1; i < nSq; ++i){
         target[i] = 0;
   }


  for(int i = 0; i < nSq; i+=step){
         target[i] = 1;
        
   }

}


void calc2PowerMatrixArray(double **MatixArray, int n, int n_len2){

   for( int i = 2; i < n_len2; ++i){
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, MatixArray[i - 1], n, MatixArray[i - 1], n, 0.0, MatixArray[i], n);

   }
}


void quickPowMatrix(double *target, double **matrixPow2, int pow, int n){

   int isEmpty = 1;

   int i = 0;
   while(pow){
      i++;

      if ( pow & 1){
         if(isEmpty){
            copyMatrix(target, matrixPow2[i], n);
            isEmpty = 0;      
         }
         else{

            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, matrixPow2[i], n, target, n, 0.0,target, n);
         }

      }
      pow >>=1;         


   }
}


int calcTotalRoutes(double *matrix,  int n){

   int sum = 0;
   int nPow2 = n * n;   
   #pragma omp parallel for private(i) shared(nPow2, matrix) reduce(+: sum)
   for(int i = 0; i < nPow2; ++i){
      sum += (int)matrix[i];

   }
   return sum;
}

void printMatrix(const double* A, int m, int n)
{
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
          printf("%8.1f", MATRIX_ELEMENT(A, m, n, i, j));
      }
      printf("\n");
   }
}




int main(int argc, char** argv)
{
   int n = 5;
   int n_len2 = 0;
   // calc array mpow_len   
   int tmp = n;

   while (tmp ){
      n_len2++;
      tmp >>= 1;
   }

   printf("N is %d, . Len of n2 pow is %d \n", n, n_len2);
  
   // allocate memory

   double *resultMatrix, *tmpMatrix, **M_pow2; 
   
   
   resultMatrix = (double *) malloc(n * n * sizeof(double));
   tmpMatrix = (double *) malloc(n * n * sizeof(double));



   M_pow2 = (double **)malloc( (n_len2 + 1) * sizeof(double*) );
   for( int i = 0 ; i <= n_len2 ; ++i){
      M_pow2[i] = (double *) malloc(n * n * sizeof(double));

   }


   // init first matrix
   initRandAdjMatrix(M_pow2[1], n, n);
   makeEMatrix(M_pow2[0], n);
   printf("Rand matrix is\n");
   printMatrix(M_pow2[1], n, n);


   // calc 2 pow(n) matrix  array
   calc2PowerMatrixArray(M_pow2, n ,n_len2);

   for(int i = 0; i <= n_len2; ++i)   {
      printf("Matrix in %d pow is\n", i);
      printMatrix(M_pow2[i],n ,n);
   }
  
   //init target matrix 
   makeEMatrix(resultMatrix, n);


   // calc routes matrix
   for (int i = 1; i < n; ++i){
      quickPowMatrix(tmpMatrix, M_pow2, i, n);
      addMatrix(resultMatrix, resultMatrix, tmpMatrix, n);

   }

   printf("Route matrix is is\n");
   printMatrix(resultMatrix, n, n);


   int totalRoutes = calcTotalRoutes(resultMatrix, n);
   printf("Total count of routes %d\n", totalRoutes);




   // free memory
   for( int i = 0; i<= n_len2; ++i)
      free(M_pow2[i]);
   free(M_pow2);
   free(resultMatrix); 
   free(tmpMatrix); 

   

   return 0;
}

 
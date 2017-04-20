#include <cstdio>
#include <ctime>
#include <iostream>
#include <algorithm>
#include <ctime>
#include <cstring>
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_spblas.h>
#include <gsl/gsl_rng.h>

//macro for convergence of power method
#define POWER_CONV 2
//macro for no convergence of power method
#define POWER_NO_CONV -2
//macro for false iteration value
#define POWER_NO_ITER -1
//macro for the maximum number of iterations
#define MAXITE 2000

void fill_spmatrix_kac(gsl_spmatrix*);
int power_method(gsl_spmatrix *m, gsl_vector *v, double* lambda, unsigned maxit, double prec, unsigned* iter);
double max_mag(gsl_vector* v);
bool check_stop(gsl_vector* v, gsl_vector* vv, gsl_vector* vvv, double prec);
void save_table(char const* filename, double** table, unsigned size1, unsigned size2, char const* format, bool latex);
void square_trid(gsl_spmatrix const* m, gsl_spmatrix** tempmm);
bool my_gsl_spmatrix_equal(gsl_spmatrix const* m1, gsl_spmatrix const* m2, double prec);
//gsl_spmatrix* my_gsl_spmatrix_triplet(const gsl_spmatrix* m);

int main(){
	//Sparse matrices pointers
	gsl_spmatrix *m, *tempm, *mm;
	//Defining random number generator: Tausworth
	const gsl_rng_type *my_rng_type;
	gsl_rng *tausrng;
	//The choosen seed
	const long int seed = 33275;

	//Configuring the random number generator as Taus
	my_rng_type = gsl_rng_taus;
	tausrng = gsl_rng_alloc(my_rng_type);
	gsl_rng_set(tausrng, seed);

	//Tridiagonal matrices dimensions. This is useful for determining the nnz for each matrix
	unsigned const sp_trid_n[] = {50, 100, 500, 1000, 5000};
	//Storing the number of different matrix sizes we will test (the size of array sp_trid_n)
	unsigned const n_sp_trid_n = 5;
	double iter_array[n_sp_trid_n];
	std::fill_n(&iter_array[0], n_sp_trid_n, 0);
	double miter_array[n_sp_trid_n];
	std::fill_n(&miter_array[0], n_sp_trid_n, 0);
	//Eigenvalue estimates
	double lambdas_array[n_sp_trid_n];
	std::fill_n(&lambdas_array[0], n_sp_trid_n, 0);
	double mlambdas_array[n_sp_trid_n];
	std::fill_n(&mlambdas_array[0], n_sp_trid_n, 0);
	//The precision used for simulation
	double symprec = 1e-3;

	//Auxiliary variables for counting the time
	clock_t time_beg, time_end;
	/*Vectors that will store the average times and number of iterations for the usual power
	 * method (times_avg and iter_avg) and modified power method (mtimes_avg and miter_avg).
	 */
	double times_array[n_sp_trid_n], mtimes_array[n_sp_trid_n];
	std::fill_n(&times_array[0], n_sp_trid_n, 0);
	std::fill_n(&mtimes_array[0], n_sp_trid_n, 0);

	//The number of replicates
	unsigned const rep = 100;

	for(unsigned i = 0; i < n_sp_trid_n; i++){
		//Defining the number of nonzero elements for the tridiagonal matrix
		unsigned const tridnnz = 3*sp_trid_n[i]-2;

		//Allocating memory for the tridiagonal matrix. Note: all the elements are set to zero
		m = gsl_spmatrix_alloc_nzmax(sp_trid_n[i], sp_trid_n[i], tridnnz, GSL_SPMATRIX_TRIPLET);
		//Allocating memory for the square of the tridiagonal matrix. Note: all the elements are set to zero
		//Not needed anymore because of the function square_trid
		//mm = gsl_spmatrix_alloc_nzmax(sp_trid_n[i], sp_trid_n[i], pentnnz, GSL_SPMATRIX_CCS);

		//Fill the matrix entries to be a Kac-Sylvester-Clement matrix
		fill_spmatrix_kac(m);

		//Converting m to CCS format
		tempm = m;
		m = gsl_spmatrix_ccs(tempm);

		//Auxiliary variable representing the initial guess for the matrix eigenvector
		gsl_vector* v = gsl_vector_alloc(sp_trid_n[i]);
		gsl_vector* _v = gsl_vector_alloc(sp_trid_n[i]);
		//Initiating all the elements of v
		for(unsigned j = 0; j < _v->size; j++){
			gsl_vector_set(_v, j, gsl_rng_uniform(tausrng));
		}
		//Alternate initialization for testing and comparison with matlab/octave
		gsl_vector_set_all(v, 1.0);

		//Run over all replicates for matrix size sp_tridi_n[i]
		for (unsigned j = 0; j < rep; ++j) {
			//Auxiliary variables used for getting the number of iterations and the eigenvalue estimate
			double lambda;
			unsigned iter;

			//Setting the initial values of v that are stored in _V
			gsl_vector_memcpy(v, _v);

			/*Calling power_method function that returns the estimate for the largest eigenvalue
			 * of a matrix
			 */
			time_beg = clock();
			power_method(m, v, &lambda, MAXITE, symprec, &iter);
			time_end = clock();
			times_array[i]+=1000*(time_end-time_beg)/(double)CLOCKS_PER_SEC;
			iter_array[i]+=iter;
			lambdas_array[i]+=fabs((lambda-(sp_trid_n[i]-1))/(sp_trid_n[i]-1));

			//Resetting all the elements of v to the same initial guess as for usual power method.
			gsl_vector_memcpy(v, _v);

			/*Calling power_method function with the matrix square that returns the estimate for the largest eigenvalue
			 * of a matrix
			 */
			time_beg = clock();//here we take the time used for computing the square into account
			//Calculating the matrix square without fast algorithm
			//gsl_spblas_dgemm(1, m, m, mm);
			//Calculating the matrix square with fast algorithm
			square_trid(m, &mm);
			/*gsl_spmatrix* testmm = NULL;
			square_trid(m, &testmm);

			if(gsl_spmatrix_equal(mm, testmm)!=1){
				printf("\n\n The matrix are NOT EQUAL\n\n");
			};
			gsl_spmatrix_free(testmm);*/


			power_method(mm, v, &lambda, MAXITE, symprec, &iter);
			time_end = clock();
			//Deallocating the memory used to store the square matrix
			gsl_spmatrix_free(mm);
			mtimes_array[i]+=1000*(time_end-time_beg)/(double)CLOCKS_PER_SEC;
			miter_array[i]+=iter;
			mlambdas_array[i]+=fabs((sqrt(lambda)-(sp_trid_n[i]-1))/(sp_trid_n[i]-1));
		}

		//Computing the average values for time, number of iterations and estimate error
		iter_array[i]/=rep;
		miter_array[i]/=rep;
		lambdas_array[i]/=rep;
		mlambdas_array[i]/=rep;
		times_array[i]/=rep;
		mtimes_array[i]/=rep;

		//Free the memory space for the matrices and vectors
		gsl_spmatrix_free(m);
		gsl_spmatrix_free(tempm);
		//gsl_spmatrix_free(mm);
		gsl_vector_free(v);
	}

	printf("\n\n*****The statistics for the usual method*****\n\n");
	printf("Size Iter Err Time\n");
	for(unsigned j=0; j<n_sp_trid_n; j++){
		printf("\t%d\t%.3f\t%.3e\t%.3f\n\n", sp_trid_n[j], iter_array[j], lambdas_array[j], times_array[j]);
	}
	printf("\n\n*****The statistics for the modified method*****\n\n");
	printf("Size Iter Err Time\n");
	for(unsigned j=0; j<n_sp_trid_n; j++){
		printf("\t%d\t%.3f\t%.3e\t%.3f\n\n", sp_trid_n[j], miter_array[j], mlambdas_array[j], mtimes_array[j]);
	}

	//Forming table to be printed through save_table function
	double** measures_table = (double**) malloc(sizeof(double*)*n_sp_trid_n);
	for(unsigned i = 0; i < n_sp_trid_n; i++){
		measures_table[i] = (double*)malloc(sizeof(double)*4);
		//Setting the content of each row of the table to be print
		measures_table[i][0] = sp_trid_n[i];
		measures_table[i][1] = iter_array[i];
		measures_table[i][2] = lambdas_array[i];
		measures_table[i][3] = times_array[i];
	}

	//File name
	char filename_usual[] = "measures_usual.txt";
	//Printing the table measures_table
	save_table(filename_usual, measures_table, n_sp_trid_n, 4, "%.0f%.0f%.3e%.3f", false);

	//Freeing all the allocated memory for printing the table
	for(unsigned i = 0; i < n_sp_trid_n; i++){
		delete[] measures_table[i];
	}
	delete[] measures_table;

	return 0;
}

void fill_spmatrix_kac(gsl_spmatrix* m){
	//Input: m is a gls_spmatrix
	//Requirement: m is in TRIPLET format, have null entries and represent a tridiagonal matrix
	//Description: this function modifies the entries of m making it become a Kac-Sylvester-Clement matrix

	//Number of rows of the matrix
	unsigned nrows = m->size1;

	//Note that the main diagonal does not need to be set since it has only null values
	for(unsigned i = 0; i < nrows-1; i++){
		//Setting the elements of the upper diagonal
		gsl_spmatrix_set(m, i, i+1, i+1);
		//Setting the elements of the lower diagonal
		gsl_spmatrix_set(m, i+1, i, nrows-i-1);
	}
}

int power_method(gsl_spmatrix *m, gsl_vector *v, double* lambda, unsigned maxit, double prec, unsigned* iter){
	/*Input:
	 * m is a gsl_spmatrix
	 * v is a gsl_vector representing the initial guess for the eigenvector associated with the
	 * largest eigenvalue
	 * maxit is the maximum number of iterations
	 * prec is the preision using for the stopping criteria for the vector difference norm
	 */
	/*Output:
	 * v represents the eigenvector associated with the largest eigenvalue
	 * lambda represents the largest eigenvalue
	 * iter is the number of iterations used for computing the largest eigenvalue
	 */
	/*Requirement:
	 * m must be initialized and the largest eigenvalue and its eigenvector
	 * must be real
	 * v must be non-null
	 * maxit must be greater than 1
	 */
	/*Description: this function returns the largest eigenvalue of the input matrix m using
	 * power method
	 */

	//Output variable
	int outputflag = -100;
	//Sanity Check: if the maximum number of iteration is invalid, return POWER_NO_ITER
	if(maxit <= 1){
		printf("\nError: the maximum number of iterations must be greater than 1.\n");
		return POWER_NO_ITER;
	}

	//Temporary and auxiliary variables:
	//Auxiliary variables for eigenvector estimate for stopping criteria
	gsl_vector* vv = gsl_vector_alloc(v->size);
	gsl_vector* vvv = gsl_vector_alloc(v->size);

	//Scaling the initial estimate for the eigenvector
	*lambda = max_mag(v);
	gsl_vector_scale(v, *lambda);

	//Auxiliary variable for counting the number of iterations
	unsigned i = 0;

	for(i = 0; i < maxit; i++){
		//Copying the value of vv to vvv
		gsl_vector_memcpy(vvv, vv);
		//Copying the value of v to vv
		gsl_vector_memcpy(vv, v);

		//Perform matrix-vector multiplication using BLAS operation
		gsl_spblas_dgemv(CblasNoTrans, 1/(*lambda), m, vv, 0, v);

		//Scaling the initial estimate for the eigenvector
		*lambda = max_mag(v);

		//Stopping criteria
		if(check_stop(v, vv, vvv, prec)){
			//if the returned value is true, leave the loop
			outputflag = POWER_CONV;
			break;
		}
	}

	//Passing the number of iterations used
	*iter = i;

	//Setting outputflag in case of no convergence
	if(i == maxit){
		outputflag = POWER_NO_CONV;
	}

	gsl_vector_free(vv);
	gsl_vector_free(vvv);

	return outputflag;
}

double max_mag(gsl_vector* v){
	/*Input:
	 * m is a gsl_vector
	 */
	/*Output:
	 * maxv is a double
	 */
	/*Requirement:
	 * v must be a non-empty vector
	 */
	/*Description: this function returns the absolute value component of
	 * the largest component
	 */
	//Value to be passed to the output
	double maxv = 0.0;

	for(unsigned i = 0; i < v->size; i++){
		double temp = gsl_vector_get(v, i);
		//If temp is greater than maxv, update maxv
		fabs(temp)>fabs(maxv)?maxv=temp:temp=0.0;
	}

	return maxv;
}

bool check_stop(gsl_vector* v, gsl_vector* vv, gsl_vector* vvv, double prec){
	/*Input:
	 * v is a gsl_vector
	 * vv is a gsl_vector
	 * vvv is a gsl_vector
	 * prec is a double
	 */
	/*Output:
	 *flag is a bool
	 */
	/*Requirement:
	 * v and vv must be non-empty vectors and prec must be greater than 0
	 */
	/*Description:
	 * The function compare vv and vvv against v (the current eigenvector estimate) and
	 * determines if the power method have achieved convergence
	 */

	//Output variable
	bool flagstop;

	//Auxiliary variables representing the norms of v, v-vv and v-vvv, respectively
	double normv = 0.0, normvvd = 0.0, normvvvd = 0.0;

	for(unsigned i = 0; i < v->size; i++){
		double tempv = gsl_vector_get(v, i);
		double tempvv = gsl_vector_get(vv, i);
		double tempvvv = gsl_vector_get(vvv, i);

		normv += pow(tempv,2);
		normvvd += pow(tempv-tempvv, 2);
		normvvvd += pow(tempv-tempvvv, 2);
	}

	normv = sqrt(normv);
	normvvd = sqrt(normvvd);
	normvvvd = sqrt(normvvvd);

	//Assigning the value of the output variable
	((normvvd/normv <= prec)||(normvvvd/normv <= prec))?flagstop=true:flagstop=false;

	return flagstop;

}

void save_table(char const* filename, double** table, unsigned size1, unsigned size2, char const* format, bool latex){
	/*Input:
	 * filename is a char representing the name of the file to be save
	 * table is the pointer to the table to be saved--pointer to pointer of doubles
	 * size1 and size2 represents the number of rows and columns of the input argument table
	 * format is a array of chars representing the format that the table must be printed
	 * separetor is the type of separation between the table entries to be printed. If you are using latex format,
	 * this variable value is discarded
	 * latex is a bool representing if the user want the table to be print in latex format or not
	 */
	/*Output:
	 * void, but the function creates the file with filename and print the table in it
	 */
	/*
	 * Requirement:
	 * filename must be a valid name for a file and have no blank space and must be not exist--unless intentionally
	 * you want overwrite them
	 */
	/*Description:
	 * the file with name filename is created and the table have either normal text or late format. The precision in the
	 * double quantities is determined by the input argument format.
	 * Example 1:
	 * You want print a table with 4 rows and 3 columns where the last column must be printed as integer and the
	 * first two columns as doubles. The first table must be in scientific notation with 3 decimal points and the second
	 * in normal representation with 2 decimal points. The elements must be separated by tab and you want normal text format.
	 * Then you call:
	 * save_table("myfilname.txt", table, 4, 3, "%.3e%.4f%d", "\t", false);
	 *
	 * Example 2:
	 * You want print a 5x10 table where all the columns are float points with 3 decimals and only the first column must
	 * be in integer format and you want it to be in latex format. You can do:
	 * save_table("myfilname.txt", table, 5, 10, "%d%.3f%.3f%.3f%.3f%.3f%.3f%.3f%.3f%.3f", " ", true);
	 */

	//Sanity Check
	FILE* filep = std::fopen(filename, "w");
	if(filep == NULL){
		printf("\n\nProblem opening the file. Leaving function.\n\n");
		return;
	}

	//Converting the input array of char to string
	std::string filename_string = std::string(filename);
	//Converting the input array of char to string
	std::string format_string = std::string(format);
	//Converting the input array of char to string
	std::string tempformat_string = std::string();
	//Auxiliary variables for printing
	char tempformat[7];
	int pos = 0;


	//Part for printing in text format
	for(unsigned i = 0; i < size1; i++){
		//Resetting pos variable
		pos = 0;
		for(unsigned j = 0; j < size2; j++){
			int fpos = format_string.find("%", pos+1);
			//Checking if it is the last element in the column
			if(fpos < 0){
				//Copying the content in the format string to the temporary string to print
				format_string.copy(tempformat, format_string.length()-pos, pos);
				//Setting the end of the string for this particular element format
				tempformat[format_string.length()-pos+1] = '\0';
				//Print the element that is the last one in the column
				fprintf(filep, tempformat, table[i][j]);
			}else{
				//Copying the content in the format string to the temporary string to print
				tempformat_string = format_string.substr(pos, fpos-pos)+"\t";
				tempformat_string.copy(tempformat, tempformat_string.length(), 0);
				tempformat[tempformat_string.length()] = '\0';
				//Update the position for the next element to be print
				pos = fpos;
				//Print the element
				fprintf(filep, tempformat, table[i][j]);
			}
		}
		fprintf(filep, "\n");
	}

	//Closing the file opened for writing the table content
	fclose(filep);
}

void square_trid(gsl_spmatrix const* m, gsl_spmatrix** mm){
	/*Input:
	 * m is a pointer for gsl_spmatrix
	 * mm is a pointer for gsl_spmatrix that will be in GSL_SPMATRIX_CCS format
	 */
	/*Output
	 * mm is the output and represents the matrix square of the input argument m
	 */
	/*Requirement
	 * m must be a square tridiagonal matrix in order to the output be correct
	 */
	/*Description:
	 * This function returns the matrix square of the input argument (m) in the memory location pointed by mm. The memory is allocated by
	 * the function and must be deallocated by the calling function. The output matrix is converted to column compressed format.
	 */

	//Sanity Check
	if(m->size1 != m->size2){
		printf("\n\nError: the input matrix must be square.\n\n");
		return;
	}

	unsigned msize = m->size1;
	//Allocating memory for the output matrix
	gsl_spmatrix* tempmm = gsl_spmatrix_alloc_nzmax(msize, msize, 5*msize-4, GSL_SPMATRIX_TRIPLET);

	//Sanity Check
	if(tempmm == NULL){
		printf("\n\nError: memory for the output matrix could not be allocated.\n\n");
		return;
	}

	//Compute the elements of row 0
	double mul1 = gsl_spmatrix_get(m, 1,0)*gsl_spmatrix_get(m, 0,1);
	double mul2 = gsl_spmatrix_get(m, 2,1)*gsl_spmatrix_get(m, 1,2);
	double add1 = gsl_spmatrix_get(m, 0,0)+gsl_spmatrix_get(m, 1,1);
	double add2 = gsl_spmatrix_get(m, 1,1)+gsl_spmatrix_get(m, 2,2);

	//Compute the elements of row 1
	gsl_spmatrix_set(tempmm, 0, 0, pow(gsl_spmatrix_get(m, 1,1),2)+mul1);
	gsl_spmatrix_set(tempmm, 0, 1, add1*gsl_spmatrix_get(m, 1,2));
	gsl_spmatrix_set(tempmm, 0, 2, gsl_spmatrix_get(m, 0,1)*gsl_spmatrix_get(m, 1,2));

	//Compute the elements of row 2
	gsl_spmatrix_set(tempmm, 1, 0, add1*gsl_spmatrix_get(m, 2,1));
	gsl_spmatrix_set(tempmm, 1, 1, mul1+pow(gsl_spmatrix_get(m, 2,2),2)+mul2);
	gsl_spmatrix_set(tempmm, 1, 2, add2*gsl_spmatrix_get(m, 2,3));
	gsl_spmatrix_set(tempmm, 1, 3, gsl_spmatrix_get(m, 1,2)*gsl_spmatrix_get(m, 2,3));

	for(unsigned i = 2; i < msize-2; i++){
		//Compute the auxiliary variables
		mul1 = mul2;
		mul2 = gsl_spmatrix_get(m, i,i+1)*gsl_spmatrix_get(m, i+1,i);
		add1 = add2;
		add2 = gsl_spmatrix_get(m, i+1,i+1)+gsl_spmatrix_get(m, i,i);

		//Compute the elements of row i
		gsl_spmatrix_set(tempmm, i, i-2, gsl_spmatrix_get(m, i-1,i-2)*gsl_spmatrix_get(m, i,i-1));
		gsl_spmatrix_set(tempmm, i, i+1, add1*gsl_spmatrix_get(m, i,i-1));
		gsl_spmatrix_set(tempmm, i, i, mul1+pow(gsl_spmatrix_get(m, i,i),2)+mul2);
		gsl_spmatrix_set(tempmm, i, i+1, add2*gsl_spmatrix_get(m, i,i+1));
		gsl_spmatrix_set(tempmm, i, i+2, gsl_spmatrix_get(m, i,i+1)*gsl_spmatrix_get(m, i+1,i+2));
	}

	//Compute auxiliary variables
	mul1 = mul2;
	mul2 = gsl_spmatrix_get(m, msize-2,msize-1)*gsl_spmatrix_get(m, msize-1,msize-2);
	add1 = add2;
	add2 = gsl_spmatrix_get(m, msize-2,msize-2)+gsl_spmatrix_get(m, msize-1,msize-1);

	//Compute elements of row msize-2
	gsl_spmatrix_set(tempmm, msize-2, msize-4, gsl_spmatrix_get(m, msize-3,msize-4)*gsl_spmatrix_get(m, msize-2,msize-3));
	gsl_spmatrix_set(tempmm, msize-2, msize-3, add1*gsl_spmatrix_get(m, msize-2,msize-3));
	gsl_spmatrix_set(tempmm, msize-2, msize-2, mul1+pow(gsl_spmatrix_get(m, msize-2,msize-2),2)+mul2);
	gsl_spmatrix_set(tempmm,msize-2, msize-1, add2*gsl_spmatrix_get(m, msize-2,msize-1));

	//Compute elements of row n-1
	gsl_spmatrix_set(tempmm, msize-1, msize-3, gsl_spmatrix_get(m, msize-2,msize-3)*gsl_spmatrix_get(m, msize-1,msize-2));
	gsl_spmatrix_set(tempmm, msize-1, msize-2, add2*gsl_spmatrix_get(m, msize-1,msize-2));
	gsl_spmatrix_set(tempmm, msize-1, msize-1, pow(gsl_spmatrix_get(m, msize-1,msize-1),2)+mul2);

	//Converting the output matrix to
	*mm = gsl_spmatrix_ccs(tempmm);
	gsl_spmatrix_free(tempmm);
}

bool my_gsl_spmatrix_equal(gsl_spmatrix const* m1, gsl_spmatrix const* m2, double prec){
	/*Input:
	 * m1 and m2 are two gsl_spmatrix
	 * prec is a double representing the maximum distance between the elements of m1 and m2
	 */
	/*Output:
	 * flag is a bool type and is true if both matrices are equal and false otherwise
	 */
	/*Requeriment:
	 * m1 and m2 must be matrices with the same size and be in a compressed format, either CCS or CRS.
	 */
	/*Description:
	 * This function checks if the matrices m1 and m2 are equals. The checking is done as follows: if one of the absolute value of the difference of the corresponding
	 * elements is higher than prec for at least one element, the matrices are considered to be different at precision prec.
	 * Otherwise, the matrices are considered equals.
	 */

	//Sanity Check
	if(m1->sptype != m2->sptype){
		printf("\n\nError: the input matrices must be in the same compressed format, either CCS or CRS.\n\n");
		return false;
	}
	if(m1->size1 != m2->size1 || m1->size2 != m2->size2){
		printf("\n\nError: the input matrices must have the same dimensions.\n\n");
		return false;
	}

	//Output variable
	bool flag;
	//Auxiliary variables
	double min, max;
	gsl_spmatrix* ma = gsl_spmatrix_alloc_nzmax(m1->size1, m1->size2, m1->nzmax, m1->sptype);
	gsl_spmatrix* mb = gsl_spmatrix_alloc_nzmax(m1->size1, m1->size2, m1->nzmax, m1->sptype);
	//Copying the values of m2 to ma
	gsl_spmatrix_memcpy(ma, m2);
	//Multiplying ma by -1
	gsl_spmatrix_scale(ma, -1);
	//Adding ma to m1 and saving to mb, which corresponds to making m1-m2
	gsl_spmatrix_add(mb, ma, m1);
	//Getting the maximum and minimum of mb
	gsl_spmatrix_minmax(mb, &min, &max);
	//Checking the condition if the matrix elements are within maximum distance of prec
	fabs(min) >= prec || fabs(max) >= prec?flag=false:flag=true;
	//Freeing memory
	gsl_spmatrix_free(ma);
	gsl_spmatrix_free(mb);

	return flag;
}

//gsl_spmatrix* my_gsl_spmatrix_triplet(const gsl_spmatrix* m){
//	/*Input:
//	 * m is a gsl_spmatrix
//	 */
//	/*Output:
//	 * mout is a gsl_spmatrix in triplet format
//	 */
//	/*Requirement:
//	 * m is a gsl_spmatrix object and must be either CCS or CRS format
//	 */
//	/*Description:
//	 * this function converts a gsl_spmatrix in CCS or CRS format to TRIPLET format. Memory is allocated
//	 * inside this function and must be deallocated by the calling function
//	 */
//
//	gsl_spmatrix* mout = gsl_spmatrix_alloc_nzmax(m->size1, m->size2, m->nz, GSL_SPMATRIX_TRIPLET);
//
//	//Index running all the non-zero elements of m
//	unsigned k = 0;
//
//	if(m->sptype==GSL_SPMATRIX_CCS){
//		for(unsigned j = 0; j < m->size2; j++){
//			unsigned p = m->p[j];
//			for(; p < m->p[j+1]; p++, k++){
//				gsl_spmatrix_set(mout, m->i[k], j, gsl_spmatrix_get(m, m->i[k], j));
//			}
//		}
//	} else if(m->sptype==GSL_SPMATRIX_CRS){
//		for(unsigned i = 0; i < m->size1; i++){
//			unsigned p = m->p[i];
//			for(; p < m->p[i+1]; p++, k++){
//				gsl_spmatrix_set(mout, i, m->i[k], gsl_spmatrix_get(m, i, m->i[k]));
//			}
//		}
//	} else{
//		printf("\n\nError: input matrix must be either in CCS or CRS format.\n");
//		return NULL;
//	}
//	return mout;
//}

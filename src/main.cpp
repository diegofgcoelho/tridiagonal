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
void fill_spmatrix_random(gsl_spmatrix* m, gsl_rng* rng);
int power_method(gsl_spmatrix *m, gsl_vector *v, double* lambda, unsigned maxit, double prec, unsigned* iter);
double max_mag(gsl_vector* v);
bool check_stop(gsl_vector* v, gsl_vector* vv, gsl_vector* vvv, double prec);
void save_table(char const* filename, double** table, unsigned size1, unsigned size2, char const* format, bool latex);
void fast_square_trid_triplet(gsl_spmatrix const* m, gsl_spmatrix** mm);
void fast_square_trid(gsl_spmatrix const* m, gsl_spmatrix** mm);
bool my_gsl_spmatrix_equal(gsl_spmatrix const* m1, gsl_spmatrix const* m2, double prec);
//The following functions are not used anymore, but let only for reference
//void square_trid_ccs(gsl_spmatrix const* m, gsl_spmatrix** mm);
//gsl_spmatrix* my_gsl_spmatrix_triplet(const gsl_spmatrix* m);

int main(){
	//Sparse matrices pointers
	gsl_spmatrix *mcrs, *tempm, *mmcrs, *mccs, *gemmccs, *mmccs;
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
	unsigned const sp_trid_n[] = {500, 1000, 5000, 10000};
	//Storing the number of different matrix sizes we will test (the size of array sp_trid_n)
	unsigned const n_sp_trid_n = 4;

	//Average iterations for estimates
	double iter_array[n_sp_trid_n], miter_array[n_sp_trid_n];
	std::fill_n(&iter_array[0], n_sp_trid_n, 0);
	std::fill_n(&miter_array[0], n_sp_trid_n, 0);
	//Eigenvalue estimates
	double lambdas_array[n_sp_trid_n], mlambdas_array[n_sp_trid_n];
	std::fill_n(&lambdas_array[0], n_sp_trid_n, 0);
	std::fill_n(&mlambdas_array[0], n_sp_trid_n, 0);
	//Average times for estimates
	double times_array[n_sp_trid_n], mtimes_array[n_sp_trid_n];
	std::fill_n(&times_array[0], n_sp_trid_n, 0);
	std::fill_n(&mtimes_array[0], n_sp_trid_n, 0);
	//Total times for squaring (gsl_spmatrix_dgemm, fast_square_trid for CRS and CCS formats, respectively)
	double** times_square = new double*[n_sp_trid_n];
	for(unsigned int i = 0; i < n_sp_trid_n; i++){
		times_square[i] = new double[4];
		std::fill_n(&times_square[i][0], 4, 0);
		//Putting the matrix in the first column
		times_square[i][0] = sp_trid_n[i];
	}

	//The precision used for simulation
	double symprec = 1e-3;

	//Auxiliary variables for counting the time
	clock_t time_beg, time_end;
	clock_t sqr_time_beg, sqr_time_end;
	double gemm_time=0.0, sqr_tri_time=0.0, sqr_crs_time=0.0, sqr_ccs_time=0.0;

	//The number of replicates
	unsigned const rep = 100;

	for(unsigned i = 0; i < n_sp_trid_n; i++){
		//Allocating memory for the tridiagonal matrix. Note: all the elements are set to zero
		mcrs = gsl_spmatrix_alloc_nzmax(sp_trid_n[i], sp_trid_n[i], 3*sp_trid_n[i]-2, GSL_SPMATRIX_TRIPLET);
		//Allocating memory for the square of the tridiagonal matrix. Note: all the elements are set to zero
		//Not needed anymore because of the function square_trid
		//mm = gsl_spmatrix_alloc_nzmax(sp_trid_n[i], sp_trid_n[i], pentnnz, GSL_SPMATRIX_CCS);

		//Fill the matrix entries to be a Kac-Sylvester-Clement matrix
		//fill_spmatrix_kac(mcrs);
		//Fill the matrix entries to be a tridiagonal matrix
		fill_spmatrix_random(mcrs, tausrng);

		//Converting m to CRS format
		tempm = mcrs;
		mcrs = gsl_spmatrix_crs(tempm);
		//Converting m to CCS format
		mccs = gsl_spmatrix_ccs(tempm);

		//Auxiliary variable representing the initial guess for the matrix eigenvector
		gsl_vector* v = gsl_vector_alloc(sp_trid_n[i]);
		gsl_vector* _v = gsl_vector_alloc(sp_trid_n[i]);
		//Initiating all the elements of v
		for(unsigned j = 0; j < _v->size; j++){
			gsl_vector_set(_v, j, gsl_rng_uniform(tausrng));
		}
		//Alternate initialization for testing and comparison with matlab/octave
		//gsl_vector_set_all(v, 1.0);

		//Run over all replicates for matrix size sp_tridi_n[i]
		for (unsigned j = 0; j < rep; ++j) {
			//Auxiliary variables used for getting the number of iterations and the eigenvalue estimate
			double lambda = 0.0;
			unsigned iter = 0;

			//Setting the initial values of v that are stored in _V
			gsl_vector_memcpy(v, _v);

			/*Calling power_method function that returns the estimate for the largest eigenvalue
			 * of a matrix
			 */
			time_beg = clock();
			power_method(mcrs, v, &lambda, MAXITE, symprec, &iter);
			time_end = clock();
			times_array[i]+=1000*(time_end-time_beg)/(double)CLOCKS_PER_SEC;
			iter_array[i]+=iter;
			lambdas_array[i]+=fabs((lambda-(sp_trid_n[i]-1))/(sp_trid_n[i]-1));

			//Resetting all the elements of v to the same initial guess as for usual power method.
			gsl_vector_memcpy(v, _v);

			/*Calling power_method function with the matrix square that returns the estimate for the largest eigenvalue
			 * of a matrix
			 */

			gsl_spmatrix* testmm = NULL;
			sqr_time_beg = clock();
			fast_square_trid_triplet(mcrs, &testmm);
			sqr_time_end = clock();
			sqr_tri_time+=1000*(sqr_time_end-sqr_time_beg)/(double)CLOCKS_PER_SEC;

			sqr_time_beg = clock();
			gemmccs = gsl_spmatrix_alloc_nzmax(sp_trid_n[i], sp_trid_n[i], 5*sp_trid_n[i]-4, GSL_SPMATRIX_CCS);
			gsl_spblas_dgemm(1, mccs, mccs, gemmccs);
			sqr_time_end = clock();
			gemm_time+=1000*(sqr_time_end-sqr_time_beg)/(double)CLOCKS_PER_SEC;
			times_square[i][1]+=1000*(sqr_time_end-sqr_time_beg)/(double)CLOCKS_PER_SEC;

			time_beg = clock();
			sqr_time_beg = clock();
			fast_square_trid(mcrs, &mmcrs);
			sqr_time_end = clock();
			sqr_crs_time+=1000*(sqr_time_end-sqr_time_beg)/(double)CLOCKS_PER_SEC;
			times_square[i][2]+=1000*(sqr_time_end-sqr_time_beg)/(double)CLOCKS_PER_SEC;
			power_method(mmcrs, v, &lambda, MAXITE, symprec, &iter);
			time_end = clock();

			sqr_time_beg = clock();
			fast_square_trid(mccs, &mmccs);
			sqr_time_end = clock();
			sqr_ccs_time+=1000*(sqr_time_end-sqr_time_beg)/(double)CLOCKS_PER_SEC;
			times_square[i][3]+=1000*(sqr_time_end-sqr_time_beg)/(double)CLOCKS_PER_SEC;

			if(!my_gsl_spmatrix_equal(mmccs, gemmccs, 1e-10)){
				printf("The matrices are not equal.\n");
				gsl_spmatrix_scale(mmccs, -1);
				gsl_spmatrix_add(mmccs, mmccs, gemmccs);
				gsl_spmatrix_scale(mmccs, 1e6);
				gsl_spmatrix_fprintf(stdout, mmccs, "%.02f");
				exit(1);
			}
			gsl_spmatrix_free(gemmccs);

			gsl_spmatrix_free(testmm);

			//Deallocating the memory used to store the square matrix
			gsl_spmatrix_free(mmcrs);
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
		gsl_spmatrix_free(mcrs);
		gsl_spmatrix_free(tempm);
		gsl_vector_free(v);
		gsl_vector_free(_v);
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

	printf("\n\n*****The statistics for squaring algorithm*****\n\n");
	printf("Size Time (dgemm)  Time (CRS)  Time (CCS)\n");
	for(unsigned j=0; j<n_sp_trid_n; j++){
		printf("\t%.0f\t%.3f\t%.3f\t%.3f\n\n", times_square[j][0], times_square[j][1], times_square[j][2], times_square[j][3]);
	}

	printf("Time required by gsl_spmatrix_dgemm:  %.2f\n", gemm_time);
	printf("Time required by fast_square_trid_triplet: %.2f\n", sqr_tri_time);
	printf("Times required by fast_square_trid for CCS and CRS formats: %.2f and %.2f\n\n", sqr_ccs_time, sqr_crs_time);

	//Forming table to be printed through save_table function based on the measures for the usual Power method
	double** measures_table = new double*[n_sp_trid_n];
	for(unsigned i = 0; i < n_sp_trid_n; i++){
		measures_table[i] = new double [4];
		//Setting the content of each row of the table to be print
		measures_table[i][0] = sp_trid_n[i];
		measures_table[i][1] = iter_array[i];
		measures_table[i][2] = lambdas_array[i];
		measures_table[i][3] = times_array[i];
	}

	//File name
	char filename_times[] = "measures_times.txt";
	//Printing the table times_square
	save_table(filename_times, times_square, n_sp_trid_n, 4, "%.0f%.3f%.3f%.3f", true);

	//File name
	char filename_usual[] = "measures_usual.txt";
	//Printing the table measures_table
	save_table(filename_usual, measures_table, n_sp_trid_n, 4, "%.0f%.0f%.3e%.3f", true);

	//Forming table to be printed through save_table function based on the measures for the modified Power method
	for(unsigned i = 0; i < n_sp_trid_n; i++){
		//Setting the content of each row of the table to be print
		measures_table[i][0] = sp_trid_n[i];
		measures_table[i][1] = miter_array[i];
		measures_table[i][2] = mlambdas_array[i];
		measures_table[i][3] = mtimes_array[i];
	}
	//File name
	char filename_modified[] = "measures_modified.txt";

	//Printing the table measures_table
	save_table(filename_modified, measures_table, n_sp_trid_n, 4, "%.0f%.0f%.3e%.3f", true);

	//Freeing all the allocated memory for printing the table
	for(unsigned i = 0; i < n_sp_trid_n; i++){
		delete [] measures_table[i];
	}
	delete [] measures_table;

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

void fill_spmatrix_random(gsl_spmatrix* m, gsl_rng* rng){
	/*Input:
	 * m is a gls_spmatrix
	 * rng is a gsl_rng
	 */
	/*Output:
	 * m is a gsl_spmatrix representing a tridiagonal matrix
	 */
	/*Requirement:
	 * m is in TRIPLET format, have null entries and represent a tridiagonal matrix
	 */
	/*Description:
	 * This function modifies the entries of m making it become a unsymmetric tridiagonal matrix with real entries whose values
	 * taken from the random number generator pointed by rng
	 */
	//Number of rows of the matrix
	unsigned nrows = m->size1;

	//Multiplicative constant
	unsigned const mulc = 100;

	//Setting the first row elements
	gsl_spmatrix_set(m, 0, 0, mulc*gsl_rng_uniform(rng));
	gsl_spmatrix_set(m, 0, 1, mulc*gsl_rng_uniform(rng));

	//Setting the elements between the first and last line
	for(unsigned i = 1; i < nrows-1; i++){
		//Setting the element at the left of the diagonal element
		gsl_spmatrix_set(m, i, i-1, mulc*gsl_rng_uniform(rng));
		//Setting the diagonal element
		gsl_spmatrix_set(m, i, i, mulc*gsl_rng_uniform(rng));
		//Setting the element at the right of the diagonal element
		gsl_spmatrix_set(m, i, i+1, mulc*gsl_rng_uniform(rng));
	}

	//Setting the last row elements
	gsl_spmatrix_set(m, nrows-1, nrows-2, mulc*gsl_rng_uniform(rng));
	gsl_spmatrix_set(m, nrows-1, nrows-1, mulc*gsl_rng_uniform(rng));
}

int power_method(gsl_spmatrix *m, gsl_vector *v, double* lambda, unsigned maxit, double prec, unsigned* iter){
	/*Input:
	 * m is a gsl_spmatrix
	 * v is a gsl_vector representing the initial guess for the eigenvector associated with the
	 * largest eigenvalue
	 * maxit is the maximum number of iterations
	 * prec is the precision using for the stopping criteria for the vector difference norm
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
	gsl_vector* vv = gsl_vector_alloc(v->size); gsl_vector_set_all(vv, 0.0);
	gsl_vector* vvv = gsl_vector_alloc(v->size); gsl_vector_set_all(vvv, 0.0);

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
	 * separator is the type of separation between the table entries to be printed. If you are using latex format,
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

	if (latex==false) {
		printf("\n\nPriting data in text format in file %s.\n\n", filename);
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
	} else {
		printf("\n\nPriting data in latex format in file %s.\n\n", filename);
		//Printing in the latex format
		fprintf(filep,"%% add the booktabs package in the main latex file\n");
		fprintf(filep,"\\begin{table}\n");
		fprintf(filep,"\\begin{center}\n");
		fprintf(filep,"\\caption{write your caption here}\n");
		fprintf(filep,"\\label{write your label here}\n");
		fprintf(filep,"\\begin{tabular}{\n");


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
						fprintf(filep, "$");
						fprintf(filep, tempformat, table[i][j]);
						fprintf(filep, "$");
					}else{
						//Copying the content in the format string to the temporary string to print
						tempformat_string = format_string.substr(pos, fpos-pos)+"\t";
						tempformat_string.copy(tempformat, tempformat_string.length(), 0);
						tempformat[tempformat_string.length()] = '\0';
						//Update the position for the next element to be print
						pos = fpos;
						//Print the element
						fprintf(filep, "$");
						fprintf(filep, tempformat, table[i][j]);
						fprintf(filep, "$ &");
					}
				}
				fprintf(filep, "\\\\\\midrule\n");
			}

		fprintf(filep,"\\end{tabular}\n");
		fprintf(filep,"\\end{center}\n");
		fprintf(filep,"\\end{table}\n");
	}
	//Closing the file opened for writing the table content
	fclose(filep);
}

void fast_square_trid_triplet(gsl_spmatrix const* m, gsl_spmatrix** mm){
	/*Note: this function execution is slower than the using the gsl_spmatrix_gemm with the same argument.
	 * This is because of the access to the elements that is using the gsl_spmatrix_get, which is very slow.
	 * The function works, but it is slow. The suare_trid_fast is being written in order to solve this problem.
	 */

	/*Input:
	 * m is a pointer for gsl_spmatrix
	 * mm is a pointer for gsl_spmatrix that will be in GSL_SPMATRIX_CRS format
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

	//Compute the auxiliary variables
	double mul1 = gsl_spmatrix_get(m, 1,0)*gsl_spmatrix_get(m, 0,1);
	double mul2 = gsl_spmatrix_get(m, 2,1)*gsl_spmatrix_get(m, 1,2);
	double add1 = gsl_spmatrix_get(m, 0,0)+gsl_spmatrix_get(m, 1,1);
	double add2 = gsl_spmatrix_get(m, 1,1)+gsl_spmatrix_get(m, 2,2);

	//Compute the elements of row 0
	gsl_spmatrix_set(tempmm, 0, 0, pow(gsl_spmatrix_get(m, 0,0),2)+mul1);
	gsl_spmatrix_set(tempmm, 0, 1, add1*gsl_spmatrix_get(m, 0, 1));
	gsl_spmatrix_set(tempmm, 0, 2, gsl_spmatrix_get(m, 0,1)*gsl_spmatrix_get(m, 1,2));

	//Compute the elements of row 1
	gsl_spmatrix_set(tempmm, 1, 0, add1*gsl_spmatrix_get(m, 1,0));
	gsl_spmatrix_set(tempmm, 1, 1, mul1+pow(gsl_spmatrix_get(m, 1,1),2)+mul2);
	gsl_spmatrix_set(tempmm, 1, 2, add2*gsl_spmatrix_get(m, 1,2));
	gsl_spmatrix_set(tempmm, 1, 3, gsl_spmatrix_get(m, 1,2)*gsl_spmatrix_get(m, 2,3));

	for(unsigned i = 2; i < msize-2; i++){
		//Compute the auxiliary variables
		mul1 = mul2;
		mul2 = gsl_spmatrix_get(m, i,i+1)*gsl_spmatrix_get(m, i+1,i);
		add1 = add2;
		add2 = gsl_spmatrix_get(m, i+1,i+1)+gsl_spmatrix_get(m, i,i);

		//Compute the elements of row i
		gsl_spmatrix_set(tempmm, i, i-2, gsl_spmatrix_get(m, i-1,i-2)*gsl_spmatrix_get(m, i,i-1));
		gsl_spmatrix_set(tempmm, i, i-1, add1*gsl_spmatrix_get(m, i,i-1));
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

void fast_square_trid(gsl_spmatrix const* m, gsl_spmatrix** mm){
	/*Input:
	 * m is a pointer for gsl_spmatrix
	 * mm is a pointer for gsl_spmatrix that will be in the same format as m
	 */
	/*Output
	 * mm is the output and represents the matrix square of the input argument m
	 */
	/*Requirement
	 * m must be a square tridiagonal matrix in compressed format, either CCS or CRS
	 */
	/*Description:
	 * This function returns the matrix square of the input argument (m) in the memory location pointed by mm. The memory is allocated by
	 * the function and must be deallocated by the calling function.
	 */

	//Note: possible improvement is to remove the update of mm->nz every time a new element is added and just update it at the end.

	//Sanity Check
	if(m->size1 != m->size2){
		printf("\n\nError: the input matrix must be square.\n\n");
		return;
	}

	unsigned msize = m->size1;
	//Allocating memory for the output matrix
	(*mm) = gsl_spmatrix_alloc_nzmax(msize, msize, 5*msize-4, m->sptype);

	//Sanity Check
	if((*mm) == NULL){
		printf("\n\nError: memory for the output matrix could not be allocated.\n\n");
		return;
	}

	size_t *mi = m->i;
	size_t *mp = m->p;
	double *md = m->data;

	//Cleaning the entries of *mm->p and setting *mm->nz to zero
	std::fill_n(&((*mm)->p[0]), msize+1, 0);
	(*mm)->nz = 0;

	//Doubles that will always use for storing the elements of each line.
	double  data[3][3] = { {0.0} };

	//Value used for computing each element
	double tempmmv = 0.0;

	//Get the elements of the first 3 rows
	for(unsigned i = 0; i < 2; i++){
		//mp[mi[i]] indicate the row
		//mi[i] indicate the column

		for(unsigned j = mp[i]; j < mp[i+1]; j++){
			data[i][mi[j]] = md[j];
		}
	}
	for(unsigned j = mp[2]; j < mp[3]; j++){
		data[2][mi[j]-1] = md[j];
	}

	//Compute the elements of row 0. Note the corrections in accessing all the elements at row 2
	double mul1 = data[1][0]*data[0][1];
	double mul2 = data[2][0]*data[1][2];
	double add1 = data[0][0]+data[1][1];
	double add2 = data[1][1]+data[2][1];

	//Compute the elements of row 0
	tempmmv = pow(data[0][0], 2)+mul1;
	if(tempmmv != 0.0){
		(*mm)->data[0] = tempmmv;
		(*mm)->i[0] = 0;
		(*mm)->p[1]++;
		(*mm)->nz++;
	}
	tempmmv = add1*data[0][1];
	if(tempmmv != 0.0){
		(*mm)->data[(*mm)->nz] = tempmmv;
		(*mm)->i[(*mm)->nz] = 1;
		(*mm)->p[1]++;
		(*mm)->nz++;
	}
	tempmmv = data[0][1]*data[1][2];
	if(tempmmv != 0.0){
		(*mm)->data[(*mm)->nz] = tempmmv;
		(*mm)->i[(*mm)->nz] = 2;
		(*mm)->p[1]++;
		((*mm)->nz)++;
	}
	//Update the value of the next component of p
	(*mm)->p[2] = (*mm)->p[1];

	//Compute the elements of row 1
	tempmmv = add1*data[1][0];
	if(tempmmv != 0.0){
		(*mm)->data[(*mm)->nz] = tempmmv;
		(*mm)->i[(*mm)->nz] = 0;
		(*mm)->p[2]++;
		((*mm)->nz)++;
	}
	tempmmv = mul1+pow(data[1][1],2)+mul2;
	if(tempmmv != 0.0){
		(*mm)->data[(*mm)->nz] = tempmmv;
		(*mm)->i[(*mm)->nz] = 1;
		(*mm)->p[2]++;
		((*mm)->nz)++;
	}
	tempmmv = add2*data[1][2];
	if(tempmmv != 0.0){
		(*mm)->data[(*mm)->nz] = tempmmv;
		(*mm)->i[(*mm)->nz] = 2;
		(*mm)->p[2]++;
		((*mm)->nz)++;
	}
	tempmmv = data[1][2]*data[2][2];
	if(tempmmv != 0.0){
		(*mm)->data[(*mm)->nz] = tempmmv;
		(*mm)->i[(*mm)->nz] = 3;
		(*mm)->p[2]++;
		(*mm)->nz++;
	}
	//Update the value of the next component of p
	(*mm)->p[3] = (*mm)->p[2];

	for (unsigned i = 2; i < msize-2; i++) {
		//Update the elements of the temporary matrix data
		//Shifting the data already preset in the matrix data
		for(unsigned p = 0; p < 2; p++){
			for(unsigned k = 0; k < 3; k++){
				data[p][k] = data[p+1][k];
			}
		}
		//Set to zero the elements in the last row of data
		data[2][0] = 0.0; data[2][1] = 0.0; data[2][0] = 0.0;
		//Update the elements in the last row of data, which will contain the elements in the i+1 row in in the input matrix m
		for(unsigned j = mp[i+1]; j < mp[i+2]; j++){
			//Always write in the last row. note the correction in the column position for the matrix data
			data[2][mi[j]-i] = md[j];
		}

		//Compute auxiliary variables
		mul1 = mul2;
		mul2 = data[1][2]*data[2][0];
		add1 = add2;
		add2 = data[2][1]+data[1][1];

		//Compute the elements of row i
		tempmmv = data[0][0]*data[1][0];
		if(tempmmv != 0.0){
			(*mm)->data[(*mm)->nz] = tempmmv;
			(*mm)->i[(*mm)->nz] = i-2;
			(*mm)->p[i+1]++;
			(*mm)->nz++;
		}
		tempmmv = add1*data[1][0];
		if(tempmmv != 0.0){
			(*mm)->data[(*mm)->nz] = tempmmv;
			(*mm)->i[(*mm)->nz] = i-1;
			(*mm)->p[i+1]++;
			(*mm)->nz++;
		}
		tempmmv = mul1+pow(data[1][1], 2)+mul2;
		if(tempmmv != 0.0){
			(*mm)->data[(*mm)->nz] = tempmmv;
			(*mm)->i[(*mm)->nz] = i;
			(*mm)->p[i+1]++;
			(*mm)->nz++;
		}
		tempmmv = add2*data[1][2];
		if(tempmmv != 0.0){
			(*mm)->data[(*mm)->nz] = tempmmv;
			(*mm)->i[(*mm)->nz] = i+1;
			(*mm)->p[i+1]++;
			(*mm)->nz++;
		}
		tempmmv = data[1][2]*data[2][2];
		if(tempmmv != 0.0){
			(*mm)->data[(*mm)->nz] = tempmmv;
			(*mm)->i[(*mm)->nz] = i+2;
			(*mm)->p[i+1]++;
			(*mm)->nz++;
		}
		//Update the value of the next component of p
		(*mm)->p[i+2] = (*mm)->p[i+1];

	}

	//Update the elements of the temporary matrix data
	//Shifting the data already preset in the matrix data
	for(unsigned p = 0; p < 2; p++){
		for(unsigned k = 0; k < 3; k++){
			data[p][k] = data[p+1][k];
		}
	}
	//Set to zero the elements in the last row of data
	data[2][0] = 0.0; data[2][1] = 0.0; data[2][2] = 0.0;
	//Update the elements in the last row of data, which will contain the elements in the i+1 row in in the input matrix m
	for(unsigned j = mp[msize-1]; j < mp[msize]; j++){
		//Always write in the last row. note the correction in the column position for the matrix data
		data[2][mi[j]-msize+3] = md[j];
	}

	//Compute auxiliary variable
	mul1 = mul2;
	mul2 = data[1][2]*data[2][1];
	add1 = add2;
	add2 = data[1][1]+data[2][2];

	//Compute the elements of row msize-2
	tempmmv = data[0][0]*data[1][0];
	if(tempmmv != 0.0){
		(*mm)->data[(*mm)->nz] = tempmmv;
		(*mm)->i[(*mm)->nz] = msize-4;
		(*mm)->p[msize-1]++;
		(*mm)->nz++;
	}
	tempmmv = add1*data[1][0];
	if(tempmmv != 0.0){
		(*mm)->data[(*mm)->nz] = tempmmv;
		(*mm)->i[(*mm)->nz] = msize-3;
		(*mm)->p[msize-1]++;
		(*mm)->nz++;
	}
	tempmmv = mul1+pow(data[1][1], 2)+mul2;
	if(tempmmv != 0.0){
		(*mm)->data[(*mm)->nz] = tempmmv;
		(*mm)->i[(*mm)->nz] = msize-2;
		(*mm)->p[msize-1]++;
		(*mm)->nz++;
	}
	tempmmv = add2*data[1][2];
	if(tempmmv != 0.0){
		(*mm)->data[(*mm)->nz] = tempmmv;
		(*mm)->i[(*mm)->nz] = msize-1;
		(*mm)->p[msize-1]++;
		(*mm)->nz++;
	}
	//Update the value of the next component of p
	(*mm)->p[msize] = (*mm)->p[msize-1];


	//Compute elements of row msize-1
	tempmmv = data[1][0]*data[2][1];
	if(tempmmv != 0.0){
		(*mm)->data[(*mm)->nz] = tempmmv;
		(*mm)->i[(*mm)->nz] = msize-3;
		(*mm)->p[msize]++;
		(*mm)->nz++;
	}
	tempmmv = add2*data[2][1];
	if(tempmmv != 0.0){
		(*mm)->data[(*mm)->nz] = tempmmv;
		(*mm)->i[(*mm)->nz] = msize-2;
		(*mm)->p[msize]++;
		(*mm)->nz++;
	}
	tempmmv = pow(data[2][2],2)+mul2;
	if(tempmmv != 0.0){
		(*mm)->data[(*mm)->nz] = tempmmv;
		(*mm)->i[(*mm)->nz] = msize-1;
		(*mm)->p[msize]++;
		(*mm)->nz++;
	}
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

//void square_trid_ccs(gsl_spmatrix const* m, gsl_spmatrix** mm){
//	/*Input:
//	 * m is a pointer for gsl_spmatrixL_SPMATRIX_CCS format
//	 * mm is a pointer for gsl_spmatrix that will be in GSL_SPMATRIX_CCS format
//	 */
//	/*Output
//	 * mm is the output and represents the matrix square of the input argument m
//	 */
//	/*Requirement
//	 * m must be a square tridiagonal matrix in GSL_SPMATRIX_CCS format in order to the output be correct
//	 */
//	/*Description:
//	 * This function returns the matrix square of the input argument (m) in the memory location pointed by mm. The memory is allocated by
//	 * the function and must be deallocated by the calling function.
//	 */
//
//	//Note: possible improvement is to remove the update of mm->nz every time a new element is added and just update it at the end.
//
//	//Sanity Check
//	if(m->size1 != m->size2){
//		printf("\n\nError: the input matrix must be square.\n\n");
//		return;
//	}
//
//	unsigned msize = m->size1;
//	//Allocating memory for the output matrix
//	(*mm) = gsl_spmatrix_alloc_nzmax(msize, msize, 5*msize-4, GSL_SPMATRIX_CCS);
//
//	//Sanity Check
//	if((*mm) == NULL){
//		printf("\n\nError: memory for the output matrix could not be allocated.\n\n");
//		return;
//	}
//
//	size_t *mi = m->i;
//	size_t *mp = m->p;
//	double *md = m->data;
//
//	//Cleaning the entries of *mm->p and setting *mm->nz to zero
//	std::fill_n(&((*mm)->p[0]), msize+1, 0);
//	(*mm)->nz = 0;
//
//	//Doubles that will always use for storing the elements of each line.
//	double  data[3][3] = { {0.0} };
//
//	//Value used for computing each element
//	double tempmmv = 0.0;
//
//	//Get the elements of the first 3 columns
//	for(unsigned i = 0; i < 2; i++){
//		//mi[i] indicate the row
//
//		for(unsigned j = mp[i]; j < mp[i+1]; j++){
//			data[mi[j]][i] = md[j];
//		}
//	}
//	for(unsigned j = mp[2]; j < mp[3]; j++){
//		data[mi[j]-1][2] = md[j];
//	}
//
//	//Compute the elements of column 0. Note the corrections in accessing all the elements at row 2
//	double mul1 = data[1][0]*data[0][1];
//	double mul2 = data[0][2]*data[2][1];
//	double add1 = data[0][0]+data[1][1];
//	double add2 = data[1][1]+data[1][2];
//
//	//Compute the elements of column 0
//	tempmmv = pow(data[0][0], 2)+mul1;
//	if(tempmmv != 0.0){
//		(*mm)->data[0] = tempmmv;
//		(*mm)->i[0] = 0;
//		(*mm)->p[1]++;
//		(*mm)->nz++;
//	}
//	tempmmv = add1*data[1][0];
//	if(tempmmv != 0.0){
//		(*mm)->data[(*mm)->nz] = tempmmv;
//		(*mm)->i[(*mm)->nz] = 1;
//		(*mm)->p[1]++;
//		(*mm)->nz++;
//	}
//	tempmmv = data[1][0]*data[2][1];
//	if(tempmmv != 0.0){
//		(*mm)->data[(*mm)->nz] = tempmmv;
//		(*mm)->i[(*mm)->nz] = 2;
//		(*mm)->p[1]++;
//		((*mm)->nz)++;
//	}
//	//Update the value of the next component of p
//	(*mm)->p[2] = (*mm)->p[1];
//
//	//Compute the elements of column 1
//	tempmmv = add1*data[0][1];
//	if(tempmmv != 0.0){
//		(*mm)->data[(*mm)->nz] = tempmmv;
//		(*mm)->i[(*mm)->nz] = 0;
//		(*mm)->p[2]++;
//		((*mm)->nz)++;
//	}
//	tempmmv = mul1+pow(data[1][1],2)+mul2;
//	if(tempmmv != 0.0){
//		(*mm)->data[(*mm)->nz] = tempmmv;
//		(*mm)->i[(*mm)->nz] = 1;
//		(*mm)->p[2]++;
//		((*mm)->nz)++;
//	}
//	tempmmv = add2*data[2][1];
//	if(tempmmv != 0.0){
//		(*mm)->data[(*mm)->nz] = tempmmv;
//		(*mm)->i[(*mm)->nz] = 2;
//		(*mm)->p[2]++;
//		((*mm)->nz)++;
//	}
//	tempmmv = data[2][1]*data[2][2];
//	if(tempmmv != 0.0){
//		(*mm)->data[(*mm)->nz] = tempmmv;
//		(*mm)->i[(*mm)->nz] = 3;
//		(*mm)->p[2]++;
//		(*mm)->nz++;
//	}
//	//Update the value of the next component of p
//	(*mm)->p[3] = (*mm)->p[2];
//
//	for (unsigned i = 2; i < msize-2; i++) {
//		//Update the elements of the temporary matrix data
//		//Shifting the data already preset in the matrix data
//		for(unsigned p = 0; p < 3; p++){
//			for(unsigned k = 0; k < 2; k++){
//				data[p][k] = data[p][k+1];
//			}
//		}
//		//Set to zero the elements in the last column of data
//		data[0][2] = 0.0; data[1][2] = 0.0; data[2][2] = 0.0;
//		//Update the elements in the last column of data, which will contain the elements in the i+1 column in the input matrix m
//		for(unsigned j = mp[i+1]; j < mp[i+2]; j++){
//			//Always write in the last column. Note the correction in the row position for the matrix data
//			data[mi[j]-i][2] = md[j];
//		}
//
//		//Compute auxiliary variables
//		mul1 = mul2;
//		mul2 = data[2][1]*data[0][2];
//		add1 = add2;
//		add2 = data[1][2]+data[1][1];
//
//		//Compute the elements of column i
//		tempmmv = data[0][0]*data[0][1];
//		if(tempmmv != 0.0){
//			(*mm)->data[(*mm)->nz] = tempmmv;
//			(*mm)->i[(*mm)->nz] = i-2;
//			(*mm)->p[i+1]++;
//			(*mm)->nz++;
//		}
//		tempmmv = add1*data[0][1];
//		if(tempmmv != 0.0){
//			(*mm)->data[(*mm)->nz] = tempmmv;
//			(*mm)->i[(*mm)->nz] = i-1;
//			(*mm)->p[i+1]++;
//			(*mm)->nz++;
//		}
//		tempmmv = mul1+pow(data[1][1], 2)+mul2;
//		if(tempmmv != 0.0){
//			(*mm)->data[(*mm)->nz] = tempmmv;
//			(*mm)->i[(*mm)->nz] = i;
//			(*mm)->p[i+1]++;
//			(*mm)->nz++;
//		}
//		tempmmv = add2*data[2][1];
//		if(tempmmv != 0.0){
//			(*mm)->data[(*mm)->nz] = tempmmv;
//			(*mm)->i[(*mm)->nz] = i+1;
//			(*mm)->p[i+1]++;
//			(*mm)->nz++;
//		}
//		tempmmv = data[2][1]*data[2][2];
//		if(tempmmv != 0.0){
//			(*mm)->data[(*mm)->nz] = tempmmv;
//			(*mm)->i[(*mm)->nz] = i+2;
//			(*mm)->p[i+1]++;
//			(*mm)->nz++;
//		}
//		//Update the value of the next component of p
//		(*mm)->p[i+2] = (*mm)->p[i+1];
//
//	}
//
//	//Update the elements of the temporary matrix data
//	//Shifting the data already preset in the matrix data
//	for(unsigned p = 0; p < 3; p++){
//		for(unsigned k = 0; k < 2; k++){
//			data[p][k] = data[p][k+1];
//		}
//	}
//	//Set to zero the elements in the last column of data
//	data[0][2] = 0.0; data[1][2] = 0.0; data[2][2] = 0.0;
//	//Update the elements in the last column of data, which will contain the elements in the i+1 column in the input matrix m
//	for(unsigned j = mp[msize-1]; j < mp[msize]; j++){
//		//Always write in the last column. Note the correction in the row position for the matrix data
//		data[mi[j]-msize+3][2] = md[j];
//	}
//
//	//Compute auxiliary variable
//	mul1 = mul2;
//	mul2 = data[1][2]*data[2][1];
//	add1 = add2;
//	add2 = data[1][1]+data[2][2];
//
//	//Compute the elements of column msize-2
//	tempmmv = data[0][0]*data[0][1];
//	if(tempmmv != 0.0){
//		(*mm)->data[(*mm)->nz] = tempmmv;
//		(*mm)->i[(*mm)->nz] = msize-4;
//		(*mm)->p[msize-1]++;
//		(*mm)->nz++;
//	}
//	tempmmv = add1*data[0][1];
//	if(tempmmv != 0.0){
//		(*mm)->data[(*mm)->nz] = tempmmv;
//		(*mm)->i[(*mm)->nz] = msize-3;
//		(*mm)->p[msize-1]++;
//		(*mm)->nz++;
//	}
//	tempmmv = mul1+pow(data[1][1], 2)+mul2;
//	if(tempmmv != 0.0){
//		(*mm)->data[(*mm)->nz] = tempmmv;
//		(*mm)->i[(*mm)->nz] = msize-2;
//		(*mm)->p[msize-1]++;
//		(*mm)->nz++;
//	}
//	tempmmv = add2*data[2][1];
//	if(tempmmv != 0.0){
//		(*mm)->data[(*mm)->nz] = tempmmv;
//		(*mm)->i[(*mm)->nz] = msize-1;
//		(*mm)->p[msize-1]++;
//		(*mm)->nz++;
//	}
//	//Update the value of the next component of p
//	(*mm)->p[msize] = (*mm)->p[msize-1];
//
//
//	//Compute elements of column msize-1
//	tempmmv = data[0][1]*data[1][2];
//	if(tempmmv != 0.0){
//		(*mm)->data[(*mm)->nz] = tempmmv;
//		(*mm)->i[(*mm)->nz] = msize-3;
//		(*mm)->p[msize]++;
//		(*mm)->nz++;
//	}
//	tempmmv = add2*data[1][2];
//	if(tempmmv != 0.0){
//		(*mm)->data[(*mm)->nz] = tempmmv;
//		(*mm)->i[(*mm)->nz] = msize-2;
//		(*mm)->p[msize]++;
//		(*mm)->nz++;
//	}
//	tempmmv = pow(data[2][2],2)+mul2;
//	if(tempmmv != 0.0){
//		(*mm)->data[(*mm)->nz] = tempmmv;
//		(*mm)->i[(*mm)->nz] = msize-1;
//		(*mm)->p[msize]++;
//		(*mm)->nz++;
//	}
//}

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

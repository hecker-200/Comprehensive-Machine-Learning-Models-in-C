#ifndef ML
#define ML
#include "b23cs1016_b23cm1022_b23ci1016_b23mt1004_models.c"

//----------------------------------------LOGISTIC REGRESSION------------------------------------------------------
float *initialize_weights(int m);
float sigmoid(float z);
float *logistic_regression_fit(float **data, float *targets, int rows, int cols, float learning_rate, int iterations);
float *predict(char filename[200], float *weights);
float accuracy(float *tru, float *pred, int size);
void logistic_regression(char filename[200], char test_filename[200]);

//----------------------------------------LINEAR REGRESSION--------------------------------------------------------
void linear_regression_fit(float **data, int size, float *m, float *c);
void mean_squared_error(char filename[200], float *y_pred, int size);
void root_mean_squared_error(char filename[200], float *y_pred, int size);
void mean_absolute_error(char filename[200], float *y_pred, int size);
void linear_regression_predict(char filename[200], float s, float c);
void linear_regression(char filename[200], char test_filename[200]);

//-------------------------------------------DATA SCALING---------------------------------------------------------
float **min_max_scaler_fit(float **data, int rows, int cols);
void create_scaled_data_file(float** data,float* targets, int rows,int cols,char filename[200]);
void min_max_scaler(char filename[200],char test_filename[200]);
float **standard_scaler_fit(float **data, int rows, int cols);
void standard_scaler(char filename[200], char test_filename[200]);

//-----------------------------------------KNN-----------------------------------------------------------------------
struct database* create_dataset(int rows, int column);
void data_feeding(FILE* fptr, int rows, int column, struct database *dataset);
void distance_calculate(int rows, int column, float array[], struct database *dataset);
void sort_database(int rows, struct database *dataset);
int check_probability(int k, struct database *dataset);
void accuracy_knn(int alpha[], int arr[], int row);
void KNN(char filename[200],char filename1[200]);

#endif
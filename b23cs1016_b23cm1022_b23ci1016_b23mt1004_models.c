#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "b23cs1016_b23cm1022_b23ci1016_b23mt1004_model_functions.c"

//-------------------------------------------LOGISTIC REGRESSION----------------------------------------------------------

// Function to initialize weights matrix with zeros
float *initialize_weights(int m)
{
    float *weights = (float *)malloc(m * sizeof(float *));

    for (int j = 0; j < m; j++)
    {
        weights[j] = 0.0;
    }

    return weights;
}

// sigmoid function
float sigmoid(float z)
{
    return 1 / (1 + exp(-z));
}

// Logistic regression function
float *logistic_regression_fit(float **data, float *targets, int rows, int cols, float learning_rate, int iterations)
{
    // Initialize weights
    float *weights = initialize_weights(cols);

    // Gradient descent
    for (int iter = 0; iter < iterations; iter++)
    {
        // Calculate predictions
        float predictions[rows];

        for (int i = 0; i < rows; i++)
        {
            float prediction = 0.0;
            for (int j = 0; j < cols; j++)
            {
                prediction += data[i][j] * weights[j];
            }
            predictions[i] = sigmoid(prediction);
        }

        // Calculate errors
        float errors[rows];
        for (int i = 0; i < rows; i++)
        {
            errors[i] = targets[i] - predictions[i];
        }

        // Update weights using gradient descent
        for (int j = 0; j < cols; j++)
        {
            float gradient = 0.0;
            for (int i = 0; i < rows; i++)
            {
                gradient += errors[i] * data[i][j];
            }
            weights[j] += (learning_rate * gradient) / rows;
        }
    }
    free(data);

    return weights;
}

// predictions for test data
float *predict(char filename[200], float *weights)
{
    FILE *df = fopen(filename, "r");

    if (df == NULL)
    {
        printf("Test file can't be loaded. \n");
        return NULL;
    }
    int n = 0, m = 0;
    float **data = load_data(filename, &n, &m);
    if (data == NULL)
    {
        fclose(df);
        printf("Error loading data from file '%s'.\n", filename);
        return NULL;
    }
    float *predictions = malloc(n * sizeof(float));
    if (predictions == NULL)
    {
        fclose(df);
        printf("Memory allocation failed for predictions.\n");
        return NULL;
    }
    for (int i = 0; i < n; i++)
    {
        float w_sum = 0;
        for (int j = 0; j < m; j++)
        {
            w_sum += data[i][j] * weights[j];
        }
        if (w_sum > 0.5)
            predictions[i] = 1;
        else
            predictions[i] = 0;
    }
    free(data);
    return predictions;
}

// accuracy score
float accuracy(float *tru, float *pred, int size)
{
    int count = 0;

    for (int i = 0; i < size; i++)
    {
        if (tru[i] == pred[i])
        {
            count++;
        }
    }

    // return the percent accuracy
    float accu = (count * 100.0) / size;
    return accu;
}

void logistic_regression(char filename[200], char test_filename[200])
{
    int rows = 0, cols = 0;
    float **data = load_data(filename, &rows, &cols);

    printf("Shape of the data is: %d %d \n", rows, cols);
    float *targets = get_targets(filename);

    // logistic starts from here
    float lr = 0.05;
    int itr = 3000;

    float *weights = logistic_regression_fit(data, targets, rows, cols, lr, itr);
    printf("Calculated Weights: ");
    for (int i = 0; i < 5; i++)
    {
        printf("%.3f ", weights[i]);
    }
    //get the predictions
    float *predictions = predict(test_filename, weights);

    // get the targets
    float *test_targets = get_targets(test_filename);

    int test_rows = 0, test_col = 0;
    calculate_csv_shape(test_filename, &test_rows, &test_col);
    char choice = 'y';
    printf("\nDo you want to print predicted values? (y/n)\n");
    getchar();
    scanf("%c", &choice);
    if ((choice == 'y') || (choice == 'Y'))
    {
        for (int i = 0; i < test_rows; i++)
        {
            printf("%.2f\t", predictions[i]);
        }
    }

    printf("\naccuracy score : %.3f %% \n", accuracy(test_targets, predictions, test_rows));
    printf("Thank You!");
}
//--------------------------------------------------------------------------------------------------------------------------------------

//----------------------------------------------------LINEAR REGRESSION---------------------------------------------------------

// Model training
void linear_regression_fit(float **data, int size, float *m, float *c)
{
    // variables initialisation
    float x_sum = 0, y_sum = 0, xsq_sum = 0, xy_sum = 0;
    float den = 0;
    for (int i = 0; i < size; i++)
    {
        x_sum += data[i][0];
        xsq_sum += data[i][0] * data[i][0];
        y_sum += data[i][1];
        xy_sum += data[i][0] * data[i][1];
    }

    // formula to calculate slope and bias
    den = size * xsq_sum - x_sum * x_sum;
    *m = (size * xy_sum - x_sum * y_sum) / den;
    *c = (y_sum * xsq_sum - x_sum * xy_sum) / den;
    printf("Calculated slope: %f\n", *m);
    printf("Calculated bias: %f\n", *c);
    free(data);
}

// Errors
// y_true is the actual values taken from the csv file itself, y_pred is the values predicted by the model

void mean_squared_error(char filename[200], float *y_pred, int size)
{
    float *y_true = get_targets(filename);
    float sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += (*y_true - *y_pred) * (*y_true - *y_pred);
        y_true++;
        y_pred++;
    }
    printf("Mean Squared Error: %.2f\n", (sum / size));
    printf("Thank You!");
}

void root_mean_squared_error(char filename[200], float *y_pred, int size)
{
    float *y_true = get_targets(filename);
    float sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += (*y_true - *y_pred) * (*y_true - *y_pred);
        y_true++;
        y_pred++;
    }
    sum = sqrt(sum);
    printf("Root Mean Squared Error: %.2f\n", (sum / size));
    printf("Thank You!");
}

void mean_absolute_error(char filename[200], float *y_pred, int size)
{
    // Get the targets
    float *y_true = get_targets(filename);
    float sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += (*y_true - *y_pred) / (*y_true);
        y_true++;
        y_pred++;
    }
    if (sum < 0)
        sum *= -1;
    sum = sqrt(sum);
    printf("Mean Absolute Error: %.2f%%\n", (sum / size) * 100);
    printf("Thank You!");
}

// Model prediction
void linear_regression_predict(char filename[200], float s, float c)
{
    FILE *df = fopen(filename, "r"); //open the file

    if (df == NULL)
    {
        printf("test file can't be loaded in predict function\n");
        return;
    }
    int n = 0, m = 0;
    
    //Load the data
    float **data = load_data(filename, &n, &m);
    if (data == NULL)
    {
        fclose(df);
        printf("Error loading data from file '%s'.\n", filename);
        return;
    }
    float *predictions = malloc(n * sizeof(float));
    if (predictions == NULL)
    {
        fclose(df);
        printf("Memory allocation failed for predictions.\n");
        return;
    }
    for (int i = 0; i < n; i++)
    {
        predictions[i] = (s)*data[i][0] + (c);
    }
    printf("Do you want to print predicted values? (y/n): ");
    char q;
    getchar();
    scanf("%c", &q);
    if ((q == 'y') || (q == 'Y'))
    {
        printf("Predicted values: \n");
        for (int i = 0; i < n; i++)
        {
            printf("%f\t", predictions[i]);
        }
    }
    fclose(df);
    // Draw the graph
    getchar();
    printf("\nDo you want to print graph(y/n):");
    char ch;
    scanf("%c", &ch);
    if ((ch == 'y') || (ch == 'Y'))
    {
        graph(filename,s,c);
    }
    int choice;
    do
    {
        printf("\nWhich error would you like to calculate? \n");
        printf("1. Mean Squared Error\n");
        printf("2. Root Mean Squared Error\n");
        printf("3. Mean Absolute Error\n");
        printf("4. No error\n");
        printf("Enter choice: ");
        scanf("%d", &choice);
        switch (choice)
        {
        case (1):
            mean_squared_error(filename, predictions, n);
            break;
        case (2):
            root_mean_squared_error(filename, predictions, n);
            break;
        case (3):
            mean_absolute_error(filename, predictions, n);
            break;
        case (4):
            break;
        default:
            printf("Invalid option\n");
            break;
        }
    } while ((choice < 1) || (choice > 4));
}

//called function
void linear_regression(char filename[200], char test_filename[200])
{
    int rows = 0, cols = 0;
    float **data = load_data(filename, &rows, &cols);

    printf("shape of the data is: %d %d \n", rows, cols);
    float m = 0, c = 0;

    //fit the model
    linear_regression_fit(data, rows, &m, &c);

    //predictions
    linear_regression_predict(test_filename, m, c);
    
    

    
}

//----------------------------------------------------------------------------------------------------------------------------------

//------------------------------------------------------DATA SCALING---------------------------------------------------------------

float **min_max_scaler_fit(float **data, int rows, int cols)
{
    float **scaled_data;
    scaled_data = malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++)
    {
        scaled_data[i] = malloc(cols * sizeof(float));
    }
    // min max scaling = data - min / (min - max)

    // to find min
    for (int i = 0; i < cols; i++)
    {
        float min = data[0][i];

        for (int j = 0; j < rows; j++)
        {

            if (data[j][i] < min)

                min = data[j][i];
        }

        float max = data[0][i];

        for (int j = 0; j < rows; j++)
        {

            if (data[j][i] > max)

                max = data[j][i];
        }

        // min max scaling

        for (int j = 0; j < rows; j++)
        {

            scaled_data[j][i] = (data[j][i] - min) / (max - min);
        }
    }
    free(data);
    return scaled_data;
}

void create_scaled_data_file(float **data, float *targets, int rows, int cols, char filename[200])
{

    FILE *fptr;
    fptr = fopen(filename, "w");
    for (int i = 0; i < cols - 1; i++)
    {
        fprintf(fptr, "%d,", i);
    }
    fprintf(fptr, "%d\n", cols - 1);
    for (int i = 0; i < rows - 1; i++)
    {
        for (int j = 0; j < cols - 1; j++)
        {
            fprintf(fptr, "%f,", data[i + 1][j + 1]);
        }
        fprintf(fptr, "%f\n", targets[i + 1]);
    }
    free(data);
    fclose(fptr);
}

void min_max_scaler(char filename[200], char test_filename[200])
{

    int rows = 0, cols = 0;
    int trows = 0, tcols = 0;
    float **data = load_data(filename, &rows, &cols);
    float **test_data = load_data(test_filename, &trows, &tcols);
    float *targets = get_targets(filename);
    float *test_targets = get_targets(test_filename);
    printf("shape of the training data is: %d %d \n", rows, cols);
    printf("shape of the test data is: %d %d \n", trows, tcols);
    float **scaled_data = min_max_scaler_fit(data, rows, cols);
    float **test_scaled_data = min_max_scaler_fit(test_data, trows, tcols);
    create_scaled_data_file(scaled_data, targets, rows, cols, "scaled_data.csv");
    create_scaled_data_file(test_scaled_data, test_targets, trows, tcols, "test_scaled_data.csv");
    printf("Your scaled data has been created\n");
    printf("Thank You!");
}

float **standard_scaler_fit(float **data, int rows, int cols)
{

    float sum1 = 0;
    float sum2 = 0;
    float **scaled_data;
    scaled_data = malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++)
    {
        scaled_data[i] = malloc(cols * sizeof(float));
    }

    // scaled data
    // first i need mean

    float length;

    length = sizeof(data) / sizeof(data[0]);

    // mean
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {

            sum1 += data[j][i];
        }

        float mean;
        mean = sum1 / length;

        // standard deviation

        for (int j = 0; j < rows; j++)
        {

            sum2 += (data[j][i] - mean) * (data[j][i] - mean);
        }

        float std;

        std = sqrt(sum2 / length);

        // scaled data
        for (int j = 0; j < rows; j++)
        {

            scaled_data[j][i] = (data[j][i] - mean) / std;
        }
    }
    free(data);
    return scaled_data;
}

void standard_scaler(char filename[200], char test_filename[200])
{
    int rows = 0, cols = 0;
    int trows = 0, tcols = 0;
    float **data = load_data(filename, &rows, &cols);
    float **test_data = load_data(test_filename, &trows, &tcols);
    float *targets = get_targets(filename);
    float *test_targets = get_targets(test_filename);
    printf("shape of the training data is: %d %d \n", rows, cols);
    printf("shape of the test data is: %d %d \n", trows, tcols);
    float **scaled_data = standard_scaler_fit(data, rows, cols);
    float **test_scaled_data = standard_scaler_fit(test_data, trows, tcols);
    create_scaled_data_file(scaled_data, targets, rows, cols, "scaled_data.csv");
    create_scaled_data_file(test_scaled_data, test_targets, trows, tcols, "test_scaled_data.csv");
    printf("Your scaled data has been created\n");
    printf("Thank You!");
}
//--------------------------------------------------------------------------------------------------------------------------------

//---------------------------------------------------------KNN--------------------------------------------------------------------

struct database
{
    float *data;
    float distance;
    int character;
};

struct database *create_dataset(int rows, int column)
{
    struct database *dataset = malloc(rows * sizeof(struct database));
    if (dataset == NULL)
    {
        printf("Memory allocation failed for dataset\n");
        return NULL;
    }

    for (int i = 0; i < rows; i++)
    {
        dataset[i].data = malloc(column * sizeof(float));
        if (dataset[i].data == NULL)
        {
            printf("Memory allocation failed for array in dataset\n");
            // Free previously allocated memory
            for (int j = 0; j < i; j++)
            {
                free(dataset[j].data);
            }
            free(dataset);
            return NULL;
        }
    }

    return dataset;
}

void data_feeding(FILE *fptr, int rows, int column, struct database *dataset)
{
    char string[100];
    fgets(string, sizeof(string), fptr); // Ignore header line

    for (int i = 0; i < rows; i++)
    {
        fgets(string, sizeof(string), fptr);
        char *token = strtok(string, ",");
        for (int j = 0; j < column; j++)
        {
            dataset[i].data[j] = atof(token);
            token = strtok(NULL, ",");
        }
        dataset[i].character = atoi(token);
    }
}

// function to calculate euler distanc
void distance_calculate(int rows, int column, float array[], struct database *dataset)
{
    for (int i = 0; i < rows; i++)
    {
        float sum = 0.0;
        for (int j = 0; j < column; j++)
        {
            sum += pow(dataset[i].data[j] - array[j], 2);
        }
        dataset[i].distance = sqrt(sum);
    }
}

// function to sort database according to the distance
void sort_database(int rows, struct database *dataset)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < rows - 1 - i; j++)
        {
            if (dataset[j].distance > dataset[j + 1].distance)
            {
                struct database temp = dataset[j];
                dataset[j] = dataset[j + 1];
                dataset[j + 1] = temp;
            }
        }
    }
}

// function to check probability of test case
int check_probability(int k, struct database *dataset)
{
    int array[5] = {0};
    for (int i = 0; i < k; i++)
    {
        if (dataset[i].character == 0)
        {
            array[0]++;
        }
        else if (dataset[i].character == 1)
        {
            array[1]++;
        }
        else if (dataset[i].character == 2)
        {
            array[2]++;
        }
        else if (dataset[i].character == 2)
        {
            array[3]++;
        }
        else if (dataset[i].character == 2)
        {
            array[4]++;
        }
    }
    int max_prob = 0;

    for (int i = 1; i < 5; i++)
    {
        if (array[i] > array[max_prob])
        {
            max_prob = i;
        }
    }

    return max_prob;
}

// function to calculate the accuracy of predictions
void accuracy_knn(int alpha[], int arr[], int row)
{
    int count = 0;
    for (int i = 0; i < row; i++)
    {
        if (alpha[i] == arr[i])
        {
            count++;
        }
    }

    float percent;

    percent = ((float)count / row) * 100;

    printf("\naccuracy is %f%%", percent);
}

// main function to implement knn
void KNN(char filename[200], char filename1[200])
{

    FILE *fptr = fopen(filename, "r");

    if (fptr == NULL)
    {
        printf("Unable to open file\n");
        return;
    }

    char string[1000];
    int rows = 0, column = 0;

    fgets(string, sizeof(string), fptr); // Ignore header line

    // to read data from the csv file
    char *token = strtok(string, ",");
    while (token != NULL)
    {
        column++;
        token = strtok(NULL, ",");
    }

    // to calculate number of rows in the csv file
    while (fgets(string, sizeof(string), fptr) != NULL)
    {
        rows++;
    }

    // Reset file pointer to the beginning of the file
    fseek(fptr, 0, SEEK_SET);

    // Allocate memory for the array dynamically
    float *array = malloc((column - 1) * sizeof(float));
    if (array == NULL)
    {
        printf("Memory allocation failed for array\n");
        fclose(fptr);
        return;
    }

    // Create dataset
    struct database *dataset = create_dataset(rows, column - 1);

    if (dataset == NULL)
    {
        fclose(fptr);
        free(array);
        return;
    }

    // Populate dataset
    data_feeding(fptr, rows, column - 1, dataset);

    fclose(fptr);

    // to open test dataset file
    FILE *fptr1 = fopen(filename1, "r");

    if (fptr1 == NULL)
    {
        printf("Unable to open test file\n");
        return;
    }

    int row = 0, col = 0, p = 0;

    fgets(string, sizeof(string), fptr1); // Ignore header line

    // to read data from csv file of test dataset
    token = strtok(string, ",");
    while (token != NULL)
    {
        col++;
        token = strtok(NULL, ",");
    }
    col = col - 1;

    // Reset file pointer to the beginning of the file
    fseek(fptr1, 0, SEEK_SET);

    // Skip the first line
    fgets(string, sizeof(string), fptr1);

    // Read the first row as an array of floats
    float arra[col];
    int alpha[100];
    int *arr = malloc(rows * sizeof(int)); // Allocate memory for arr
    if (arr == NULL)
    {
        printf("Memory allocation failed for arr\n");
        fclose(fptr1);
        free(array);
        free(dataset);
        return;
    }

    while (fgets(string, sizeof(string), fptr1) != NULL)
    {
        token = strtok(string, ",");
        for (int i = 0; i < col; i++)
        {
            arra[i] = atof(token);
            token = strtok(NULL, ",");
        }
        alpha[row++] = atoi(token);

        // Calculate distances
        distance_calculate(rows, column - 1, arra, dataset);

        sort_database(rows, dataset);

        // Increment row count

        // Calculate accuracy

        arr[p++] = check_probability(5, dataset);
    }
    char choice;
    printf("Do you want to print predicted values?: (y/n) ");
    getchar();
    scanf("%c", &choice);
    if ((choice == 'y') || (choice == 'Y'))
    {
        for (int i = 0; i < p; i++)
        {
            printf("%d\t", arr[i]);
        }
    }

    // calculate accuracy
    accuracy_knn(alpha, arr, row);

    // Free allocated memory
    free(array);
    free(arr);

    // Free allocated memory
    for (int i = 0; i < rows; i++)
    {
        free(dataset[i].data);
    }
    free(dataset);
    fclose(fptr1);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h> 

// to have the size of the data
void calculate_csv_shape(char filename[200], int *rows, int *cols)
{

    FILE *file = fopen(filename, "r");

    if (file == NULL)
    {
        printf("Error opening file!\n");
        return;
    }
    // Initialize variables to count rows and columns

    int temp_cols = 0;

    char line[4048];

    // Read the file line by line
    while (fgets(line, sizeof(line), file) != NULL)
    {

        // Split the line by comma to count columns
        char *token = strtok(line, ",");
        temp_cols = 0;

        while (token != NULL)
        {
            temp_cols++;
            token = strtok(NULL, ",");
        }

        // If this is the first line, count the number of columns
        if (*rows == 0)
        {
            *cols = temp_cols;
        }

        // Increment the number of rows
        (*rows)++;
    }
    
    fclose(file);
}

// csv file data loading
float **load_data(char filename[200], int *rows, int *cols)
{
    // first find the shape of the data
    calculate_csv_shape(filename, rows, cols);

    int n = *rows, m = *cols;

    FILE *df = fopen(filename, "r");

    if (df == NULL)
    {
        printf("Data can't loaded right now.\n");
        return NULL;
    }

    // Allocate memory for the 2D array
    float **loaded_data = malloc(n * sizeof(float *));
    for (int i = 0; i < n; i++)
    {
        loaded_data[i] = malloc(m * sizeof(float));
    }

    // Load the data from file to the 2D array
    char line[4096];
    for (int i = 0; i < n; i++)
    {
        if (fgets(line, sizeof(line), df) != NULL)
        {
            char *token = strtok(line, ",");
            for (int j = 0; j < m && token != NULL; j++)
            {
                loaded_data[i][j] = atof(token);
                token = strtok(NULL, ",");
            }
        }
        else
        {
            printf("Error in reading data from file.\n");
            fclose(df);
            return NULL;
        }
    }

    // Close the file
    fclose(df);

    // Return the pointer to the loaded data
    return loaded_data;
}

// function to get the target names
float *get_targets(char filename[200])
{
    int n = 0, m = 0;

    float **data = load_data(filename, &n, &m);

    // memory allocation
    float *targets = (float *)malloc(n * sizeof(float));
    if (targets == NULL)
    {
        printf("Memory allocation failed for target variables.\n");
        return NULL;
    }
    for (int i = 0; i < n; i++)
    {
        targets[i] = data[i][m - 1];
    }
    free(data);

    return targets;
}

//function to print graph for linear regression
#define MAX_LINE_LENGTH 1000
#define FILENAME_LENGTH 100

void plotGraph(double x[], double y[], int numPoints) {
    FILE *gnuplotPipe = popen("\"C:\\Program Files\\gnuplot\\bin\\gnuplot.exe\" -persist", "w");
    if (gnuplotPipe) {
        fprintf(gnuplotPipe, "plot '-' with points pt 7 lc rgb 'red' ps 0.5\n"); // 'pt 7' specifies the point style, 'lc rgb 'red'' specifies the line color, and 'ps 0.5' specifies the point size
        for (int i = 0; i < numPoints; ++i)
            fprintf(gnuplotPipe, "%lf %lf\n", x[i], y[i]);
        fprintf(gnuplotPipe, "e\n");
        fflush(gnuplotPipe);
        pclose(gnuplotPipe);
    } else {
        printf("Error opening GNU plot.\n");
    }
}

void plotLine(float m, float c) {
    FILE *gnuplotPipe = popen("\"C:\\Program Files\\gnuplot\\bin\\gnuplot.exe\" -persist", "w");
    if (gnuplotPipe) {
        fprintf(gnuplotPipe, "plot %f*x + %f\n", m, c);
        fflush(gnuplotPipe);
        // getchar(); // pause
        pclose(gnuplotPipe);
    } else {
        printf("Error opening GNU plot.\n");
    }
}

void graph(char filename[200],float m,float c) {
    
    // Remove trailing newline character if present
    if (filename[strlen(filename) - 1] == '\n')
        filename[strlen(filename) - 1] = '\0';

    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file.\n");
        return;
    }

    char line[MAX_LINE_LENGTH];
    double x[MAX_LINE_LENGTH], y[MAX_LINE_LENGTH]; // Assuming no more than MAX_LINE_LENGTH points
    int numPoints = 0;

    while (fgets(line, MAX_LINE_LENGTH, file)) {
        char *token = strtok(line, ",");
        x[numPoints] = atof(token);
        token = strtok(NULL, ",");
        y[numPoints] = atof(token);
        ++numPoints;
    }

    fclose(file);

    plotGraph(x, y, numPoints);

    plotLine(m,c);


}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Multivaraite Linear Regression Parameters:
const int n_iter = 1000;
const double alpha = 0.01;
const double epsilon = 0.001;
const int feature_num = 10;

// Functions used:
void get_data(char* line, double** data, char *filename);
void print_data(double** data, int row, int col);
int get_row(char *filename);
int get_col(char *filename);
//void reg_one(double** data1);
double predict(double* w, double* data_jk, int feature_num);
double** get_training_set(double** data, int total_date);
double* gradient_descent(double** training_set, int feature_num, int training_num, int n_iter);

int main()
{    // ++++++++++++++++++++++ Data Preprocessing ++++++++++++++++++++++++++++++++++//
    // Read raw Data from CSV, i.e. P_it
    char filename[] = "D:/001 Spring 2019/4500APP Programming/hw/HW2_1/dump2.csv";
    char line[10000];
    double **data;
    int row, col;
    row = get_row(filename);
    col = get_col(filename);
    printf("# of rows = %d\n", row);
    printf("# of cols = %d\n", col);
    
    data = (double **)malloc(row * sizeof(int *));
    for (int i = 0; i < row; ++i) {
        data[i] = (double *)malloc(col * sizeof(double));
    }
    get_data(line, data, filename);
    
    // Test data storation:
    printf("10th Asset 1st day's data: %f \n", data[10][0]);
    
    // Generate delta's: diff_it  = P_it - P_it_1
    double **diff_data;
    diff_data = (double **)malloc(row * sizeof(int *));// A one dim array of pointers
    
    int i = 0, j = 0;
    for (i = 0; i < row; i++) {
        diff_data[i] = (double *)malloc((col - 1) * sizeof(double));
    } // for every 1-dim pointer, generate one-row of pointer
    
    for (i = 0; i < row; ++i) {
        for (j = 1; j < col; ++j) {
            diff_data[i][j - 1] = data[i][j] - data[i][j - 1];
        }
    }
    //print_data(data, row, col);
    
    // Visualize the transformed data:
    int diff_row = row;
    int diff_col = col - 1;
    printf("# of rows in DIFF: %d, # of cols in DIFF: %d", diff_row, diff_col);
    printf("1st diff of 1st Asset: %f\n", diff_data[0][0]); // 0.937500
    printf("1st diff of Last Asset: %f\n", diff_data[row - 1][0]); //0.526500
    printf("Last diff of 1st Asset: %f\n", diff_data[0][col - 2]); //- 0.250200
    printf("Last diff of 1st Asset: %f\n", diff_data[row - 1][col - 2]); //0.474300
    
    /*Generate data for Asset 1: This can be converted into a for loop*/
    double* data_1;
    data_1 = (double *)malloc(diff_col * sizeof(double));
    
    for (i = 0; i < diff_col; ++i) {
        data_1[i] = diff_data[0][i];
        if (i == 0 || i == diff_col - 1) {
            printf("data_1[%d] = %f\n", i, data_1[i]);
        }
    }
    double* training_set1 = get_training_set(data_1, diff_col);
    /* Implement Gradient Descent */
    double* theta1 = gradient_descent(training_set1, feature_num, diff_col - feature_num, n_iter);
}

/* Transform Asset j's daily data: data_j (#249) into #239 samples: */
double** get_training_set(double* data_j, int total_date) {
    // generate training_set
    printf("\nGenerating training_set !\n");
    double **training_set;
    training_set = (double **)malloc((total_date - feature_num) * sizeof(int *));
    int i = 0, j = 0;
    for (i = 0; i < (total_date - feature_num); i++) {
        training_set[i] = (double *)malloc((feature_num + 1) * sizeof(double));
    } // for every 1-dim pointer, generate one-row of pointer
    
    // input data into the training_set
    int n_rows = total_date - feature_num;
    for (i = 0; i < feature_num; ++i) {
        for (j = 0; j < n_rows; ++j) {
            training_set[j][i] = data_j[j + feature_num - i - 1];
        }
    }
    for (j = 0; j < n_rows; ++j) {
        training_set[j][feature_num] = data_j[j + feature_num];
        
        if (j == 0 || j == n_rows - 1) {
            for (i = 0; i < feature_num + 1; ++i) {
                printf("training_set[%d][%d] = %f\n", j, i, training_set[j][i]);
            }
        }
    }
    return training_set;
    
}

/* Predict() : get the hypothesis h, given weights and data_j */
double predict(double* w, double* data_jk, int feature_num) {
    double h = 0;
    
    for (int i = 0; i < feature_num; i++) {
        h += w[i] * data_jk[i];
    }
    return h;
}

/* Implement Gradient Descent */
// here training_set is all samples of Asset_j
double* gradient_descent(double** training_set, int feature_num, int training_num, int n_iter)
{    /* Initailize weight*/
    double w[10] = { 0 };
    double loss = 100;
    
    for (int n = 0; n < n_iter && loss > epsilon; n++) {
        loss = 0;
        printf("%d th iteration\n ", n);
        /*Initialize Theta*/
        double del_theta[10] = { 0 };
        
        for (int i = 0; i < feature_num; i++) {
            //printf("\n +++++ Caculating Gradient in terms of Feature {%d}+++++\n", i);
            del_theta[i] = 0.0; // initialize the wei.ght to 0
            for (int m = 0; m < training_num; m++) {
                // Generate one training sample:
                double* training_set_m;
                training_set_m = (double *)malloc((feature_num) * sizeof(double));
                for (int k = 0; k < feature_num; k++) {
                    training_set_m[k] = training_set[m][k];
                }
                // Caculate the hypothesis h = w_T * x
                double h = predict(w, training_set_m, feature_num);
                double gradient = (h - training_set[m][feature_num]) * training_set[m][i] / (double)training_num;
                
                del_theta[i] += gradient;
                
            }/* End of Cal gradient for each feature: del_theta_i*/
        }/*End of Cal all gradients: del_theta*/
        
        printf("++++++++++++++++++++++ Finish Cal the Gradient --> updating Theta for {%d} th iteration\n",n );
        for (int i = 0; i < feature_num; i++) {
            w[i] -= alpha * del_theta[i];
            //printf("%dth feature's coefficient = %f\n ", i, w[i]);
        }/*End of Updating feature coeffi vector: w */
        
        //Calculate Error based on the updated feature weights
        for (int m = 0; m < training_num; m++) {
            double x_m[10] = { 0 };
            
            for (int k = 0; k < feature_num; k++) {
                x_m[k] = training_set[m][k];
            }
            
            double diff = predict(w, x_m, feature_num) - training_set[m][feature_num];
            double diff_sqr = pow(diff, 2);
            
            loss += diff_sqr / (2 * training_num);
        }/* End of calculating Error for each iteration*/
        
        printf("\n{%d}th loss= %f\n\n", n, loss);
        
    }/* End of all iteration loop */
    
    printf("Final Result:\n");
    for (int i = 0; i < feature_num; i++) {
        printf("%dth feature's coefficient = %.3lf\n ", i, w[i]);
    }
    return w;
}

void get_data(char* line, double** data, char *filename) {
    FILE* stream = fopen(filename, "r");
    int i = 0;
    while (fgets(line, 10000, stream)) //read data in lines
    {
        int j = 0;
        char *tok;
        char* tmp = strdup(line);
        for (tok = strtok(line, ","); tok && *tok; j++, tok = strtok(NULL, ",\n")) {
            data[i][j] = atof(tok);/*convert str into float*/
        }/* tokenize data*/
        i++;
        free(tmp);
    }
    fclose(stream);
}

void print_data(double** data, int row, int col)
{
    int i = 0, j = 0;
    for (i = 0; i < 10; i++) {
        for (j = 0; j < 10; j++) {
            printf("%f\t", data[i][j]);
        }
        printf("\n");
    }
}

int get_row(char *filename)
{
    char line[10000];
    int i = 0;
    FILE* stream = fopen(filename, "r");
    while (fgets(line, 10000, stream)) {
        i++;
    }
    fclose(stream);
    return i;
}

int get_col(char *filename)
{
    char line[10000];
    int i = 0;
    FILE* stream = fopen(filename, "r");
    fgets(line, 10000, stream);
    char* token = strtok(line, ",");
    while (token) {
        token = strtok(NULL, ",");
        i++;
    }
    fclose(stream);
    return i;
}

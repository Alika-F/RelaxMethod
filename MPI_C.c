#define scalar double
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <malloc.h>
#include <tgmath.h>
#include <mpi.h>

#ifndef HPC_HEADER_H
#define HPC_HEADER_H

#endif //HPC_HEADER_H

//#define STOP_ITER
//#define VERBATIM

// utils.h
#ifndef HPC_UTILS_H
#define HPC_UTILS_H

#endif //HPC_UTILS_H


#define MAX_LEN_OF_NUM 256
#define MICRO_TO_SEC 1./1000000

struct Input {
    unsigned int matrix_size;
    scalar *matrix;
    scalar *b;
    scalar *correct;
};


struct Input read_input_file(const char *filename);
scalar read_scalar(FILE *f);

bool assert_equals(const scalar *res, const scalar *correct, unsigned int len, scalar max_error, bool v);

bool
pass_test(const struct Input *input, scalar relax, scalar stop_epsilon, unsigned int max_iter, unsigned int test_number,
          unsigned int num_of_threads, scalar accuracy);

void
execute_tests(const char **tests, unsigned int num_of_tests, scalar relax, scalar stop_epsilon, unsigned int max_iter,
              unsigned int num_of_threads, scalar accuracy, unsigned int num_of_repeat);

// relax.h
#ifndef HPC_RELAX_H
#define HPC_RELAX_H

#endif //HPC_RELAX_H


struct Matrix {
    scalar *array;
    unsigned int w_matrix;
};


struct Result {
    scalar *result;
    scalar accuracy;
};


struct MultiplyParams {
    struct Matrix *matrix;
    const scalar *vector;
    unsigned int i_matrix;
    //unsigned int j_start;
    unsigned int len_of_vector;
    unsigned int num_of_threads;
};


struct ThreadParams {
    const struct MultiplyParams *mul_params;
    unsigned int my_id;
    scalar *result;
};


struct Result
calc_relax(struct Matrix a_matrix, const double *b, scalar relax_param, scalar stop_epsilon, unsigned int max_iter,
           unsigned int num_of_threads);
bool check_stop(scalar last_epsilon, scalar stop_epsilon, unsigned int iter, unsigned int max_iter);
scalar calc_metric(const scalar *vector, unsigned int len_of_vector);
void
calc_new_x(scalar *current_x, struct Matrix *a_matrix, const scalar *b, scalar relax_param, scalar *epsilon_i,
           unsigned int num_of_threads);

scalar multiply(struct MultiplyParams *params);

scalar thread_multiply(void *param);

scalar get_value(const struct Matrix *matrix, unsigned int i, unsigned int j);


//utils.c

struct Input read_input_file(const char *filename) {
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (myid != 0)
    {
        struct Input res = {
            .b = NULL, .matrix=NULL, .matrix_size=1, .correct=NULL
        };
    }
    FILE *f = fopen(filename, "r");

    char buffer[MAX_LEN_OF_NUM] = {};
    unsigned int matrix_size;
    fscanf(f, "%s", buffer);
    matrix_size = atoi(buffer);

    scalar *matrix = calloc(matrix_size * matrix_size, sizeof(scalar));
    for (int i = 0; i < matrix_size; i++)
        for (int j = 0; j < matrix_size; j++) {
            *(matrix + i * matrix_size + j) = read_scalar(f);
        }

    scalar *b = calloc(matrix_size, sizeof(scalar));
    for (int j = 0; j < matrix_size; j++) {
        b[j] = read_scalar(f);
    }

    scalar *correct = calloc(matrix_size, sizeof(scalar));
    for (int j = 0; j < matrix_size; j++) {
        correct[j] = read_scalar(f);
    }

    fclose(f);

    struct Input res = {matrix_size, matrix, b, correct};
    return res;
}


scalar read_scalar(FILE *f) {
    //char buffer[MAX_LEN_OF_NUM] = {};
    //fscanf(f, "%s", buffer);
    scalar res;// = strtod(buffer, NULL);
    fscanf(f, "%lf", &res);
    return res;
}


bool assert_equals(const scalar *res, const scalar *correct, unsigned int len, scalar max_error, bool v) {
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if (myid != 0)
        return true;

    else
    {
        scalar *dif = calloc(len, sizeof(scalar));
        for (int i = 0; i < len; i++)
            dif[i] = res[i] - correct[i];

        scalar error = calc_metric(dif, len);

        if (v)
            printf("\nError: %f\n", error);

        free(dif);
        return error <= max_error;
    }
}

bool
pass_test(const struct Input *input, scalar relax, scalar stop_epsilon, unsigned int max_iter, unsigned int test_number,
          unsigned int num_of_threads, scalar accuracy) {

    struct Matrix matrix = {.array = input->matrix, .w_matrix = input->matrix_size};

    struct Result result = calc_relax(matrix, input->b, relax, stop_epsilon, max_iter, num_of_threads);

#ifdef VERBATIM
    printf("Test #%d\nResult: \n", test_number);
    for (int i = 0; i < matrix.w_matrix; i++)
        printf("%f\t", result.result[i]);
    printf("\nCorrect answer is: \n");
    for (int i = 0; i < matrix.w_matrix; i++)
        printf("%f\t", input->correct[i]);

#define ASSERT_EQ_V true
#else
#define ASSERT_EQ_V false
#endif

    bool correct = assert_equals(result.result, input->correct, matrix.w_matrix, accuracy, ASSERT_EQ_V);
#ifdef VERBATIM
    if (correct)
        printf("\nTest passed successfully\n");
    else
        printf("\nTest not passed\n");
#endif

    return correct;
}

void
execute_tests(const char **tests, unsigned int num_of_tests, scalar relax, scalar stop_epsilon, unsigned int max_iter,
              unsigned int num_of_threads, scalar accuracy, unsigned int num_of_repeat) {

    unsigned int num_correct = 0;
    double sum_time = 0;
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    for (int i = 0; i < num_of_tests; i++)
    {
        struct timeval start, end;
        gettimeofday(&start, NULL);
        bool correct = true;
        struct Input input = read_input_file(tests[i]);
        
        for (int j = 0; j < num_of_repeat; j++)
            correct = correct && pass_test(&input, relax, stop_epsilon, max_iter, i, num_of_threads, accuracy);

        gettimeofday(&end, NULL);

        long seconds = (end.tv_sec - start.tv_sec);
        long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
        double time_passed = (double) micros * MICRO_TO_SEC / num_of_repeat;
        if (myid == 0)
            printf("TEST#%d TIME: %f\n\n", i, time_passed);
        sum_time += time_passed;
        if (correct)
        {
            num_correct++;
        }
        if (myid == 0)
        {
            free(input.matrix);
            free(input.b);
            free(input.correct);
        }
    }
    if (myid == 0) {
        //printf("\nCorrect answers %d/%d\n", num_correct, num_of_tests);
        printf("Mean time: %f\n", sum_time / num_of_tests);
    }
}


// relax.c


struct Result
calc_relax(struct Matrix a_matrix, const double *b, scalar relax_param, scalar stop_epsilon, unsigned int max_iter,
           unsigned int num_of_threads) {
    const unsigned int len_of_x = a_matrix.w_matrix;
    scalar last_epsilon;
    int myid;
    scalar *current_x;
    scalar epsilon_i[len_of_x];
    
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if (myid == 0) {
        current_x = calloc(len_of_x, sizeof(scalar));
    }

    int iter = 0;
    while (true) {
        calc_new_x(current_x, &a_matrix, b, relax_param, epsilon_i, num_of_threads);
        last_epsilon = calc_metric(epsilon_i, len_of_x);
        iter++;
        //printf("%d rank iter %d epsilon %f\n", myid, iter, last_epsilon);

        bool can_stop = check_stop(last_epsilon, stop_epsilon, iter, max_iter);
        MPI_Bcast(&can_stop, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (can_stop)
            break;
    } 

    struct Result result;
    result.result = current_x;
    result.accuracy = last_epsilon;
    return result;
}


bool check_stop(scalar last_epsilon, scalar stop_epsilon, unsigned int iter, unsigned int max_iter) {
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    return (last_epsilon <= stop_epsilon) && (myid == 0)
#ifdef STOP_ITER
           || iter >= max_iter;
#else
        ;
#endif
}


scalar calc_metric(const scalar *vector, unsigned int len_of_vector) {
    // calculate norm of vector
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if (myid != 0)
        return 0;
    scalar res = 0;
    for (int i = 0; i < len_of_vector; i++) {
        res += fabs(vector[i]); //* vector[i];
    }
    return res;
}


void
calc_new_x(scalar *current_x, struct Matrix *a_matrix, const scalar *b, scalar relax_param, scalar *epsilon_i,
           unsigned int num_of_threads) {

    unsigned int len_of_x = a_matrix->w_matrix;
    MPI_Bcast(&len_of_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    struct MultiplyParams params = {
            a_matrix,
            current_x,
            0,
            len_of_x,
            num_of_threads
    };

    for (int i = 0; i < len_of_x; i++) {
        params.i_matrix = i;
        scalar xi = 0;
        if (myid == 0) {
            xi = current_x[i];
            current_x[i] = 0;
        }
        scalar lr = -relax_param * multiply(&params);

        if (myid == 0) {
            scalar a_ii = get_value(a_matrix, i, i);
            scalar mid = (1 - relax_param) * xi + relax_param * b[i] / a_ii;

            scalar new_x_i = lr / a_ii + mid;

            epsilon_i[i] = new_x_i - xi;
            current_x[i] = new_x_i;
        }
    }
}


scalar multiply(struct MultiplyParams *params) {

    int k, j;
    MPI_Comm_size(MPI_COMM_WORLD, &k);
    MPI_Comm_rank(MPI_COMM_WORLD, &j);

    int sum_len = params->matrix->w_matrix;
    MPI_Bcast(&sum_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int len_for_me = sum_len / k;
    /*if (myid < sum_len % k)
        len_for_me++;*/

    scalar *abuffer = calloc(len_for_me, sizeof(scalar));
    scalar *xbuffer = calloc(len_for_me, sizeof(scalar));
    MPI_Scatter(params->matrix->array + params->i_matrix * params->matrix->w_matrix,
                len_for_me, MPI_DOUBLE, abuffer, len_for_me, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Scatter(params->vector,
                len_for_me, MPI_DOUBLE, xbuffer, len_for_me, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    struct Matrix new_matrix = {
        .array = abuffer, .w_matrix=len_for_me*len_for_me  // WARNING - некорректная ширина
    };
    struct MultiplyParams new_params = {
        .matrix=&new_matrix, .i_matrix=0, .len_of_vector=len_for_me, .num_of_threads=1, .vector=xbuffer
    };

    struct ThreadParams thread_params = {
            .result=NULL, .my_id=0, .mul_params=params
    };

    scalar res = thread_multiply(params);

    scalar *sums = calloc(k, sizeof(scalar));
    MPI_Gather(&res, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    scalar sum = 0;
    if (j == 0) {
        for (int i = 0; i < k; i++) {
            sum += sums[i];
        }
    }
    free(sums);
    free(xbuffer);
    free(abuffer);

    return sum;
}


scalar thread_multiply(void *param) {
    const struct ThreadParams *thread_params = (const struct ThreadParams *)param;
    const struct MultiplyParams *mul_params = thread_params->mul_params;
    scalar res = 0;

    for (unsigned int j = thread_params->my_id; j < mul_params->len_of_vector; j += mul_params->num_of_threads) {
        res += get_value(mul_params->matrix, mul_params->i_matrix, j) * mul_params->vector[j];
    }

    return res;
}


scalar get_value(const struct Matrix *matrix, unsigned int i, unsigned int j) {
    return *(matrix->array + i * matrix->w_matrix + j);
}


// main.c
void test_relax() {
    const char *tests[] = {

            "tests/test0",
            "tests/test1",
            "tests/test2",
            "tests/test3",
            "tests/test4"
    };
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    int num_of_tests = 5;
    int repeats = 10;
    execute_tests(tests, num_of_tests, 1.1, 0.00001, 10000, 1, 0.001, repeats);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    test_relax();
    MPI_Finalize();
    return 0;
}




#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

/*
  [o---o---o---o
   |   |   | \ |
   o---o---o---o
   | \ | \]| # | #
 [[o---o---o]]-0
   | # | \ | \#|
   o---o---o---o

   --- [  first area  ]
   --- [[ second area ]]
   --- #  third area  #
   --- 0 - the vertex for calc
*/

typedef struct {
    int n;
    int* ia;
    int* ja;
} base_mtrx_t;

typedef struct {
    double* a;
    double* b;
} slae_t;

typedef struct {
    double* x;
    int iter_count;
    double l2_discr;
} solve_slae_t;

typedef struct {
    int n;
    int* ia;
    int* ja;
    double* a;
} mtrx_t;

typedef struct {
    double time_spmv;
    double time_dot;
    double time_axpy;
} time_base_op_t;

int max(int a, int b);
int count_diag_elem(int cnt_cell, int k1, int k2);

inline int max(int a, int b) {
    if (a >= b) {
        return a;
    }
    else {
        return b;
    }
}

inline int count_diag_elem(int cnt_cell, int k1, int k2) {
    return cnt_cell / (k1 + k2) * k2 + max(cnt_cell % (k1 + k2) - k1, 0);
}

int check_input(char* var) {
    int result;
    char overflow;

    if (sscanf(var, "%d%c", &result, &overflow) == 1) {
        if (strchr(var, '.') == NULL) {
            return result;
        }
        else {
            printf("Incorrect input: the variable can only accept only one integer value greater than 0.\n");
            exit(1);
        }
    }
    else {
        printf("Incorrect input: the variable can only accept only one integer value greater than 0.\n");
        exit(1);
    }
}

void check_less_zero(int parameter) {
    if (parameter <= 0) {
        printf("Incorrect input: the variable can only accept only one integer value greater than 0.\n");
        exit(1);
    }
}

base_mtrx_t generate(int nx, int ny, int k1, int k2, int threads) {
    base_mtrx_t mtrx;
    
    mtrx.n = (nx + 1) * (ny + 1);
    mtrx.ia = (int*)malloc(sizeof(int) * (mtrx.n + 1));

    int with_diag_cell = count_diag_elem(nx * ny, k1, k2);
    int hor_vert_edges = (nx + 1) * ny + (ny + 1) * nx;
    int edges = with_diag_cell + hor_vert_edges;
    int size_ja = 2 * edges + mtrx.n;

    mtrx.ja = (int*)malloc(sizeof(int) * size_ja);

#pragma omp parallel for num_threads(threads) collapse(2)
    for (int i = 0; i < ny + 1; i++) {
        for (int j = 0; j < nx + 1; j++) {
            int count = 1;
            int coord_vertex = i * (nx + 1) + j;

            int diag = i > 0 ? count_diag_elem((i - 1) * nx + max(j - 1, 0), k1, k2) : 0;
            int res1 = 2 * (diag + i * nx + max(i - 1, 0) * (nx + 1)) + (nx + 1) * (i > 0);
            int res2 = 2 * max(j - 1, 0) + j * (i > 0) + (i < ny) * j + (j > 0);
            int res3 = count_diag_elem(i * nx + j * (i < ny), k1, k2) - diag;
            int result = res1 + res2 + res3 + coord_vertex;

            mtrx.ja[result] = coord_vertex;

            if (i > 0) {
                mtrx.ja[result + count] = (i - 1) * (nx + 1) + j;
                count++; //[i-1][j]
            }
            if (i < ny) {
                mtrx.ja[result + count] = (i + 1) * (nx + 1) + j;
                count++; //[i+1][j]
            }
            if (j > 0) {
                mtrx.ja[result + count] = i * (nx + 1) + j - 1;
                count++; //[i][j-1]
            }
            if (j < nx) {
                mtrx.ja[result + count] = i * (nx + 1) + j + 1;
                count++; //[i][j+1]
            }

            if (i < ny && j < nx && ((nx * i + j) % (k1 + k2)) >= k1) {
                mtrx.ja[result + count] = (i + 1) * (nx + 1) + j + 1;
                count++; //[i+1][j+1]
            }
            if (i > 0 && j > 0 && ((nx * (i - 1) + (j - 1)) % (k1 + k2)) >= k1) {
                mtrx.ja[result + count] = (i - 1) * (nx + 1) + j - 1;
                count++; //[i-1][j-1]
            }

            mtrx.ia[coord_vertex] = result;
        }
    }

    mtrx.ia[mtrx.n] = size_ja;

    return mtrx;
}

slae_t fill(base_mtrx_t mtrx, int threads) {
    slae_t slae;

    slae.a = (double*)malloc(sizeof(double) * mtrx.ia[mtrx.n]);
    slae.b = (double*)malloc(sizeof(double) * mtrx.n);

#pragma omp parallel for num_threads(threads)
    for (int i = 0; i < mtrx.n; i++) {
        double sum = 0;
        int coord_j = -1;
        for (int j_idx = mtrx.ia[i]; j_idx < mtrx.ia[i + 1]; j_idx++) {
            int j = mtrx.ja[j_idx];
            if (i != j) {
                slae.a[j_idx] = cos(i * j + i + j);
                sum += fabs(slae.a[j_idx]);
            }
            else {
                coord_j = j_idx;
            }
        }
        slae.a[coord_j] = 1.234 * sum;

        slae.b[i] = sin(i);
    }

    return slae;
}

void spmv(mtrx_t mtrx, double* vect, double* res, double* time, int t) {
    double start_time, end_time;

    start_time = omp_get_wtime();
#pragma omp parallel for num_threads(t)
    for (int i = 0; i < mtrx.n; i++) {
        res[i] = 0;
        for (int j_idx = mtrx.ia[i]; j_idx < mtrx.ia[i + 1]; j_idx++) {
            int j = mtrx.ja[j_idx];
            res[i] += mtrx.a[j_idx] * vect[j];
        }
    }
    end_time = omp_get_wtime();
    *time += end_time - start_time;
}

double dot(double* vect1, double* vect2, int len_vect, double* time, int t) {
    double res = 0;
    double start_time, end_time;

    start_time = omp_get_wtime();
#pragma omp parallel for num_threads(t) reduction(+:res)
    for (int i = 0; i < len_vect; i++) {
        res += vect1[i] * vect2[i];
    }
    end_time = omp_get_wtime();
    *time += end_time - start_time;

    return res;
}

void axpy(double alpha, double* vect1, double* vect2, int len_vect, double* res, double* time, int t) {
    double start_time, end_time;

    start_time = omp_get_wtime();
#pragma omp parallel for num_threads(t)
    for (int i = 0; i < len_vect; i++) {
        res[i] = alpha * vect1[i] + vect2[i];
    }
    end_time = omp_get_wtime();
    *time += end_time - start_time;
}

void copy_vect(double* src_vect, double* dst_vect, int len_vect, int t) {
#pragma omp parallel for num_threads(t)
    for (int i = 0; i < len_vect; i++) {
        dst_vect[i] = src_vect[i];
    }
}

void fill_vect(double* vect, int len_vect, double cnst, int t) {
#pragma omp parallel for num_threads(t)
    for (int i = 0; i < len_vect; i++) {
        vect[i] = cnst;
    }
}

solve_slae_t solve(base_mtrx_t mtrx, slae_t slae, double eps, int maxit, time_base_op_t *time, int thread) {
    solve_slae_t slv;
    slv.x = (double*)malloc(sizeof(double) * mtrx.n);

    double ro_prev = 0, ro = 0, beta, alpha;
    double* vect_r = (double*)malloc(sizeof(double) * mtrx.n);
    double* vect_z = (double*)malloc(sizeof(double) * mtrx.n);
    double* vect_p = (double*)malloc(sizeof(double) * mtrx.n);
    double* vect_q = (double*)malloc(sizeof(double) * mtrx.n);

    mtrx_t mtrx_a;
    mtrx_a.n = mtrx.n;
    mtrx_a.ia = mtrx.ia;
    mtrx_a.ja = mtrx.ja;
    mtrx_a.a = slae.a;

    mtrx_t  mtrx_m;
    mtrx_m.ia = (int*)malloc(sizeof(int) * (mtrx.n + 1));
    mtrx_m.ja = (int*)malloc(sizeof(int) * mtrx.n);
    mtrx_m.a = (double*)malloc(sizeof(double) * mtrx.n);
    mtrx_m.n = mtrx_a.n;

#pragma omp parallel for num_threads(thread)
    for (int i = 0; i < mtrx.n; i++) {
        mtrx_m.ia[i] = i;
        mtrx_m.ja[i] = i;
    }
    mtrx_m.ia[mtrx.n] = mtrx.n;

#pragma omp parallel for num_threads(thread)
    for (int i = 0; i < mtrx.n; i++) {
        for (int j_idx = mtrx_a.ia[i]; j_idx < mtrx_a.ia[i + 1]; j_idx++) {
            int j = mtrx_a.ja[j_idx];
            if (i == j) {
                mtrx_m.a[i] = 1 / mtrx_a.a[j_idx];
                break;
            }
        }
    }

    fill_vect(slv.x, mtrx.n, 0, thread);
    copy_vect(slae.b, vect_r, mtrx.n, thread);

    for (int k = 0; k < maxit; k++) {
        spmv(mtrx_m, vect_r, vect_z, &time->time_spmv, thread);
        ro = dot(vect_r, vect_z, mtrx.n, &time->time_dot, thread);

        if (k != 0) {
            beta = ro / ro_prev;
            axpy(beta, vect_p, vect_z, mtrx.n, vect_p, &time->time_axpy, thread);
        }
        else {
            copy_vect(vect_z, vect_p, mtrx.n, thread);
        }

        ro_prev = ro;

        spmv(mtrx_a, vect_p, vect_q, &time->time_spmv, thread);
        alpha = ro / dot(vect_p, vect_q, mtrx.n, &time->time_dot, thread);
        axpy(alpha, vect_p, slv.x, mtrx.n, slv.x, &time->time_axpy, thread);
        axpy(-alpha, vect_q, vect_r, mtrx.n, vect_r, &time->time_axpy, thread);

        slv.iter_count = k + 1;

        if (ro <= eps * eps) {
            break;
        }
    }

    slv.l2_discr = ro;

    free(vect_r);
    free(vect_z);
    free(vect_p);
    free(vect_q);
    free(mtrx_m.ia);
    free(mtrx_m.ja);
    free(mtrx_m.a);

    return slv;
}

int main (int argc, char** argv) {
    if (argc <= 1) {
        printf("--------------------< HELP >--------------------\n");
        printf("The program accepts command line arguments in the following:\n\n");
        printf("./main nx ny k1 k2 T eps maxit debug_print\n\n");
        printf("--- nx          - (int) the number of cells in the grid by x\n");
        printf("--- ny          - (int) the number of cells in the grid by y\n");
        printf("--- k1          - (int) parameter for the number of empty cells\n");
        printf("--- k2          - (int) parameter for the number of diagonal cells\n");
        printf("--- T           - (int) the number of threads for OpenMP\n");
        printf("--- eps         - (double) the stopping critetion, which determines\n");
        printf("                  the accuracy of the solution\n");
        printf("--- maxit       - (int) maximum number of iterations for Solve\n");
        printf("--- debug_print - (int) flag that enables debugging printing.\n");
        printf("                  Accepts values: 0 | 1\n");
        printf("                  Disabled by default: debug_print = 0\n");
        exit(1);
    }

    int nx, ny, k1, k2, t, maxit, deb_print;
    double eps;
    char overflow;
    nx = check_input(argv[1]);
    check_less_zero(nx);
    ny = check_input(argv[2]);
    check_less_zero(ny);
    k1 = check_input(argv[3]);
    k2 = check_input(argv[4]);

    if (k1 == 0 && k2 == 0) {
        printf("Incorrect input: k1 and k2 cannot be equal to 0 at the same time.\n");
        exit(1);
    }

    t = check_input(argv[5]);
    check_less_zero(t);

    if (sscanf(argv[6], "%lf%c", &eps, &overflow) != 1) {
        printf("Incorrect input: the variable can only accept only one integer value greater than 0.\n");
        exit(1);
    }

    maxit = check_input(argv[7]);
    check_less_zero(maxit);

    if (argc < 9) {
        deb_print = 0;
    }
    else {
        deb_print = check_input(argv[8]);
        if (deb_print != 0) {
            if (deb_print != 1) {
                printf("Incorrect input: deb_print can only accept integer values 0 or 1.\n");
                exit(1);
            }
        }
    }

    double start_time, end_time, time_solve;
    time_base_op_t time;
    time.time_spmv = 0;
    time.time_dot = 0;
    time.time_axpy = 0;

    printf("Name function  |  Execution time\n");
    printf("---------------+----------------\n");
    start_time = omp_get_wtime();
    base_mtrx_t mtrx = generate(nx, ny, k1, k2, t);
    end_time = omp_get_wtime();
    printf("Generate       |  %.6lf\n", end_time - start_time);

    start_time = omp_get_wtime();
    slae_t slae = fill(mtrx, t);
    end_time = omp_get_wtime();
    printf("Fill           |  %.6lf\n", end_time - start_time);

    start_time = omp_get_wtime();
    solve_slae_t slv = solve(mtrx, slae, eps, maxit, &time, t);
    end_time = omp_get_wtime();
    time_solve = end_time - start_time;
    printf("Solve          |  %.6lf\n", time_solve);

    printf("SpMV           |  %.6lf\n", time.time_spmv);
    printf("dot            |  %.6lf\n", time.time_dot);
    printf("axpy           |  %.6lf\n\n", time.time_axpy);

    double gflops_spmv = 2 * (mtrx.ia[mtrx.n] + mtrx.n) * slv.iter_count / (time.time_spmv * 1000000000);
    double gflops_dot = 4 * mtrx.n * slv.iter_count / (time.time_dot * 1000000000);
    double gflops_axpy = 3 * mtrx.n * slv.iter_count / (time.time_axpy * 1000000000);
    double gflops_solve = (2 * (mtrx.ia[mtrx.n] + mtrx.n) * slv.iter_count + \
                          4 * mtrx.n * slv.iter_count + 6 * mtrx.n * slv.iter_count + \
                          mtrx.n + 3 * slv.iter_count) / (time_solve * 1000000000);

    printf("Name function  |  ~GFLOPS\n");
    printf("---------------+----------------\n");
    printf("Solve          |  %.6lf\n", gflops_solve);
    printf("SpMV           |  %.6lf\n", gflops_spmv);
    printf("dot            |  %.6lf\n", gflops_dot);
    printf("axpy           |  %.6lf\n\n", gflops_axpy);

    printf("The discrepancy: %.5lf (after %d iteration(s))\n", slv.l2_discr, slv.iter_count);
    if (slv.l2_discr > eps * eps) {
        printf("For the specified maxit: %d the system discrepancy doesn't satisfy\n", maxit);
        printf("the specified accuracy: %.5lf.\n", eps);
    }

    if (deb_print == 1) {
        printf("N = %d\n", mtrx.n);

        printf("IA = [");
        for (int i = 0; i < mtrx.n + 1; i++) {
            if (i != mtrx.n) {
                printf("%d, ", mtrx.ia[i]);
            }
            else {
                printf("%d]\n", mtrx.ia[i]);
            }
        }

        printf("JA = [");
        for (int i = 0; i < mtrx.ia[mtrx.n]; i++) {
            if (i != mtrx.ia[mtrx.n] - 1) {
                printf("%d, ", mtrx.ja[i]);
            }
            else {
                printf("%d]\n", mtrx.ja[i]);
            }
        }

        printf("A = [");
        for (int i = 0; i < mtrx.ia[mtrx.n]; i++) {
            if (i != mtrx.ia[mtrx.n] - 1) {
                printf("%lf, ", slae.a[i]);
            }
            else {
                printf("%lf]\n", slae.a[i]);
            }
        }

        printf("b = [");
        for (int i = 0; i < mtrx.n; i++) {
            if (i != mtrx.n - 1) {
                printf("%lf, ", slae.b[i]);
            }
            else {
                printf("%lf]\n", slae.b[i]);
            }
        }
    }

    free(mtrx.ia);
    free(mtrx.ja);
    free(slae.a);
    free(slae.b);
    free(slv.x);

    return 0;
}

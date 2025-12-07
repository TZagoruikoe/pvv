#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define MAX_ADJACENT_V 7

typedef struct {
    int n;
    int (*arr)[MAX_ADJACENT_V];
    int cnt_elem; //count elements including halo
} base_mtrx_t;

typedef struct {
    double (*a)[MAX_ADJACENT_V];
    double* b;
} slae_t;

typedef struct {
    double* x;
    int iter_count;
    double l2_discr;
} solve_slae_t;

typedef struct {
    int n;
    int (*arr)[MAX_ADJACENT_V];
    double (*a)[MAX_ADJACENT_V];
} mtrx_t;

typedef struct {
    double time_spmv;
    double time_dot;
    double time_axpy;
} time_base_op_t;

typedef struct {
    int count_elem;
    int start_elem;
} for_coord_t;

typedef struct {
    int nghbr_rank[6];
    int nghbr_cnt_elem[6];
    MPI_Datatype vect_column;
} comm_info_t;

int max(int a, int b);
int min(int a, int b);

inline int max(int a, int b) {
    if (a >= b) {
        return a;
    }
    else {
        return b;
    }
}

inline int min(int a, int b) {
    if (a <= b) {
        return a;
    }
    else {
        return b;
    }
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
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            exit(1);
        }
    }
    else {
        printf("Incorrect input: the variable can only accept only one integer value greater than 0.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(1);
    }
}

void check_less_zero(int parameter) {
    if (parameter <= 0) {
        printf("Incorrect input: the variable can only accept only one integer value greater than 0.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(1);
    }
}

for_coord_t offset_coord(int n, int coords, int p) {
    for_coord_t offset;

    offset.start_elem = coords * n / p + min(n % p, coords);
    offset.count_elem = coords < n % p ? n / p + 1 : n / p;

    return offset;
}

//count elements including halo
int count_elem(int nx, int ny, for_coord_t offset[2]) {
    int cnt_elem = offset[0].count_elem * offset[1].count_elem;

    if (offset[1].start_elem > 0) {
        if (offset[0].start_elem > 0) {
            cnt_elem++;
        }

        cnt_elem += offset[0].count_elem;
    }

    if (offset[0].start_elem > 0) {
        cnt_elem += offset[1].count_elem;
    }

    if (offset[0].start_elem + offset[0].count_elem < nx + 1) {
        cnt_elem += offset[1].count_elem;
    }

    if (offset[1].start_elem + offset[1].count_elem < ny + 1) {
        cnt_elem += offset[0].count_elem;

        if (offset[0].start_elem + offset[0].count_elem < nx + 1) {
            cnt_elem++;
        }
    }

    return cnt_elem;
}

int l2g(int local, int nx, int ny, for_coord_t offset[2]) {
    int global;

    if (local < offset[0].count_elem * offset[1].count_elem) {
        global = (local / offset[0].count_elem + offset[1].start_elem) * (nx + 1) + 
                 (local % offset[0].count_elem + offset[0].start_elem);
        return global;
    }
    //for halo
    else {
        local -= offset[0].count_elem * offset[1].count_elem;
        
        //top shadow side
        if (offset[1].start_elem > 0) {
            if (offset[0].start_elem > 0) {
                if (local == 0) {
                    global = (offset[1].start_elem - 1) * (nx + 1) + (offset[0].start_elem - 1);
                    return global;
                }
                else {
                    local--;
                }
            }
            
            if (local < offset[0].count_elem) {
                global = (offset[1].start_elem - 1) * (nx + 1) + local + offset[0].start_elem;
                return global;
            }
            else {
                local -= offset[0].count_elem;
            }
        }

        //left shadow side
        if (offset[0].start_elem > 0) {
            if (local < offset[1].count_elem) {
                global = (offset[1].start_elem + local) * (nx + 1) + offset[0].start_elem - 1;
                return global;
            }
            else {
                local -= offset[1].count_elem;
            }
        }

        //right shadow side
        if (offset[0].start_elem + offset[0].count_elem < nx + 1) {
            if (local < offset[1].count_elem) {
                global = (offset[1].start_elem + local) * (nx + 1) + offset[0].start_elem + offset[0].count_elem;
                return global;
            }
            else {
                local -= offset[1].count_elem;
            }
        }

        //bottom shadow side
        if (offset[1].start_elem + offset[1].count_elem < ny + 1) {
            if (local < offset[0].count_elem) {
                global = (offset[1].start_elem + offset[1].count_elem) * (nx + 1) + local + offset[0].start_elem;
                return global;
            }
            else {
                local -= offset[0].count_elem;
            }

            if (offset[0].start_elem + offset[0].count_elem < nx + 1) {
                if (local == 0) {
                    global = (offset[1].start_elem + offset[1].count_elem) * (nx + 1) + (offset[0].start_elem + offset[0].count_elem);
                    return global;
                }
            }
        }
    }

    return -1;
}

int g2l(int global, int nx, int ny, for_coord_t offset[2]) {
    int local;
    int local_x, local_y;

    local_x = global % (nx + 1) - offset[0].start_elem;
    local_y = global / (nx + 1) - offset[1].start_elem;

    if (local_x >= 0 && local_x < offset[0].count_elem && 
        local_y >= 0 && local_y < offset[1].count_elem) {
        local = local_y * offset[0].count_elem + local_x;
        return local;
    }
    //for halo
    else {
        local = offset[0].count_elem * offset[1].count_elem;
        
        //top shadow side
        if (offset[1].start_elem > 0) {
            if (offset[0].start_elem > 0) {
                if (local_x == -1 && local_y == -1) {
                    return local;
                }
                else {
                    local++;
                }
            }

            if (local_y == -1 && local_x >= 0 && local_x < offset[0].count_elem) {
                local += local_x;
                return local;
            }
            else {
                local += offset[0].count_elem;
            }
        }

        //left shadow side
        if (offset[0].start_elem > 0) {
            if (local_x == -1 && local_y >= 0 && local_y < offset[1].count_elem) {
                local += local_y;
                return local;
            }
            else {
                local += offset[1].count_elem;
            }
        }

        //right shadow side
        if (offset[0].start_elem + offset[0].count_elem < nx + 1) {
            if (local_x == offset[0].count_elem && local_y >= 0 && local_y < offset[1].count_elem) {
                local += local_y;
                return local;
            }
            else {
                local += offset[1].count_elem;
            }
        }

        //bottom shadow side
        if (offset[1].start_elem + offset[1].count_elem < ny + 1) {
            if (local_y == offset[1].count_elem && local_x >= 0 && local_x < offset[0].count_elem) {
                local += local_x;
                return local;
            }
            else {
                local += offset[0].count_elem;
            }

            if (offset[0].start_elem + offset[0].count_elem < nx + 1) {
                if (local_y == offset[1].count_elem && 
                    local_x == offset[0].count_elem) {
                    return local;
                }
            }
        }
    }

    return -1;
}

base_mtrx_t generate(int nx, int ny, int k1, int k2, for_coord_t offset[2]) {
    base_mtrx_t mtrx;
    
    mtrx.n = offset[1].count_elem * offset[0].count_elem;
    mtrx.cnt_elem = count_elem(nx, ny, offset);
    mtrx.arr = malloc(sizeof(int) * mtrx.n * MAX_ADJACENT_V);

    for (int local_i = 0; local_i < offset[1].count_elem; local_i++) {
        for (int local_j = 0; local_j < offset[0].count_elem; local_j++) {
            int count = 1;
            int coord_vertex = local_i * offset[0].count_elem + local_j;
            mtrx.arr[coord_vertex][0] = l2g(coord_vertex, nx, ny, offset);
            int i = offset[1].start_elem + local_i;
            int j = offset[0].start_elem + local_j;
            
            if (i > 0 && j > 0 && ((nx * (i - 1) + (j - 1)) % (k1 + k2)) >= k1) {
                mtrx.arr[coord_vertex][count] = (i - 1) * (nx + 1) + j - 1;
                count++; //[i-1][j-1]
            }
            if (i > 0) {
                mtrx.arr[coord_vertex][count] = (i - 1) * (nx + 1) + j;
                count++; //[i-1][j]
            }
            if (j > 0) {
                mtrx.arr[coord_vertex][count] = i * (nx + 1) + j - 1;
                count++; //[i][j-1]
            }
            if (j < nx) {
                mtrx.arr[coord_vertex][count] = i * (nx + 1) + j + 1;
                count++; //[i][j+1]
            }
            if (i < ny) {
                mtrx.arr[coord_vertex][count] = (i + 1) * (nx + 1) + j;
                count++; //[i+1][j]
            }
            if (i < ny && j < nx && ((nx * i + j) % (k1 + k2)) >= k1) {
                mtrx.arr[coord_vertex][count] = (i + 1) * (nx + 1) + j + 1;
                count++; //[i+1][j+1]
            }
            
            while (count < MAX_ADJACENT_V) {
                mtrx.arr[coord_vertex][count] = -1;
                count++;
            }
        }
    }

    return mtrx;
}

slae_t fill(base_mtrx_t mtrx, int nx, int ny, for_coord_t offset[2]) {
    slae_t slae;

    slae.a = malloc(sizeof(double) * mtrx.n * MAX_ADJACENT_V);
    slae.b = malloc(sizeof(double) * mtrx.n);

    for (int local_i = 0; local_i < mtrx.n; local_i++) {
        double sum = 0;
        int coord_j = -1;
        int i = l2g(local_i, nx, ny, offset);
        for (int j_idx = 0; j_idx < MAX_ADJACENT_V; j_idx++) {
            int j = mtrx.arr[local_i][j_idx];
            if (j == -1) {
                break;
            }
            if (i != j) {
                slae.a[local_i][j_idx] = cos(i * j + i + j);
                sum += fabs(slae.a[local_i][j_idx]);
            }
            else {
                coord_j = j_idx;
            }
        }
        slae.a[local_i][coord_j] = 1.234 * sum;

        slae.b[local_i] = sin(i);
    }

    return slae;
}

/* Communication
diag_top_left     --- nghbr_rank[0]
dim_y_top         --- nghbr_rank[1]
dim_x_left        --- nghbr_rank[2]
dim_x_right       --- nghbr_rank[3]
dim_y_bottom      --- nghbr_rank[4]
diag_bottom_right --- nghbr_rank[5]
*/
comm_info_t fill_comm_struct(MPI_Comm comm_cart, int rank, for_coord_t offset[2]) {
    comm_info_t comm_info;
    MPI_Cart_shift(comm_cart, 0, 1, &comm_info.nghbr_rank[2], &comm_info.nghbr_rank[3]);
    MPI_Cart_shift(comm_cart, 1, -1, &comm_info.nghbr_rank[4], &comm_info.nghbr_rank[1]);

    comm_info.nghbr_cnt_elem[2] = comm_info.nghbr_rank[2] != MPI_PROC_NULL ? offset[1].count_elem : 0;
    comm_info.nghbr_cnt_elem[3] = comm_info.nghbr_rank[3] != MPI_PROC_NULL ? offset[1].count_elem : 0;
    comm_info.nghbr_cnt_elem[4] = comm_info.nghbr_rank[4] != MPI_PROC_NULL ? offset[0].count_elem : 0;
    comm_info.nghbr_cnt_elem[1] = comm_info.nghbr_rank[1] != MPI_PROC_NULL ? offset[0].count_elem : 0;
    
    //exist neighbour top left
    if (comm_info.nghbr_rank[1] != MPI_PROC_NULL && comm_info.nghbr_rank[2] != MPI_PROC_NULL) {
        int coords[2];
        MPI_Cart_coords(comm_cart, rank, 2, coords);
        
        coords[0]--;
        coords[1]--;

        MPI_Cart_rank(comm_cart, coords, &comm_info.nghbr_rank[0]);

        comm_info.nghbr_cnt_elem[0] = 1;
    }
    else {
        comm_info.nghbr_rank[0] = MPI_PROC_NULL;
        comm_info.nghbr_cnt_elem[0] = 0;
    }

    //exist neighbour bottom right
    if (comm_info.nghbr_rank[4] != MPI_PROC_NULL && comm_info.nghbr_rank[3] != MPI_PROC_NULL) {
        int coords[2];
        MPI_Cart_coords(comm_cart, rank, 2, coords);
        
        coords[0]++;
        coords[1]++;

        MPI_Cart_rank(comm_cart, coords, &comm_info.nghbr_rank[5]);

        comm_info.nghbr_cnt_elem[5] = 1;
    }
    else {
        comm_info.nghbr_rank[5] = MPI_PROC_NULL;
        comm_info.nghbr_cnt_elem[5] = 0;
    }

    MPI_Type_vector(offset[1].count_elem, 1, offset[0].count_elem, MPI_DOUBLE, &comm_info.vect_column);
    MPI_Type_commit(&comm_info.vect_column);

    return comm_info;
}

void communication(comm_info_t comm_info, MPI_Comm comm_cart, for_coord_t offset[2], double* vect) {
    MPI_Request reqs[12];

    MPI_Isend(&vect[0], 1, MPI_DOUBLE, comm_info.nghbr_rank[0], 5, comm_cart, &reqs[0]);
    MPI_Isend(&vect[0], offset[0].count_elem, MPI_DOUBLE, comm_info.nghbr_rank[1], 4, comm_cart, &reqs[1]);
    MPI_Isend(&vect[0], 1, comm_info.vect_column, comm_info.nghbr_rank[2], 3, comm_cart, &reqs[2]);
    MPI_Isend(&vect[offset[0].count_elem - 1], 1, comm_info.vect_column, comm_info.nghbr_rank[3], 2, comm_cart, &reqs[3]);
    MPI_Isend(&vect[offset[0].count_elem * (offset[1].count_elem - 1)], offset[0].count_elem, MPI_DOUBLE, 
              comm_info.nghbr_rank[4], 1, comm_cart, &reqs[4]);
    MPI_Isend(&vect[offset[0].count_elem * offset[1].count_elem - 1], 1, MPI_DOUBLE, 
              comm_info.nghbr_rank[5], 0, comm_cart, &reqs[5]);

    int pos = 6;
    int idx = offset[0].count_elem * offset[1].count_elem; //index of halo start
    for (int i_nghbr = 0; i_nghbr < 6; i_nghbr++) {
        if (comm_info.nghbr_rank[i_nghbr] != MPI_PROC_NULL) {
            MPI_Irecv(&vect[idx], comm_info.nghbr_cnt_elem[i_nghbr], MPI_DOUBLE, 
                      comm_info.nghbr_rank[i_nghbr], i_nghbr, comm_cart, &reqs[pos]);
            pos++;
            idx += comm_info.nghbr_cnt_elem[i_nghbr];
        }
    }

    MPI_Waitall(pos, reqs, MPI_STATUS_IGNORE);
}

void spmv(mtrx_t mtrx, double* vect, double* res, double* time, int nx, int ny, 
          for_coord_t offset[2], comm_info_t comm_info, MPI_Comm comm_cart) {
    double start_time, end_time;

    communication(comm_info, comm_cart, offset, vect);

    MPI_Barrier(comm_cart);
    start_time = MPI_Wtime();

    for (int i = 0; i < mtrx.n; i++) {
        res[i] = 0;
        for (int j_idx = 0; j_idx < MAX_ADJACENT_V; j_idx++) {
            int j = mtrx.arr[i][j_idx];
            if (j == -1) {
                break;
            }
            int local_j = g2l(j, nx, ny, offset);
            res[i] += mtrx.a[i][j_idx] * vect[local_j];
        }
    }

    MPI_Barrier(comm_cart);
    end_time = MPI_Wtime();
    *time += end_time - start_time;
}

void spmv_v2(mtrx_t mtrx, double* vect, double* res) {
    for (int i = 0; i < mtrx.n; i++) {
        res[i] = 0;
        res[i] += vect[i] / mtrx.a[i][0];
    }
}

double dot(double* vect1, double* vect2, int len_vect, double* time, MPI_Comm comm_cart) {
    double res = 0, tmp_res = 0;
    double start_time, end_time;

    MPI_Barrier(comm_cart);
    start_time = MPI_Wtime();

    for (int i = 0; i < len_vect; i++) {
        tmp_res += vect1[i] * vect2[i];
    }

    MPI_Allreduce(&tmp_res, &res, 1, MPI_DOUBLE, MPI_SUM, comm_cart);

    MPI_Barrier(comm_cart);
    end_time = MPI_Wtime();
    *time += end_time - start_time;

    return res;
}

void axpy(double alpha, double* vect1, double* vect2, int len_vect, double* res, double* time,
          MPI_Comm comm_cart) {
    double start_time, end_time;

    MPI_Barrier(comm_cart);
    start_time = MPI_Wtime();

    for (int i = 0; i < len_vect; i++) {
        res[i] = alpha * vect1[i] + vect2[i];
    }
    
    MPI_Barrier(comm_cart);
    end_time = MPI_Wtime();
    *time += end_time - start_time;
}

void copy_vect(double* src_vect, double* dst_vect, int len_vect) {
    for (int i = 0; i < len_vect; i++) {
        dst_vect[i] = src_vect[i];
    }
}

void fill_vect(double* vect, int len_vect, double cnst) {
    for (int i = 0; i < len_vect; i++) {
        vect[i] = cnst;
    }
}

solve_slae_t solve(base_mtrx_t mtrx, slae_t slae, double eps, int maxit, time_base_op_t *time, 
                   for_coord_t offset[2], MPI_Comm comm_cart, comm_info_t comm_info, int nx, int ny) {
    solve_slae_t slv;
    slv.x = malloc(sizeof(double) * mtrx.n);

    double ro_prev = 0, ro = 0, beta, alpha;
    double* vect_r = malloc(sizeof(double) * mtrx.n);
    double* vect_z = malloc(sizeof(double) * mtrx.n);
    double* vect_p = malloc(sizeof(double) * mtrx.cnt_elem);
    double* vect_q = malloc(sizeof(double) * mtrx.n);

    mtrx_t mtrx_a;
    mtrx_a.n = mtrx.n;
    mtrx_a.arr = mtrx.arr;
    mtrx_a.a = slae.a;

    fill_vect(slv.x, mtrx.n, 0);
    copy_vect(slae.b, vect_r, mtrx.n);

    for (int k = 0; k < maxit; k++) {
        spmv_v2(mtrx_a, vect_r, vect_z);
        ro = dot(vect_r, vect_z, mtrx.n, &time->time_dot, comm_cart);

        if (k != 0) {
            beta = ro / ro_prev;
            axpy(beta, vect_p, vect_z, mtrx.n, vect_p, &time->time_axpy, comm_cart);
        }
        else {
            copy_vect(vect_z, vect_p, mtrx.n);
        }

        ro_prev = ro;

        spmv(mtrx_a, vect_p, vect_q, &time->time_spmv, nx, ny, offset, comm_info, comm_cart);
        alpha = ro / dot(vect_p, vect_q, mtrx.n, &time->time_dot, comm_cart);
        axpy(alpha, vect_p, slv.x, mtrx.n, slv.x, &time->time_axpy, comm_cart);
        axpy(-alpha, vect_q, vect_r, mtrx.n, vect_r, &time->time_axpy, comm_cart);

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

    return slv;
}

int main (int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        if (argc <= 1) {
            printf("--------------------< HELP >--------------------\n");
            printf("The program accepts command line arguments in the following:\n\n");
            printf("./main nx ny k1 k2 T eps maxit debug_print\n\n");
            printf("--- nx          - (int) the number of cells in the grid by x\n");         // --- 1
            printf("--- ny          - (int) the number of cells in the grid by y\n");         // --- 2
            printf("--- k1          - (int) parameter for the number of empty cells\n");      // --- 3
            printf("--- k2          - (int) parameter for the number of diagonal cells\n");   // --- 4
            printf("--- eps         - (double) the stopping critetion, which determines\n");  // --- 5
            printf("                  the accuracy of the solution\n");
            printf("--- maxit       - (int) maximum number of iterations for Solve\n");       // --- 6
            printf("--- px          - (int) the number of parts into which the grid\n");      // --- 7
            printf("                  is divided in the x direction");
            printf("--- py          - (int) the number of parts into which the grid\n");      // --- 8
            printf("                  is divided in the y direction");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    int nx, ny, k1, k2, maxit, px, py;
    double eps;
    char overflow;
    nx = check_input(argv[1]);
    check_less_zero(nx);
    ny = check_input(argv[2]);
    check_less_zero(ny);
    k1 = check_input(argv[3]);
    k2 = check_input(argv[4]);

    if (k1 == 0 && k2 == 0 && rank == 0) {
        printf("Incorrect input: k1 and k2 cannot be equal to 0 at the same time.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (sscanf(argv[5], "%lf%c", &eps, &overflow) != 1 && rank == 0) {
        printf("Incorrect input: the variable can only accept only one double value greater than 0.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    maxit = check_input(argv[6]);
    check_less_zero(maxit);

    px = check_input(argv[7]);
    check_less_zero(px);

    py = check_input(argv[8]);
    check_less_zero(py);

    if (rank == 0) {
        if (px * py != size) {
            printf("Incorrect input: incorrect decomposition.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Comm comm_cart;
    int dims[2] = {px, py};
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm_cart);

    int rank_cart;
    MPI_Comm_rank(comm_cart, &rank_cart);

    int coords[2] = {0, 0};
    MPI_Cart_coords(comm_cart, rank_cart, 2, coords);

    for_coord_t offset[2];
    offset[0] = offset_coord(nx + 1, coords[0], dims[0]);
    offset[1] = offset_coord(ny + 1, coords[1], dims[1]);

    comm_info_t comm_info = fill_comm_struct(comm_cart, rank, offset);

    double start_time = 0, end_time = 0, time_solve;
    time_base_op_t time;
    time.time_spmv = 0;
    time.time_dot = 0;
    time.time_axpy = 0;

    MPI_Barrier(comm_cart);
    if (rank == 0) {
        printf("Name function  |  Execution time\n");
        printf("---------------+----------------\n");
        start_time = MPI_Wtime();
    }
    base_mtrx_t mtrx = generate(nx, ny, k1, k2, offset);
    MPI_Barrier(comm_cart);
    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("Generate       |  %.6lf\n", end_time - start_time);
    }

    MPI_Barrier(comm_cart);
    if (rank == 0) {
        start_time = MPI_Wtime();
    }
    slae_t slae = fill(mtrx, nx, ny, offset);
    MPI_Barrier(comm_cart);
    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("Fill           |  %.6lf\n", end_time - start_time);
    }

    MPI_Barrier(comm_cart);
    if (rank == 0) {
        start_time = MPI_Wtime();
    }
    solve_slae_t slv = solve(mtrx, slae, eps, maxit, &time, offset, comm_cart, comm_info, nx, ny);
    MPI_Barrier(comm_cart);
    if (rank == 0) {
        end_time = MPI_Wtime();
        time_solve = end_time - start_time;

        printf("Solve          |  %.6lf\n", time_solve);
        printf("SpMV           |  %.6lf\n", time.time_spmv);
        printf("dot            |  %.6lf\n", time.time_dot);
        printf("axpy           |  %.6lf\n\n", time.time_axpy);
    }

    if (rank == 0) {
        printf("The discrepancy: %.5lf (after %d iteration(s))\n", slv.l2_discr, slv.iter_count);
        if (slv.l2_discr > eps * eps) {
            printf("For the specified maxit: %d the system discrepancy doesn't satisfy\n", maxit);
            printf("the specified accuracy: %.5lf.\n", eps);
        }
    }

    MPI_Type_free(&comm_info.vect_column);

    free(mtrx.arr);
    free(slae.a);
    free(slae.b);
    free(slv.x);

    MPI_Finalize();

    return 0;
}

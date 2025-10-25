#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

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
    int *ia;
    int *ja;
} base_mtrx_t;

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

base_mtrx_t generate(int nx, int ny, int k1, int k2) {
    base_mtrx_t mtrx;
    
    mtrx.n = (nx + 1) * (ny + 1);
    mtrx.ia = (int*)malloc(sizeof(int) * (mtrx.n + 1));

    int with_diag_cell = count_diag_elem(nx * ny, k1, k2);
    int hor_vert_edges = (nx + 1) * ny + (ny + 1) * nx;
    int edges = with_diag_cell + hor_vert_edges;
    int size_ja = 2 * edges + mtrx.n;
    printf("size_ja = %d\n", size_ja);

    mtrx.ja = (int*)malloc(sizeof(int) * size_ja);

    for (int i = 0; i < ny + 1; i++) {
        for (int j = 0; j < nx + 1; j++) {
            int count = 1;
            int coord_vertex = i * (nx + 1) + j;

            int diag = count_diag_elem(max(i - 1, 0) * nx + max(j - 1, 0), k1, k2);
            int res1 = 2 * (diag + i * nx + max(i - 1, 0) * (nx + 1)) + (nx + 1) * (i > 0);
            int res2 = 2 * max(j - 1, 0) + j * (i > 0) + (i < ny) * j + (j > 0);
            int res3 = count_diag_elem(i * nx + j * (i < ny), k1, k2) - diag;
            int result = res1 + res2 + res3 + coord_vertex;
            printf("result = %d\n", result);

            mtrx.ja[result] = coord_vertex;

            if (i > 0) {
                mtrx.ja[result + count] = (i - 1) * (nx + 1) + j;
                count++; //[i-1][j]
            }
            if (i < nx) {
                mtrx.ja[result + count] = (i + 1) * (nx + 1) + j;
                count++; //[i+1][j]
            }
            if (j > 0) {
                mtrx.ja[result + count] = i * (nx + 1) + j - 1;
                count++; //[i][j-1]
            }
            if (j < ny) {
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

int main (int argc, char** argv) {
    if (argc <= 1) {
        printf("--------------------< HELP >--------------------\n");
        printf("--- nx          - the number of cells in the grid by x\n");
        printf("--- ny          - the number of cells in the grid by y\n");
        printf("--- k1          - parameter for the number of empty cells\n");
        printf("--- k2          - parameter for the number of diagonal cells\n");
        printf("--- debug_print - flag that enables debugging printing.\n");
        printf("                  Disabled by default: debug_print = 0\n");
        //printf("\n");
        exit(1);
    }

    int nx = atoi(argv[1]);
    int ny = atoi(argv[2]);
    int k1 = atoi(argv[3]);
    int k2 = atoi(argv[4]);
    int deb_print;
    if (argc < 6) {
        deb_print = 0;
    }
    else {
        deb_print = atoi(argv[5]);
    }

    base_mtrx_t mtrx = generate(nx, ny, k1, k2);

    if (deb_print == 1) {
        printf("N = %d\n", mtrx.n);

        printf("IA = [");
        for (int i = 0; i < mtrx.n; i++) {
            if (i != mtrx.n - 1) {
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
    }

    return 0;
}






/* Exact labelled fleet count, 10x10, fleet [5,4,3,3,2], touching allowed.
   Bonus check only: does humandecoded's own count reproduce BPK's
   30,093,975,536? NOT part of the tournament engine. */
#include <stdio.h>
#include <stdint.h>
#include <omp.h>
typedef unsigned __int128 mask_t;
static mask_t placements[5][200];
static int nplace[5];
static const int fleet[5] = {5,4,3,3,2};
static unsigned long long rec(int idx, mask_t used) {
    if (idx == 5) return 1ULL;
    unsigned long long total = 0;
    for (int i = 0; i < nplace[idx]; i++) {
        mask_t m = placements[idx][i];
        if ((m & used) == 0) total += rec(idx + 1, used | m);
    }
    return total;
}
int main(void) {
    for (int s = 0; s < 5; s++) {
        int L = fleet[s], k = 0;
        for (int r = 0; r < 10; r++)
            for (int c = 0; c + L <= 10; c++) {
                mask_t m = 0;
                for (int i = 0; i < L; i++) m |= (mask_t)1 << (r*10 + c + i);
                placements[s][k++] = m;
            }
        for (int r = 0; r + L <= 10; r++)
            for (int c = 0; c < 10; c++) {
                mask_t m = 0;
                for (int i = 0; i < L; i++) m |= (mask_t)1 << ((r+i)*10 + c);
                placements[s][k++] = m;
            }
        nplace[s] = k;
        printf("ship %d length %d placements %d\n", s, L, k);
    }
    unsigned long long total = 0;
    double t0 = omp_get_wtime();
    #pragma omp parallel for reduction(+:total) schedule(dynamic)
    for (int i = 0; i < nplace[0]; i++)
        total += rec(1, placements[0][i]);
    double t1 = omp_get_wtime();
    printf("LABELLED_FLEETS=%llu\n", total);
    printf("BPK=30093975536\n");
    printf("MATCH=%s\n", total == 30093975536ULL ? "YES" : "NO");
    printf("UNLABELLED_HALF=%llu\n", total / 2);
    printf("SECONDS=%.1f\n", t1 - t0);
    return 0;
}

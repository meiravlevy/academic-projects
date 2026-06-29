#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "lib_all_reduce.h"

int test(pg_handle* handle);
int logic_test(pg_handle* handle);

double time_diff(struct timespec start, struct timespec end) {
  double start_usec = start.tv_sec * 1e6 + start.tv_nsec / 1e3;
  double end_usec = end.tv_sec * 1e6 + end.tv_nsec / 1e3;
  return end_usec - start_usec;
}

int main(int argc, char const *argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s \"server1 server2 ... servern\"\n", argv[0]);
    return 1;
  }

  pg_handle* pg;
  if (connect_process_group(argv[1], (void**)&pg)) {
    close_pg_handle (pg);
    return EXIT_FAILURE;
  }


  if (logic_test(pg)) {
    close_pg_handle (pg);
    return EXIT_FAILURE;
  }
  if (test(pg)) {
    close_pg_handle (pg);
    return EXIT_FAILURE;
  }
  close_pg_handle(pg);
  return 0;
}


// Create work request ID that encodes both:
// - Which slot in staging buffer (for finding data when it completes)
// - Which chunk number overall (for knowing where to reduce it in outbuf)


int test(pg_handle* handle){
  const size_t buffer_size = 3<<20; // 3 million integers
  printf("Testing pg_all_reduce with buffer size: %ld bytes\n", buffer_size*sizeof(int));
  int* recvbuf = calloc(buffer_size,sizeof(int));
  if(!recvbuf) {
    fprintf(stderr, "alloc failed\n");
    return -1;
  }

  int* sendbuf = calloc(buffer_size,sizeof(int));
  if (!sendbuf) {
    fprintf(stderr, "alloc failed\n");
    free(recvbuf);
    return -1;
  }
  for (size_t i = 0; i < buffer_size; ++i)
  {
    sendbuf[i] = 1;
  }
  int iters = 1000;
  for (int exp = 3; exp < 21; ++exp)
  {
    int size = 3<<exp;
    // start clock
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < iters; ++i)
    {
      if (pg_all_reduce((void*)sendbuf, (void*)recvbuf,
                        size, DT_INT32, OP_SUM, handle) == -1) {
        fprintf(stderr, "pg_all_reduce failed\n");
        free(sendbuf);
        free(recvbuf);
        return EXIT_FAILURE;
      }
    }

    // stop clock
    clock_gettime(CLOCK_MONOTONIC, &end);

    // calculate
    double sec = (double)time_diff(start, end)/ 1e6; // convert to seconds
    int usec = (int)time_diff(start, end);
    printf("Test with size %d: took %.2f seconds for %d iterations\n", size, sec, iters);
    printf("Average time per iteration: %d microseconds\n", usec / iters);
    printf("Average throughput: %.2f Mb/s\n", (8.0 * sizeof(int) * size * iters) / (sec * (1<<20)));
    printf("\n");

  }
  free(recvbuf);
  free(sendbuf);
  return 0;
}


//ours check logic all_reduce sendbuf != recvbuf
int logic_test(pg_handle* handle) {
  const int rank = handle->rank;
  const int P    = handle->nprocs;


  // Allocate big enough once; we'll reuse for all counts.
  const size_t buffer_size = 3u << 20; // 3 * 2^20 = 3,145,728 ints
  int* recvbuf = calloc(buffer_size,sizeof(int));
  if(!recvbuf) {
    fprintf(stderr, "alloc failed\n");
    return -1;
  }

  int* sendbuf = calloc(buffer_size,sizeof(int));
  if (!sendbuf) {
    fprintf(stderr, "alloc failed\n");
    free(recvbuf);
    return -1;
  }

  for (int exp = 3; exp < 21; ++exp) {
    size_t count = 3 << exp;   // up to 3<<20 == buffer_size

    // Initialize input: value(i, rank) = 10*i + rank
    for (size_t i = 0; i < count; ++i) {
      sendbuf[i] = (int)(10 * i + rank);
//      printf("%d ", sendbuf[i]);
    }
//    printf ("\n");

    // Run all-reduce (RS + AG). recvbuf will hold the full reduced array.
    if (pg_all_reduce(sendbuf, recvbuf, (int)count,
                      DT_INT32, OP_SUM, handle)) {
      fprintf(stderr, "[Rank %d] pg_all_reduce returned error\n", rank);
      free(sendbuf);
      free(recvbuf);
      close_pg_handle(handle);
      return EXIT_FAILURE;
    }

    // Verify: for every i, sum_r (10*i + r) = 10*i*P + P*(P-1)/2
    const int base_sum_ranks = (P - 1) * P / 2;
    int errors = 0;
    for (size_t i = 0; i < count; ++i) {
      const int expected = (int)(10 * (int)i * P + base_sum_ranks);
      const int got = recvbuf[i];
//      printf("%d ", recvbuf[i]);
      if (got != expected) {
      if (errors < 16) {
        fprintf(stderr, "[Rank %d] mismatch at i=%zu: got %d, expected %d\n",
                rank, i, got, expected);
      }
        ++errors;
      }
    }
//    printf("\n");

    if (errors == 0) {
      printf("[Rank %d] ALL-REDUCE PASS (count=%zu)\n", rank, count);
    } else {
      printf("[Rank %d] ALL-REDUCE FAIL: %d mismatches (count=%zu)\n",
             rank, errors, count);
    }
  }

  free(sendbuf);
  free(recvbuf);
  return EXIT_SUCCESS;
}

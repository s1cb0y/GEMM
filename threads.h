

#include <stdint.h>
#ifdef __unix__
    #include <pthread.h>
    #include <unistd.h>
#elif defined(_WIN32) || defined(WIN32)
    #include <windows.h>
#endif

#ifndef THREADS 
    #define THREADS 1
#endif


void blockedMultiply(uint32_t start_y, uint32_t end_y);

typedef struct ThreadArgs{
    int N; //mat dimensions
    int n; //slice number
} ThreadArgs;

#ifdef __unix__
void *threadMultiply(void* args){
    ThreadArgs args = {N,j};
    int start_y = args->N/THREADS * args->n;
    int end_y   = args->N/THREADS * (args->n+1);
    blockedMultiply(start_y, end_y);
    return NULL;
}    
void CreateAndExecuteThreads(){
    pthread_t threads[THREADS];
    for(int j = 0; j< THREADS; j++){
        ThreadArgs args = {N,j};
        pthread_create(&threads[j], NULL, threadMultiply, (void*)&args);
    }
    for(int j = 0; j< THREADS; j++){
        pthread_join(threads[j], NULL);
    }
}

#elif defined(_WIN32) || defined(WIN32)

DWORD WINAPI threadMultiply(void* _args) {
  ThreadArgs* args = (ThreadArgs*)_args;
  int start_y = args->N/THREADS * args->n;
  int end_y   = args->N/THREADS * (args->n+1);
  blockedMultiply(start_y, end_y);
  return 0;
}

void CreateAndExecuteThreads(int N){
    HANDLE threads[THREADS];
    for(int j = 0; j< THREADS; j++){
        ThreadArgs args = {N,j};
        threads[j] = (HANDLE) CreateThread(NULL, 0, threadMultiply, (LPVOID)&args, 0, NULL);
    }
    for(int j = 0; j< THREADS; j++){
        WaitForSingleObject(threads[j], INFINITE);
    }
}
#endif
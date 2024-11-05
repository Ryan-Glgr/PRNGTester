#ifndef PRNGTESTER
#define PRNGTESTER

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <iostream>
#include <pthread.h>
#include <sys/shm.h>
#include <cmath>
#include <cfloat>

#define VECTOR_SIZE 30      // use four different named constants so that we can understand what in the world is happening
#define NUM_VECTORS 30
#define NUM_TESTS 30
#define NUM_FUNCTIONS 10

// make two matrices of each of our five given ranges.
// these are all global, so we can parallelize our computation of them.
double **randMatrix512;        
double **uRandomMatrix512;
double **randMatrix100;
double **uRandomMatrix100;
double **randMatrix30;
double **uRandomMatrix30;
double **randMatrix500;
double **uRandomMatrix500;
double **randMatrix32;
double **uRandomMatrix32;

// files to store the solutions we found from our functions
FILE *randSolutionCSV;     // store all the solutions we got from the rand matrices
FILE *uRandomSolutionCSV;  // store all the solutions we got from the urandom matrices
FILE *improvementCSV;  // store all the improvements we got from the solutions

typedef struct{
    double *bestSolutionVector;    // PRNG solution vector that gave us the best one
    double bestSolution;           // solution value
    double *allSolutions;          // all 30 solutions
    time_t time;                  // time of computation for the function on this matrix. this is the entire time the thread took, not just solving one solution vector
} solution;

typedef struct{
    double (*function)(double*);
    double **matrix;
} threadArguments;

typedef struct{
    double min;
    double max;
} bounds;

// boudns of each function, in order. this is useful for the local searching
bounds functionBounds[NUM_FUNCTIONS * 2] = {
    {-512, 512},
    {-512, 512},
    {-100, 100},
    {-100, 100},
    {-100, 100},
    {-100, 100},
    {-30, 30},
    {-30, 30},
    {-500, 500},
    {-500, 500},
    {-30, 30},
    {-30, 30},
    {-30, 30},
    {-30, 30},
    {-32, 32},
    {-32, 32},
    {-32, 32},
    {-32, 32},
    {-500, 500},
    {-500, 500}
};

typedef struct {
    double (*mathFunction)(double*); // Pointer to the math function (adjust the type as needed)
    solution *bestSolution;          // Pointer to the best solution
    bounds *functionBounds;          // Pointer to the bounds structure
} searchArgument;

solution **solutions;           // global variable all the threads write to

double delta = .15;                   // this guy is how we are going to change our solutions using local search

double makeRandomDoubleInBounds(int randNum, int functionMin, int functionMax);

void populateMatrix(double **inputMatrix, int functionMin, int functionMax, char method);

double **allocateMatrix();

void initializeMatrices();

void populate();

void openFiles();               // function to open our CSVs
void closeFiles();               // function to CLOSE our CSVs

void saveMatrices();

void writeMatricesToCSVs(double **inputMatrix, FILE *outputCSV);    // function to write our matrices into our CSVs
void writeSolutionsToCSVs();   // function to write our solutions into our CSVs

void freeMatrices();    

// FILE pointers where we will dump our data for distributions
FILE *randMatrix512f;
FILE *uRandomMatrix512f; 
FILE *randMatrix100f; 
FILE *uRandomMatrix100f; 
FILE *randMatrix30f; 
FILE *uRandomMatrix30f; 
FILE *randMatrix500f;
FILE *uRandomMatrix500f; 
FILE *randMatrix32f; 
FILE *uRandomMatrix32f; 

// our ten big scary math functions to run our matrices through. 
double schwefelsFunction(double *vector);
double deJongOne(double *vector);
double rosenBrockSaddle(double *vector);
double rastrigin(double *vector);
double griewangk(double *vector);
double sineEnvelopeSineWave(double *vector);
double stretchVSineWave(double *vector); 
double ackleyOne(double *vector);
double ackleyTwo(double *vector);
double eggHolder(double *vector);

void *doMathFunctions(void *seed); // this is where our threads go to start. From here, they go to whichever function matches their seed. 

solution* makeSolution();       // returns a new empty solution 

void* doMathFunctions(void *arguments);  // function that all our threads get thrown at. We pass in the seed.

void *localSearch(void* arguments); // where the magic happens

// this guy just moves our vector back in bounds
void repairVector(double *vector, bounds *functionBounds);

#endif

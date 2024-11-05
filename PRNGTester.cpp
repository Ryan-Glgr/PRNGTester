#include "PRNGTester.h"

double makeRandomDoubleInBounds(int randNum, int min, int max) {
    // Generate whole number between min and max
    if (randNum < 0) 
        randNum *= -1;
    
    return (double)((randNum % ((max - min + 1))) + min);
}

void populateMatrix(double **inputMatrix, int functionMin, int functionMax, char method) {
    
    // file handle for the urandom device
    FILE *uRand = NULL;
    
    if (method == 'u') {
        uRand = fopen("/dev/urandom", "rb");
        if (uRand == NULL) {
            perror("Failed to open /dev/urandom");
            exit(-1);
        }
    }

    // Quick double for loop to populate the matrix
    for (int i = 0; i < NUM_VECTORS; i++) {
        for (int j = 0; j < VECTOR_SIZE; j++) {
            
            int randomValue;
            
            // Use rand() or /dev/urandom based on 'method'
            if (method == 'r') 
                randomValue = rand();

             else
                fread(&randomValue, sizeof(int), 1, uRand);
            
            // Generate whole number in the range [functionMin, functionMax]
            inputMatrix[i][j] = makeRandomDoubleInBounds(randomValue, functionMin, functionMax);
        }
    }

    if (uRand) 
        fclose(uRand);
    
}

double **allocateMatrix() {
    // Allocate memory for randMatrix and check for errors
    double **retArr = (double **)malloc(NUM_VECTORS * sizeof(double *));

    for (int i = 0; i < NUM_VECTORS; i++)
        retArr[i] = (double *)malloc(VECTOR_SIZE * sizeof(double));

    return retArr;
}

void initializeMatrices(){  
        randMatrix512 = allocateMatrix();  
        uRandomMatrix512 = allocateMatrix();
        randMatrix100 = allocateMatrix();
        uRandomMatrix100 = allocateMatrix();
        randMatrix30 = allocateMatrix();
        uRandomMatrix30 = allocateMatrix();
        randMatrix500 = allocateMatrix();
        uRandomMatrix500 = allocateMatrix();
        randMatrix32 = allocateMatrix();
        uRandomMatrix32 = allocateMatrix();
}

void populate(){
        populateMatrix(randMatrix512, -512, 512, 'r');
        populateMatrix(uRandomMatrix512, -512, 512, 'u');
        populateMatrix(randMatrix100, -100, 100, 'r');
        populateMatrix(uRandomMatrix100, -100, 100, 'u');
        populateMatrix(randMatrix30, -30, 30, 'r');
        populateMatrix(uRandomMatrix30, -30, 30, 'u');
        populateMatrix(randMatrix500, -500, 500, 'r');
        populateMatrix(uRandomMatrix500, -500, 500,'u');
        populateMatrix(randMatrix32, -32, 32, 'r');
        populateMatrix(uRandomMatrix32, -32, 32, 'u');
}

void openFiles() {
    // Initialize the CSVs and check for errors
    randMatrix512f = fopen("randMatrix512.csv", "w");
    uRandomMatrix512f = fopen("uRandomMatrix512.csv", "w");
    randMatrix100f = fopen("randMatrix100.csv", "w");
    uRandomMatrix100f = fopen("uRandomMatrix100.csv", "w");
    randMatrix30f = fopen("randMatrix30.csv", "w");
    uRandomMatrix30f = fopen("uRandomMatrix30.csv", "w");
    randMatrix500f = fopen("randMatrix500.csv", "w");
    uRandomMatrix500f = fopen("uRandomMatrix500.csv", "w");
    randMatrix32f = fopen("randMatrix32.csv", "w");
    uRandomMatrix32f = fopen("uRandomMatrix32.csv", "w");
    
    randSolutionCSV = fopen("randOutput.csv", "w");
    uRandomSolutionCSV = fopen("uRandomOutput.csv", "w");

    improvementCSV = fopen("improvements.csv", "w");

    // write in our headers
    // we don't need headers for the matrices, since theyre just a dump of integers. 
    fprintf(randSolutionCSV, "time, best Solution Found,");
    fprintf(uRandomSolutionCSV, "time, best Solution Found,");
    fprintf(improvementCSV, "Iterations, best solution found, original best solution found,");

    // use a loop because we want to know the indices. 
    for (int i = 0; i < VECTOR_SIZE; i++) {
        fprintf(uRandomSolutionCSV, "AS [%x],", i);
        fprintf(randSolutionCSV, "AS [%x],", i);
    }

    for (int i = 0; i < NUM_VECTORS; i++) {
        fprintf(randSolutionCSV, "BS [%x],", i);
        fprintf(uRandomSolutionCSV, "BS [%x],", i);
    }
    fprintf(randSolutionCSV, "\n");
    fprintf(uRandomSolutionCSV, "\n");
}

void closeFiles(){
    fclose(randSolutionCSV);
    fclose(uRandomSolutionCSV);
    fclose(randMatrix512f);
    fclose(uRandomMatrix512f);
    fclose(randMatrix100f);
    fclose(uRandomMatrix100f);
    fclose(randMatrix30f);
    fclose(uRandomMatrix30f);
    fclose(randMatrix500f);
    fclose(uRandomMatrix500f);
    fclose(randMatrix32f);
    fclose(uRandomMatrix32f);
    fclose(improvementCSV);
}

void saveMatrices(){
        // now that we've collected our data on these matrices, we can write all of it into the CSVs.
        writeMatricesToCSVs(randMatrix512, randMatrix512f);
        writeMatricesToCSVs(uRandomMatrix512, uRandomMatrix512f);
        writeMatricesToCSVs(randMatrix100, randMatrix100f);
        writeMatricesToCSVs(uRandomMatrix100, uRandomMatrix100f);
        writeMatricesToCSVs(randMatrix30, randMatrix30f);
        writeMatricesToCSVs(uRandomMatrix30, uRandomMatrix30f);
        writeMatricesToCSVs(randMatrix500, randMatrix500f);
        writeMatricesToCSVs(uRandomMatrix500, uRandomMatrix500f);
        writeMatricesToCSVs(randMatrix32, randMatrix32f);
        writeMatricesToCSVs(uRandomMatrix32, uRandomMatrix32f);
}

void writeMatricesToCSVs(double **inputMatrix, FILE *outputCSV) {
    // Write randMatrix and uRandomMatrix into their respective CSVs
    for (int i = 0; i < NUM_VECTORS; i++) {
        // Write randMatrix[i][j] into the randMatrices CSV file
        for (int j = 0; j < VECTOR_SIZE; j++) {
            fprintf(outputCSV, "%d", (int)inputMatrix[i][j]);  // Writing as an int, since we are only using interger inputs
            if (j < VECTOR_SIZE - 1) 
                fprintf(outputCSV, ",");  // Add comma except for last element in row

        }
        fprintf(outputCSV, "\n");  // Newline after each vector        
    }
}

void writeSolutionsToCSVs() {

    // Write solution data into randSolutionCSV or uRandomSolutionCSV
    for (int i = 0; i < 2 * NUM_FUNCTIONS; i++) {
        
        FILE *output = (i % 2 == 0) ? randSolutionCSV : uRandomSolutionCSV;

        if (solutions[i] == NULL){
            fprintf(output, "Local Search, no improvement\n");
            continue;
        }

        // Write time as an integer (if you want it in hex, keep it as is, otherwise change to %f)
        fprintf(output, "%ld,", solutions[i]->time); // Use %f for double representation

        // Write best solution (assuming it's a double)
        fprintf(output, "%f,", solutions[i]->bestSolution); // Use %f for double representation

        // Write all found solutions
        for (int j = 0; j < NUM_VECTORS; j++) {
            fprintf(output, "%f,", solutions[i]->allSolutions[j]); // Use %f for double representation
        }

        // Write best solution vector
        for (int j = 0; j < VECTOR_SIZE; j++) {
            fprintf(output, "%d,", (int)solutions[i]->bestSolutionVector[j]); // Use %f for double representation
        }

        // End the row with a newline
        fprintf(output, "\n");
    }
}

void writeLocalSearchImprovementsToCSVs(solution **newSols) {
    // Start a new line for both CSVs
    fprintf(improvementCSV, "\n");

    // Write solution data into randSolutionCSV or uRandomSolutionCSV
    for (int i = 0; i < 2 * NUM_FUNCTIONS; i++) {
        
        if (newSols[i] == NULL){
            continue;
        }

        // Write time as an integer (if you want it in hex, keep it as is, otherwise change to %f)
        fprintf(improvementCSV, "%ld,", newSols[i]->time); // Use %f for double representation

        // Write best solution (assuming it's a double)
        fprintf(improvementCSV, "%f,", newSols[i]->bestSolution); // Use %f for double representation

        // we hid our original best solution in the all solutions array coming back from local search
        fprintf(improvementCSV, "%f,", newSols[i]->allSolutions[0]); // Use %f for double representation

        // End the row with a newline
        fprintf(improvementCSV, "\n");
    }
}

void freeMatrices(double **inputMatrix){
    for (int i = 0; i < NUM_VECTORS; i++){
        free(inputMatrix[i]);
    }
    free(inputMatrix);
}

double schwefelsFunction(double *vector){ 

    // easy part of each one
    double sum = 0;
    double outer = (418.9829 * VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; i++){
        // Get the current element from the matrix
        double xi = vector[i];

        // Compute the absolute value of the current element
        double abs_xi = fabs(xi);

        // Compute the square root of the absolute value
        double sqrt_abs_xi = sqrt(abs_xi);

        // Compute the sine of the square root
        double sine_term = sin(sqrt_abs_xi);

        // Multiply by -xi
        double product = (-1) * xi * sine_term;

        // Add the result to the sum
        sum += product;
    }
    return outer - sum;
}

double deJongOne(double *vector){ 
    double sum = 0;
    for (int i = 0; i < VECTOR_SIZE; i++){
            sum += vector[i] * vector[i];
    }
    return sum;
}

// THE SADDLE IS BROKEN
// PRODUCING NOTHING BUT INTEGER MIN VALUE!!!
double rosenBrockSaddle(double *vector){
    
    double sum = 0;
    for (int i = 0; i < VECTOR_SIZE - 1; i++){
        // Get the current element and the next element from the matrix
        double xi = vector[i];
        double xi_next = vector[i + 1];

        // Compute the squared difference term

        // Wikipedia shows this function as having x[i + 1] minus x[i]^2, rather than the other way around. 
        // https://en.wikipedia.org/wiki/Rosenbrock_function
        // the way it was written on the project handout was producing massive negative numbers. 
        double diff_squared = pow((xi_next - (xi * xi)), 2);

        // Compute the squared (1 - xi) term
        double one_minus_xi_squared = pow((1 - xi), 2);

        // Add the result to the sum
        sum += (100 * diff_squared) + one_minus_xi_squared;
    }
    return sum;

}

double rastrigin(double *vector){
    
    // computation time
    double outer = (10 * VECTOR_SIZE);
    double sum = 0;
    for (int i = 0; i < VECTOR_SIZE; i++){
        // Get the current matrix element
        double xi = vector[i];

        // Compute the squared term
        double xi_squared = pow(xi, 2);

        // Compute the cosine term
        double cos_term = 10 * cos(2 * M_PI * xi);

        // Add the result to the sum
        sum += (xi_squared - cos_term);
    }
    return sum * outer;
}

double griewangk(double *vector) {

        double sum = 0;
        for (int i = 0; i < VECTOR_SIZE; i++){
            sum += pow(vector[i], 2)/4000;
        }

        double product = 1;
        for (int i = 0; i < VECTOR_SIZE; i++){
            product *= cos(vector[i] / sqrt(i + 1));
        }

    return 1 + sum - product;
}

double sineEnvelopeSineWave(double *vector) {
        
    double sum = 0;
    for(int i = 0; i < VECTOR_SIZE - 1; i++){
        // Get the squared terms for the current and next vector elements
        double xi_squared = pow(vector[i], 2);
        double xi1_squared = pow(vector[i + 1], 2);

        // Compute the sum of squares
        double sum_of_squares = xi_squared + xi1_squared;

        // Compute the sine term and square it
        double sine_term = sin(sum_of_squares - 0.5);
        double sine_term_squared = pow(sine_term, 2);

        // Compute the denominator term and square it
        double denominator = pow(1 + 0.001 * sum_of_squares, 2);

        // Add the current term to the sum
        sum += 0.5 + (sine_term_squared / denominator);
    }
    return -1 * sum;
}

double stretchVSineWave(double *vector) {
        
        double sum = 0;
        // Loop through each element in the vector
        for (int i = 0; i < VECTOR_SIZE - 1; i++) {
            // Calculate the sum of squares of consecutive elements
            double sumOfSquares = pow(vector[i], 2) + pow(vector[i + 1], 2);

            // Take the fourth root of the sum of squares
            double fourthRoot = pow(sumOfSquares, 0.25);

            // Take the tenth root, multiply by 50, and calculate the sine (squared)
            double sineTerm = pow(sin(50 * pow(sumOfSquares, 0.1)), 2);

            // Add 1 to the sine term and multiply by the fourth root
            sum += fourthRoot * (sineTerm + 1);
        }

    return sum;
}

double ackleyOne(double *vector) {

    // Calculate the constant factor
    double expFactor = 1.0f / exp(0.2f);
    double sum = 0;
    for (int i = 0; i < VECTOR_SIZE - 1; i++) {
        double x_i = vector[i];
        double x_next_i = vector[i + 1];
        double sumOfSquares = pow(x_i, 2) + pow(x_next_i, 2);
        sum += expFactor * sqrt(sumOfSquares) + 3 * (cos(2 * x_i) + sin(2 * x_next_i));
    }
    return sum;
}

double ackleyTwo(double *vector) {

    double sum = 0;

    // Calculate the sum for each vector
    for (int i = 0; i < VECTOR_SIZE - 1; i++) {
        double xCurrent = vector[i];
        double xNext = vector[i+ 1];

        // Compute the components of the Ackley Two function
        double term1 = 20.0 + exp(1); // Adding 20 and e
        double term2 = 20.0 / (exp(0.2) * sqrt((pow(xCurrent, 2) + pow(xNext, 2)) / 2));
        double term3 = exp(0.5 * (cos(2 * M_PI * xCurrent) + cos(2 * M_PI * xNext)));

        // Combine the terms
        sum += term1 - term2 - term3;
    }
    return sum;
}

double eggHolder(double *vector) {
        
        double sum = 0;
        for (int i = 0; i < VECTOR_SIZE - 1; i++) {
            // Calculate the first term: -x_i * sin(sqrt(|x_i - x_{i+1} - 47|))
            double term1 = -vector[i] * 
                            sin(sqrt(fabs(vector[i] - 
                            vector[i + 1] - 47)));

            // Calculate the second term: -(x_{i+1} + 47) * sin(sqrt(|x_{i+1} + 47 + x_i^2|))
            double term2 = -(vector[i + 1] + 47) * 
                            sin(sqrt(fabs(vector[i + 1] + 47 + 
                            vector[i] / 2)));

        // Update the sum
        sum += term1 + term2;
    }
    return  sum;
}

solution *makeSolution() {
    // Allocate memory for the solution structure
    auto *sol = (solution *)calloc(1, sizeof(solution));

    // Allocate memory for allSolutions
    sol->allSolutions = (double *)calloc(NUM_VECTORS, sizeof(double));

    // Set the best solution to infinity
    sol->bestSolution = INFINITY;

    // Allocate memory for bestSolutionVector
    sol->bestSolutionVector = (double *)calloc(VECTOR_SIZE, sizeof(double));

    return sol;
}

void *localSearch(void* arguments){

    // cast our input pointer to our struct
    searchArgument *sa = (searchArgument *)arguments;

    // make our temp array
    double *temp = (double *)malloc(VECTOR_SIZE * sizeof(double));
    
    // our array of values that get spit out by ugly math equation as we add deltas to temp
    double *xn = (double *)malloc(VECTOR_SIZE * sizeof(double));

    // start with our best solution. if it improves, we replace it into last solution. 
    // we need this so we don't change the input vector, and we can also see if it improves iteration by iteration.
    double lastSolution = sa->bestSolution->bestSolution;

    // our solution fitness value
    double finalSolution;

    // copy over temp
    memcpy(temp, sa->bestSolution->bestSolutionVector, VECTOR_SIZE * sizeof(double));

    int iterations = 0;

    // now we're copied and ready for business
    for(int i = 0; i < VECTOR_SIZE; i++){
            
        // calculate our new temp[i]
        temp[i] -= delta;

        // do our math with our same vector, except with delta added to whatever X[i] is. 
        // the symbol soup means it is a double, but it comes from a function. A function which takes a double POINTER, and that function is whatever math function is, passing temp as an argument. 
        double newObjectiveFunction = ((double (*)(double*))(sa->mathFunction))(temp);

        // now we calculate the new partial solution
        xn[i] = lastSolution - (delta * (newObjectiveFunction -  lastSolution));

        // after all those calculations, reset temp.
        temp[i] = sa->bestSolution->bestSolutionVector[i];

        // now we check our bounds for the solution Xn we found
        repairVector(xn, functionBounds);   

        // now the final showdown. Does our new solution beat that old thing. if so, we return the new solution.
        finalSolution = ((double (*)(double*))(sa->mathFunction))(xn);

        // this is so we know how many times we ran local search to get a better value
        iterations++;

        // if we didn't improve just get out
        if(finalSolution > lastSolution)
            break;

        // update our last solution. 
        lastSolution = finalSolution;
    } 

    // done with temp
    free(temp);

    // if final solution beats what we started with, then we return the new one. 
    if(lastSolution < sa->bestSolution->bestSolution){
        
        solution *newSol = makeSolution();
        // since local search runs fast, we don't have anything worthwhile from measuring time, so we instead count iterations that we improved
        newSol->time = iterations;
        memcpy(newSol->bestSolutionVector, xn, VECTOR_SIZE * sizeof(double));
        free(xn);
        newSol->bestSolution = lastSolution;
        newSol->allSolutions[0] = sa->bestSolution->bestSolution; // track our original input as well. 

        pthread_exit((void *)newSol);
    }

    free(xn);
    // we just return null if the old solution was still better.
    pthread_exit(nullptr);
}

// we're going the mickey mouse route. Since many of our objective functions are very symmetrical, what we can do is mod the vector by its bounds.
// IE, we're playing pac man, go off the left edge, come back on the right.
// a little wrap around
void repairVector(double *vector, bounds *b) {
    
    double range = b->max - b->min;
    
    for (int i = 0; i < VECTOR_SIZE; i++) {
        if (vector[i] > b->max) {
            vector[i] = b->min + fmod(vector[i] - b->min, range);
        } else if (vector[i] < b->min) {
            vector[i] = b->max - fmod(b->min - vector[i], range);
        }
    }
}

void* doMathFunctions(void *arguments){
    
    // cast our input pointer to our struct
    threadArguments *args = (threadArguments *)arguments;

    // make a blank solution
    solution *sol = makeSolution();

    // get our start time
    time_t start = time(NULL);

    // get our best solution
    double bestSolution = INFINITY;

    for (int i = 0; i < NUM_VECTORS; i++){
        
        // do our math function and get that value
        double value = ((double (*)(double*))(args->function))(args->matrix[i]);
        sol->allSolutions[i] = value;
        
        // update our best value if this is best
        if (abs(value) < bestSolution){
            bestSolution = value;
            memcpy(sol->bestSolutionVector, args->matrix[i], VECTOR_SIZE * sizeof(double));
        }
    }
    sol->bestSolution = bestSolution;
    sol->time = time(NULL) - start;
    return (void *)sol;
}

int main(){
    
    // initialize our shared memory
    initializeMatrices();

    // this big ugly monster has to live in main sadly. 
    // you can't put it in the header file or hide it anywhere really because the matrices have to be made already.
    threadArguments args[NUM_FUNCTIONS * 2] = {
        {schwefelsFunction, randMatrix512},
        {schwefelsFunction, uRandomMatrix512},
        {deJongOne, randMatrix100},
        {deJongOne, uRandomMatrix100},
        {rosenBrockSaddle, randMatrix100},
        {rosenBrockSaddle, uRandomMatrix100},
        {rastrigin, randMatrix30},
        {rastrigin, uRandomMatrix30},
        {griewangk, randMatrix500},
        {griewangk, uRandomMatrix500},
        {sineEnvelopeSineWave, randMatrix30},
        {sineEnvelopeSineWave, uRandomMatrix30},
        {stretchVSineWave, randMatrix30},
        {stretchVSineWave, uRandomMatrix30},
        {ackleyOne, randMatrix32},
        {ackleyOne, uRandomMatrix32},
        {ackleyTwo, randMatrix32},
        {ackleyTwo, uRandomMatrix32},
        {eggHolder, randMatrix500},
        {eggHolder, uRandomMatrix500},
    };

    // open up the CSVs
    openFiles();

    solutions = (solution **)malloc(NUM_FUNCTIONS * 2 * sizeof(solution *));

    // make a new array of solutions for when we do local search on our best solutions
    solution **newSolutions = (solution **)malloc(NUM_FUNCTIONS * 2 * sizeof(solution *));

    // make an array of search arguments
    searchArgument **searchArgs = (searchArgument **)malloc(NUM_FUNCTIONS * 2 * sizeof(searchArgument *));

    for (int i = 0; i < NUM_TESTS; i++) {

        // call our function which populates these boys
        populate();

        // first we spawn 20 new threads.
        // then we make those 20 all pick a function and one of the matrices, and run the calculations
        pthread_t threads[NUM_FUNCTIONS * 2];
        for(int j = 0; j < NUM_FUNCTIONS * 2; j++)
            pthread_create(&threads[j], nullptr, doMathFunctions, (void *)&args[j]);

        // Collect the results from the threads
        for (int j = 0; j < NUM_FUNCTIONS * 2; j++) {
            void *thread_result;  // Temporary variable to hold the thread's return value
            pthread_join(threads[j], &thread_result);  // Join the thread and get the return value

            // Cast the void * back to solution * and store it in the solutions array
            solutions[j] = (solution *)thread_result;
        }
        // write the solutions to the CSVs
        writeSolutionsToCSVs();

        // write all our matrix inputs so we can track the PRNGs against each other. 
        saveMatrices();

        // array to store the threads in that localSearch runs with
        pthread_t searchThreads[NUM_FUNCTIONS * 2];  // Array to hold threads for localSearch

        // create 20 new threads, each time we initialize a new search argument as well.
        for (int j = 0; j < NUM_FUNCTIONS * 2; j++) {
            searchArgs[j] = (searchArgument *)malloc(sizeof(searchArgument));
            searchArgs[j]->bestSolution = solutions[j];
            searchArgs[j]->mathFunction = args[j].function;
            searchArgs[j]->functionBounds = &functionBounds[j];

            // Create thread for each call to localSearch
            pthread_create(&searchThreads[j], NULL, localSearch, (void *)searchArgs[j]);
        }

        // Collect results from threads
        for (int j = 0; j < NUM_FUNCTIONS * 2; j++) {
            void *thread_result;
            pthread_join(searchThreads[j], &thread_result);

            // Cast thread result back to solution* and store it in newSolutions
            newSolutions[j] = (solution *)thread_result;
        }

        writeLocalSearchImprovementsToCSVs(newSolutions);

        // Clean up searchArgs memory
        for (int j = 0; j < NUM_FUNCTIONS * 2; j++) {
            free(searchArgs[j]);
        }

        // free all that memory from the initial solutions
        for(int j = 0; j < NUM_FUNCTIONS * 2; j++){
            free(solutions[j]->allSolutions);
            free(solutions[j]->bestSolutionVector);
            free(solutions[j]);
        }
    
        // clean up newSolutions as well
        for(int j = 0; j < NUM_FUNCTIONS * 2; j++){      
            // since we very well could have some NULLS...
            if (newSolutions[j] == NULL) continue;

            free(newSolutions[j]->allSolutions);
            free(newSolutions[j]->bestSolutionVector);
            free(newSolutions[j]);
        }
    
    }// end testing 

    // free all the matrices
    freeMatrices(randMatrix512);
    freeMatrices(uRandomMatrix512);
    freeMatrices(randMatrix100);
    freeMatrices(uRandomMatrix100);
    freeMatrices(randMatrix30);
    freeMatrices(uRandomMatrix30);
    freeMatrices(randMatrix500);
    freeMatrices(uRandomMatrix500);
    freeMatrices(randMatrix32);
    freeMatrices(uRandomMatrix32);

    free(solutions);
    free(newSolutions);
    free(searchArgs);

    closeFiles();

    printf("\nFunction Testing Complete\nProducing Stats...\n\n");

    return 0;
}
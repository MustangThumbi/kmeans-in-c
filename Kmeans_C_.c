#include <stdio.h>
#include <math.h>
#include <stdlib.h>



// intitialize the cluster centers to zero


void initialize_centers(int k, int num_of_features, double centers[][num_of_features]) {
    int i, j;
    for (i = 0; i < k; i++) {
        for (j = 0; j < num_of_features; j++) {
            centers[i][j] = 0.0;
        }
    }
}

// calculate the distance between the old and new cluster centers
double euclidean_distance(int num_of_features, double* point1, double* point2) {
int i;
double distance = 0.0;
for (i=0; i<num_of_features; i++) {
    distance += (point1[i] - point2[i]) * (point1[i] - point2[i]);
}
return sqrt(distance);
}


//choose k initial cluster centers
// choose the 1st center randomly
// choose the next centers to be the farthest from the previous centers
void choose_initial_centers(int k, int num_of_points, int num_of_features, double points[][num_of_features], double centers[][num_of_features]) {
    int i, j, n;
    int *is_center = (int *)malloc(num_of_points * sizeof(int));
    for (i = 0; i < num_of_points; i++) {
        is_center[i] = 0;
    }
    n = rand() % num_of_points;
    is_center[n] = 1;
    for (j = 0; j < num_of_features; j++) {
        centers[0][j] = points[n][j];
    }
    for (i = 1; i < k; i++) {
        double max_distance = 0.0;
        for (n = 0; n < num_of_points; n++) {
            if (!is_center[n]) {
                double distance = 0.0;
                for (j = 0; j < num_of_features; j++) {
                    distance += (points[n][j] - centers[i-1][j]) * (points[n][j] - centers[i-1][j]);
                }
                if (distance > max_distance) {
                    max_distance = distance;
                    for (j = 0; j < num_of_features; j++) {
                        centers[i][j] = points[n][j];
                    }
                }
            }
        }
    }
    free(is_center);
}

// calculate the euclidean distance between two points

// kmeans clustering
void kmeans(int k, int num_of_points, int num_of_features, double points[][num_of_features],
int* membership, double cluster_centers[][num_of_features], double* cluster_sizes) {
int i, j, point_index, center_index, cluster_index;
double min_distance, distance;
// assign each point to the cluster with the closest center
for (point_index=0; point_index<num_of_points; point_index++) {
    min_distance = 1.0e+30;
    for (center_index=0; center_index<k; center_index++) {
        distance = euclidean_distance(num_of_features, points[point_index], cluster_centers[center_index]);
        if (distance < min_distance) {
            min_distance = distance;
            cluster_index = center_index;
        }
    }
    membership[point_index] = cluster_index;
}
// update the cluster centers
for (i=0; i<k; i++) {
    cluster_sizes[i] = 0.0;
    for (j=0; j<num_of_features; j++) {
        cluster_centers[i][j] = 0.0;
    }
}
for (point_index=0; point_index<num_of_points; point_index++) {
    cluster_index = membership[point_index];
    cluster_sizes[cluster_index]++;
    for (j=0; j<num_of_features; j++) {
        cluster_centers[cluster_index][j] += points[point_index][j];
    }
}
for (i=0; i<k; i++) {
    for (j=0; j<num_of_features; j++) {
        cluster_centers[i][j] /= cluster_sizes[i];
    }
}
}

// main function
int main(int argc, char** argv) {
int i, j, k, num_of_points, num_of_features, num_of_iterations, num_of_clusters;
double* points;
double* cluster_centers;
double* cluster_sizes;
int* membership;
double start_time, end_time;
// read the input file
FILE* file = fopen("diabetes.csv", "r");
fscanf(file, "%d", &num_of_points);
fscanf(file, "%d", &num_of_features);
fscanf(file, "%d", &num_of_clusters);
fscanf(file, "%d", &num_of_iterations);
points = (double*) malloc(num_of_points * num_of_features * sizeof(double));
for (i=0; i<num_of_points; i++) {
    for (j=0; j<num_of_features; j++) {
        fscanf(file, "%lf", &points[i * num_of_features + j]);
    }
}
fclose(file);
// allocate memory for the cluster centers and sizes
cluster_centers = (double*) malloc(num_of_clusters * num_of_features * sizeof(double));
cluster_sizes = (double*) malloc(num_of_clusters * sizeof(double));
// allocate memory for the membership
membership = (int*) malloc(num_of_points * sizeof(int));
// choose the initial cluster centers
choose_initial_centers(num_of_clusters, num_of_points, num_of_features, points, cluster_centers);
// start the timer
start_time = omp_get_wtime();
// run kmeans
for (i=0; i<num_of_iterations; i++) {
    kmeans(num_of_clusters, num_of_points, num_of_features, points, membership, cluster_centers, cluster_sizes);
}
// end the timer
end_time = omp_get_wtime();
// print the results
printf("Number of points: %d", num_of_points);
printf("Number of features: %d", num_of_features);
printf("Number of clusters: %d", num_of_clusters);
printf("Number of iterations: %d", num_of_iterations);
printf("Time: %lf", end_time - start_time);
// free the memory
free(points);
free(cluster_centers);
free(cluster_sizes);
free(membership);
return 0;
}


   











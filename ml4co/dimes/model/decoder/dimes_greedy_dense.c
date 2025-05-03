#include<stdio.h>
#include<stdlib.h>
#include<math.h>

void print_array(int n, int * arr) {
    for (int i=0; i<n; i++) {
        printf("%d ", arr[i]);
    }
}

void print_double(double * nan){
    printf("%lf\n", *nan);
}

void check_data(double * dist, int n) {
    for (int i = 0; i < n * n; i++) {
        if (!isnormal(dist[i]) && dist[i]!=0.) {
            printf("Error: the data is invalid! [%d]-%lf\n", i, dist[i]);
            exit(1);
        }
    }
}
            
int nearest(int last, int n, double* dist, int* node_flag){
    int cur_min_dist = 1e8;
    int res;
    for (int j=0; j<n; j++){ 
        // try node j
        if (node_flag[j]) continue;
        // from node last -> j
        if (dist[last * n + j] < cur_min_dist) {
            cur_min_dist = dist[last * n + j];
            res = j;
        }
    }
    return res;
}


int farthest(int last, int n, double* dist, int* node_flag){
    int cur_min_dist = -1e8;
    int res;
    for (int j=0; j<n; j++){ 
        // try node j
        if (node_flag[j]) continue;
        // from node last -> j
        if (dist[last * n + j] > cur_min_dist) {
            cur_min_dist = dist[last * n + j];
            res = j;
        }
    }
    return res;
}

void random_walk(int n, double * dist, int * path, double * cost) {
    
}

void nearest_neighbor(int n, double * dist, int * path, double * cost) {
    // n is the number of nodes
    // size(path) == doublen
    // size(dist) == n * n
    check_data(dist, n);
    
    int * node_flag = (int *)malloc(sizeof(int) * n); // recording whether a node is visited
    if (node_flag == NULL) {printf("Error malloc.\n"); exit(0);}

    double cur_min_dist = 1e8;
    int start = rand() % n, last, cur;

    for (int i=0; i<n; i++) node_flag[i] = 0;
    last = start;
    node_flag[last] = 1;
    path[0] = last;
    *cost = 0;

    // search from node start
    for (int step=1; step<n; step++) { // try n-1 steps;
        last = path[step] = nearest(last, n, dist, node_flag);
        node_flag[last] = 1;
    }

    // record the solution
    for (int step=0; step<n; step++) *cost += dist[path[step] * n + path[(step + 1) % n]];

    free(node_flag);
}



void nearest_insertion(int n, double * dist, int * path, double * cost) {
    // n is the number of nodes
    // size(path) == doublen
    // size(dist) == n * n
    check_data(dist, n);
    
    int * node_flag = (int *)malloc(sizeof(int) * n); // recording whether a node is visited
    int * nearest_idx = (int *)malloc(sizeof(int) * n);
    if (node_flag == NULL) {printf("Error malloc.\n"); exit(0);}

    double cur_min_dist = 1e8;
    int start = rand() % n, next, cur;

    for (int i=0; i<n; i++) node_flag[i] = 0;
    node_flag[start] = 1;
    path[0] = start;
    *cost = 0;
  
    next = path[1] = nearest(start, n, dist, node_flag);
    node_flag[next] = 1;

    for(int i = 0; i < n; i++) {
        if (node_flag[i]) continue;
        if (dist[start * n + i] < dist[next * n + i])
            nearest_idx[i] = start;
        else 
            nearest_idx[i] = next;
    }

    // search from node start
    for (int step=2; step<n; step++) { // try n-1 steps;
        for (int i = 0; i < n; i++){
            // if visited ignore it 
            if (node_flag[i]) continue;
            // initialize with first unvisited
            if(node_flag[next]){
                next = i;
                continue;
            }
            // if nearer to the current tour, save. 
            if(dist[nearest_idx[i], i] < dist[nearest_idx[next], i]){
                next = i;
            }
        }

        // find nearest node in the tour to nextCity
        int target = 1e8;
        int idx;
        for (int j = 0; j < step; j++) {
            int tmp = dist[path[j]*n+next] + dist[next*n+path[(j+1)%step]] - dist[j*n+(j+1)%n];
            if (tmp < target) {
                idx = j;
                target = tmp;
            }
        }

        // insertion
        if (idx == step - 1) path[step] = next;
        int j;
        for (j = step; j >= idx + 2; j--) {
            path[j] = path[j-1];
        }
        path[j] = next;

        // mark nextCity
        node_flag[next] = 1;

        // update nearest neighbours to unvisited nodes 
        for(int i=0; i<n; i++){
            if(node_flag[i])
                continue;
            if(dist[next*n+i] < dist[nearest_idx[i]*n+i]) ///***
                nearest_idx[i] = next;
        }
    }

    // record the solution
    for (int step=0; step<n; step++) *cost += dist[path[step] * n + path[(step + 1) % n]];

    free(node_flag);
}


void farthest_insertion(int n, double * dist, int * path, double * cost)
{
	// unsigned int (*dist)[n] = batch_dist[data_index];

	int visited[n];
	int next[n];
	
	for (int i = 0 ; i < n; i ++) 
	{
		visited[i] = 0;
		next[i] = i;
	}

	visited[0] = 1;

	for (int size=1; size<n; size++)
	{
		int min_cost_inc;
		int global_cost_inc = 0-0x7FFFFFFF;
		int bs=-1, bm=-1, be=-1;
		
		for (int m = 0 ; m < n; m++)	// candidate node to insert
		{
			int ls, le;
			int s = 0;
			
			if (visited[m]) continue;

			min_cost_inc = 0x7FFFFFFF;

			for (int i =0; i<size; i++)
			{
				int e = next[s];
				
				// check
				int cost_inc = dist[s * n + m] + dist[m * n + e] - dist[s * n + e];
				if(cost_inc < min_cost_inc)
				{
					min_cost_inc = cost_inc;
					ls = s;
					le = e;
				}
				
				// move
				s = next[s];
			}

			if (min_cost_inc > global_cost_inc)
			{
				global_cost_inc = min_cost_inc;
				bs = ls;
				bm = m;
				be = le;
			}
		}

		// insert node
		next[bs] = bm;
		next[bm] = be;
		visited[bm] = 1;
	}

	unsigned int total_len = 0;
	int ss = 0;

	for (int i = 0; i < n; i++)
	{
		total_len += dist[ss * n + next[ss]];

        path[i] = ss;
		
		// #if PRINT_RESULT_TRAJ==1
		// printf("%d->%d, %d\n", ss, next[ss], total_len);
		// fflush(stdout);
		// #endif

		ss = next[ss];
	}
	// #if PRINT_RESULT_TRAJ==true
	// printf("%n");
	// #endif

	// return total_len;
    *cost = total_len;
}
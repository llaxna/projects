#include <stdio.h>

struct Task {
    int id;
    int arrival;
    int burst;
};

void calculate_waiting_turnaround_time(struct Task tasks[], int n, float *avg_waiting, float *avg_turnaround) {
    int total_waiting = 0;
    int total_turnaround = 0;
    int finish_time[n];
    int remaining[n];

    for (int i = 0; i < n; i++) {
        remaining[i] = tasks[i].burst;
    }

    int done = 0;
    int current = 0;
    int shortest = 0;

    while (done < n) {
        shortest = -1;
        for (int i = 0; i < n; i++) {
            if (tasks[i].arrival <= current && remaining[i] > 0) {
                if (shortest == -1 || remaining[i] < remaining[shortest]) {
                    shortest = i;
                }
            }
        }

        if (shortest == -1) {
            current++;
        } else {
            finish_time[shortest] = current + tasks[shortest].burst;
            total_waiting += current - tasks[shortest].arrival;
            total_turnaround += finish_time[shortest] - tasks[shortest].arrival;
            remaining[shortest] = 0;
            done++;
            current = finish_time[shortest];
        }
    }

    *avg_waiting = (float)total_waiting / n;
    *avg_turnaround = (float)total_turnaround / n;
}

int main() {
    struct Task jobs[] = {
        {1, 0, 4},
        {2, 2, 2},
        {3, 4, 1},
        {4, 6, 3},
        {5, 7, 5}
    };

    int n = sizeof(jobs) / sizeof(jobs[0]);

    float avg_waiting, avg_turnaround;
    calculate_waiting_turnaround_time(jobs, n, &avg_waiting, &avg_turnaround);

    printf("Average Waiting Time (SJF): %.2f\n", avg_waiting);
    printf("Average Turnaround Time (SJF): %.2f\n", avg_turnaround);

    return 0;
}



# Performance Evaluation of CPU Scheduling Algorithms

## Introduction
In operating systems, efficient CPU scheduling plays a crucial role in optimizing system performance and resource utilization. The allocation of CPU time between competing processes impacts factors such as response time, throughput, and overall system efficiency. This report evaluates the performance of two basic CPU scheduling algorithms: **First-Come First-Served (FCFS) and Shortest-Job-First (SJF)**, both of which are non-preemptive.

The primary goal of this report is to compare the operational characteristics, strengths, and limitations of FCFS and SJF. We aim to understand their performance under different workload conditions by analyzing metrics like average turnaround time and average waiting time.

## Methodology (Algorithms)
### First-Come First-Served (FCFS) – Non-Preemptive
FCFS is the simplest CPU scheduling algorithm. In this method, processes are executed in the order in which they arrive in the ready queue. The first process that arrives gets executed first, and so on, until all processes are completed. FCFS is non-preemptive, meaning once a process starts executing, it runs to completion before another process can begin.

#### Characteristics:
- Simple and easy to implement
- Can lead to high average waiting times when longer jobs arrive before shorter ones

### Shortest-Job-First (SJF) – Non-Preemptive
SJF is a CPU scheduling algorithm that selects the process with the shortest burst time (the time required to complete its execution) to execute first. In this method, processes with shorter burst times are given priority over longer ones, which tends to reduce average waiting times.

#### Characteristics:
- Tends to minimize waiting time by prioritizing shorter jobs
- More complex to implement, as it requires knowledge of the burst time for each process
- Can lead to starvation for longer processes if shorter ones keep arriving

## Results
Our experimental results showed the following metrics for each algorithm:

- FCFS:
  - Average Waiting Time: 1.6 ms
  - Average Turnaround Time: 4.6 ms
- SJF:
  - Average Waiting Time: 1.4 ms
  - Average Turnaround Time: 4.4 ms
    
SJF consistently performed better than FCFS in terms of both average waiting time and turnaround time. The efficiency of SJF is evident in its ability to minimize delays for shorter processes by prioritizing them, thus improving overall system performance.

## Conclusions
Through the comparison of **First-Come First-Served (FCFS) and Shortest-Job-First (SJF)** algorithms, we concluded the following:

- **FCFS** is simple and easy to implement, but it often results in higher average waiting times, especially when long jobs arrive earlier in the queue. This is due to its inherent limitation of executing tasks in the order they arrive, without considering their burst times.

- **SJF**, although more complex, generally leads to better performance by reducing average waiting times. It prioritizes shorter jobs, reducing delays for smaller processes. However, its effectiveness depends on the burst times and arrival order of processes, and it can lead to starvation of longer jobs.

In our case, SJF showed a clear advantage over FCFS, with an average waiting time of 1.4 ms and an average turnaround time of 4.4 ms, as compared to FCFS’s 1.6 ms waiting time and 4.6 ms turnaround time.

Ultimately, the choice between FCFS and SJF depends on the system's specific needs, such as whether simplicity or efficiency is prioritized, as well as the characteristics of the workload.

#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int main() {
    pid_t pid;

    // Create a new process using fork()
    pid = fork();

    // If fork() returns a negative value, it means the process creation failed
    if (pid < 0) {
        perror("Fork failed");
        return 1;
    }

    // If fork() returns 0, it means we are executing the child process
    if (pid == 0) {
        printf("I am the child process with PID: %d\n", getpid());
    }
    // If fork() returns a positive value, it means we are executing the parent process
    else {
        printf("I am the parent process with PID: %d and my child has PID: %d\n", getpid(), pid);
        wait(NULL);  // Wait for the child process to finish
        printf("Child process has finished\n");
    }

    return 0;
}

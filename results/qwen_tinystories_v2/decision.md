# Stage decision

The second-dataset replication is complete, but its scientific promotion gate failed. On all 21,990 examples in the official TinyStories validation split, task-selected polar magnitude3+phase5 increased last-token NLL by 0.005245, while the larger block-GPTQ real4 control increased it by 0.002479. Polar therefore did not reproduce its WikiText-2 advantage over block-GPTQ.

The pipeline stops here under the predeclared rule to stop when cross-dataset evidence invalidates the general superiority claim. The remaining scheduled checkpoint, backend, and kernel stages would characterize engineering behavior but cannot repair the failed replication without changing the hypothesis or method.

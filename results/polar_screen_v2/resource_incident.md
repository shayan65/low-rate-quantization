# Unintended full-run incident

During verification of the full-run guard, the script upload failed because of a connection reset. The following SSH command still executed the older unguarded launcher under the separate `results/polar_guard_check` root. This was an agent error and violated the user's small-test resource instruction.

The unintended first job, hybrid_mag4phase4_seed0, completed in 2,485.54 seconds (41.4 minutes); the following job was interrupted after approximately 233 seconds (3.9 minutes) according to its last recorded status. Total unintended GPU work was therefore approximately 45 minutes, plus any unsaved progress. These jobs are not part of the 12-job sample screen or its statistical table. All relevant launcher/training processes were stopped; GPU idle was observed at 0% utilization / 9 MiB allocation.

The launcher was subsequently uploaded successfully and verified to refuse full runs by default. A second guard was added to the full-data training command itself; it refuses to allocate a model unless an explicit full-run flag is present. The full launcher also requires a passing sample-screen report. The current sample screen failed its gate. No further full runs are authorized by the current resource policy.

Future deployment checks must inspect transfer success before issuing dependent commands and verify a script's guard before giving it any real training invocation. Connection drops must stop the deployment sequence rather than fall through to an older installed version.

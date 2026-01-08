# NUMA Performance Analysis with tokio-metrics

## Objective

Use tokio-metrics to determine if NUMA-related task migration and work stealing are causing performance degradation in the iris-mpc CPU implementation. This analysis will inform whether the dual-runtime NUMA optimization is worth implementing.

## Hypothesis

If NUMA node boundaries are causing performance issues, we should observe:
- **High task steal rates** during compute-intensive operations
- **Increased task latency** correlating with steal events
- **Uneven worker utilization** suggesting cross-NUMA migration overhead

## Key Metrics to Collect

### Runtime-Level Metrics (tokio-metrics)
- `total_steal_count` - Total work steals across runtime
- `total_steal_operations` - Number of steal attempts  
- `mean_poll_duration` - Average task execution time
- `total_poll_duration` - Total time spent polling tasks
- `total_slow_poll_duration` - Time spent in slow polls (>1ms)
- `workers_count` - Active worker threads
- `total_polls` - Total task polls for rate calculations

### Application-Specific Metrics
- Eye-specific task completion rates
- Search operation timing by orientation
- Cross-eye operation coordination timing

## Implementation Plan

### 1. Dependencies

Add to `Cargo.toml`:
```toml
[dependencies]
tokio-metrics = "0.3"
libc = "0.2"  # For NUMA node detection
```

### 2. Instrumentation Points

#### 2.1 Runtime Monitor Setup

**File: `iris-mpc-cpu/src/execution/hawk_main.rs`**

**Location: `HawkActor::new()` method**
```rust
// Add after network setup, around line 480
let runtime_monitor = tokio_metrics::RuntimeMonitor::new(&tokio::runtime::Handle::current());

// Spawn metrics collection task
tokio::spawn(async move {
    let mut interval = tokio::time::interval(Duration::from_secs(1));
    loop {
        interval.tick().await;
        if let Some(metrics) = runtime_monitor.intervals().next().await {
            // Emit core tokio metrics
            metrics::gauge!("tokio_workers_count").set(metrics.workers_count as f64);
            metrics::gauge!("tokio_total_steal_count").set(metrics.total_steal_count as f64);
            metrics::gauge!("tokio_total_steal_operations").set(metrics.total_steal_operations as f64);
            metrics::histogram!("tokio_mean_poll_duration_nanos").record(metrics.mean_poll_duration.as_nanos() as f64);
            metrics::histogram!("tokio_total_poll_duration_nanos").record(metrics.total_poll_duration.as_nanos() as f64);
            metrics::histogram!("tokio_total_slow_poll_duration_nanos").record(metrics.total_slow_poll_duration.as_nanos() as f64);
            metrics::gauge!("tokio_total_polls").set(metrics.total_polls as f64);
            
            // Calculate derived metrics
            let steal_rate = if metrics.total_polls > 0 {
                metrics.total_steal_operations as f64 / metrics.total_polls as f64
            } else { 0.0 };
            metrics::histogram!("tokio_steal_rate").record(steal_rate);
            
            // NUMA performance indicators
            let slow_poll_ratio = if metrics.total_poll_duration.as_nanos() > 0 {
                metrics.total_slow_poll_duration.as_nanos() as f64 / metrics.total_poll_duration.as_nanos() as f64
            } else { 0.0 };
            metrics::histogram!("tokio_slow_poll_ratio").record(slow_poll_ratio);
            
            // Red flag detection: steal increase + disproportionate slow poll increase
            metrics::histogram!("tokio_steal_vs_slow_poll_correlation").record(steal_rate * slow_poll_ratio);
            
            // NUMA node tracking
            let current_numa_node = unsafe { 
                libc::sched_getcpu() / 16  // Assuming 16 cores per NUMA node, adjust as needed
            };
            metrics::gauge!("current_numa_node").set(current_numa_node as f64);
        }
    }
});
```

#### 2.2 Job-Level Instrumentation

**File: `iris-mpc-cpu/src/execution/hawk_main.rs`**

**Location: `handle_job()` function, around line 1624**
```rust
async fn handle_job(
    hawk_actor: &mut HawkActor,
    sessions: &mut SessionGroups,
    request: HawkRequest,
) -> Result<HawkResult> {
    let job_start = Instant::now();
    let start_numa_node = unsafe { libc::sched_getcpu() };
    
    tracing::info!("Processing an Hawk job…");
    let now = Instant::now();

    // ... existing code until do_search ...

    // Before parallel search operations
    let search_start = Instant::now();
    let search_start_steals = get_current_steal_count(); // Helper function to add
    
    let ((search_normal, matches_normal), (_, matches_mirror)) = try_join!(
        instrument_search_task("normal", do_search(Orientation::Normal)),
        instrument_search_task("mirror", do_search(Orientation::Mirror)),
    )?;
    
    let search_end_steals = get_current_steal_count();
    let search_duration = search_start.elapsed();
    
    // Emit search-specific metrics
    metrics::histogram!("search_duration_ms").record(search_duration.as_millis() as f64);
    metrics::counter!("search_steals_during_operation").increment((search_end_steals - search_start_steals) as u64);
    
    let end_numa_node = unsafe { libc::sched_getcpu() };
    if start_numa_node != end_numa_node {
        metrics::counter!("job_numa_migrations").increment(1);
        tracing::warn!("Job migrated NUMA nodes: {} -> {}", start_numa_node, end_numa_node);
    }

    // ... rest of existing function
}
```

#### 2.3 Search Operation Instrumentation

**File: `iris-mpc-cpu/src/execution/hawk_main.rs`**

**Location: Add helper functions**
```rust
// Add these helper functions near the top of the file

async fn instrument_search_task<F, R>(operation_name: &'static str, future: F) -> R
where
    F: Future<Output = R>,
{
    let task_start = Instant::now();
    let start_numa = unsafe { libc::sched_getcpu() };
    
    let result = future.await;
    
    let task_duration = task_start.elapsed();
    let end_numa = unsafe { libc::sched_getcpu() };
    
    // Track task-specific metrics
    metrics::histogram!("search_task_duration_ms")
        .record(task_duration.as_millis() as f64, &[("operation", operation_name)]);
    
    if start_numa != end_numa {
        metrics::counter!("search_task_numa_migrations")
            .increment(1, &[("operation", operation_name)]);
    }
    
    metrics::histogram!("search_task_numa_node")
        .record(end_numa as f64, &[("operation", operation_name)]);
    
    result
}

fn get_current_steal_count() -> u64 {
    // This is a simplified version - in practice you'd need to maintain a reference
    // to the RuntimeMonitor or use a different approach to get current steal count
    // For now, we'll track this through our periodic metrics collection
    0 // Placeholder - implement based on your metrics collection strategy
}
```

#### 2.4 Eye-Specific Search Tracking

**File: `iris-mpc-cpu/src/execution/hawk_main.rs`**

**Location: Inside the `do_search` closure, around line 1650**
```rust
let do_search = async |orient| -> Result<_> {
    let eye_operation_start = Instant::now();
    let search_queries = &request.queries(orient);
    
    // Track which orientation is being processed
    let orientation_str = match orient {
        Orientation::Normal => "normal",
        Orientation::Mirror => "mirror",
    };
    
    let (luc_ids, request_types) = {
        // ... existing code ...
    };

    // Track search parameters
    metrics::gauge!("search_query_count")
        .set(search_queries.len() as f64, &[("orientation", orientation_str)]);

    // ... existing intra_results setup ...

    // Before the main search call
    let hnsw_search_start = Instant::now();
    
    let search_results = search::search::<SearchRotations>(
        sessions_search,
        search_queries,
        search_ids,
        search_params,
        NEIGHBORHOOD_MODE,
    )
    .await?;
    
    let hnsw_search_duration = hnsw_search_start.elapsed();
    metrics::histogram!("hnsw_search_duration_ms")
        .record(hnsw_search_duration.as_millis() as f64, &[("orientation", orientation_str)]);

    // ... rest of existing function ...
    
    let total_eye_duration = eye_operation_start.elapsed();
    metrics::histogram!("eye_operation_total_duration_ms")
        .record(total_eye_duration.as_millis() as f64, &[("orientation", orientation_str)]);

    Ok((search_results, match_result))
};
```

### 3. Analysis Criteria

#### Performance Indicators

**Good NUMA Performance (No optimization needed):**
- Steal rate < 0.05 (5% of tasks)
- Task migrations < 1% of operations
- Consistent poll duration across orientations
- Even distribution of work across worker threads

**Poor NUMA Performance (Optimization beneficial):**
- Steal rate > 0.15 (15% of tasks)  
- Task migrations > 5% of operations
- High correlation between steal events and increased latency
- Uneven NUMA node utilization

#### Key Correlations to Monitor

1. **Steal Rate vs Latency**: High steal rate correlating with increased `mean_poll_duration`
2. **Search Duration vs NUMA Migration**: Operations that migrate nodes taking longer
3. **Eye Operation Balance**: Significant difference in performance between left/right eye operations
4. **Worker Thread Distribution**: Concentration of work on specific NUMA nodes

### 4. Data Collection Strategy

#### Metrics Export

Configure your existing metrics system to export these new metrics:
```rust
// Add to your metrics configuration
metrics::describe_gauge!("tokio_workers_count", "Active tokio worker threads");
metrics::describe_histogram!("tokio_steal_rate", "Rate of work stealing operations");
metrics::describe_histogram!("search_duration_ms", "Duration of search operations");
metrics::describe_counter!("job_numa_migrations", "Jobs that migrated between NUMA nodes");
metrics::describe_histogram!("eye_operation_total_duration_ms", "Total eye operation duration");
```

#### Test Scenarios

1. **Baseline Load Test**: Normal production-like workload
2. **High Concurrency Test**: Stress test with many concurrent jobs
3. **Mixed Orientation Test**: Jobs with different normal/mirror ratios
4. **NUMA Binding Test**: Force bind to single NUMA node for comparison

### 5. Expected Timeline

- **Day 1 Morning (2-3 hours)**: Add tokio-metrics dependency and basic runtime monitoring
- **Day 1 Afternoon (3-4 hours)**: Instrument handle_job and search operations  
- **Day 2 Morning (2-3 hours)**: Add eye-specific tracking and helper functions
- **Day 2 Afternoon (2-3 hours)**: Test instrumentation and collect baseline data

### 6. Decision Matrix

Based on collected metrics:

| Steal Rate | Migration Rate | Latency Correlation | Decision |
|------------|----------------|-------------------|----------|
| < 5% | < 1% | None | Skip NUMA optimization |
| 5-15% | 1-5% | Weak | Consider simple thread pinning |
| > 15% | > 5% | Strong | Implement dual runtime approach |

### 7. Validation Commands

```bash
# Monitor metrics in real-time during test
curl localhost:9090/metrics | grep -E "(tokio_|search_|job_numa)"

# Check NUMA node distribution
numastat -p $(pgrep iris-mpc-cpu)

# Verify thread affinity
ps -eLo pid,tid,psr,comm | grep iris-mpc-cpu
```

This instrumentation will provide clear, quantitative evidence of whether NUMA optimization would provide meaningful performance benefits.
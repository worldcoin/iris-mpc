# NUMA Workload Separation: Tokio Runtime + Iris Worker Split

## Overview

This approach separates compute workloads by NUMA node based on their memory access patterns, avoiding the complexity of dual tokio runtimes while maximizing memory bandwidth utilization.

## Architecture

### Current Problem
- Tokio runtime threads and iris worker threads compete for memory bandwidth
- Iris store data can be accessed from wrong NUMA node during distance calculations
- Memory-bound iris operations (64% of CPU time) lack dedicated memory bandwidth

### Proposed Solution

```
NUMA Node 0: Tokio Runtime (36% CPU, Lightweight)
├── Graph topology (VectorId connections only)
├── HNSW traversal (pointer chasing)
├── Network sessions & ABY3 coordination  
├── Job orchestration & result aggregation
└── Cross-NUMA task dispatch

NUMA Node 1: Iris Workers + Data (64% CPU, Memory-Intensive)
├── Iris store allocation (iris codes & masks)
├── Distance computations (high memory bandwidth)
├── ABY3 crypto operations
└── NUMA-local data processing
```

### Data Flow
1. **Graph traversal** (NUMA 0): Find candidate VectorIds through HNSW
2. **Task dispatch**: Send VectorIds + query data to iris workers 
3. **Distance computation** (NUMA 1): Fetch iris data, compute distances locally
4. **Result return**: Send distance results back to tokio runtime

## Key Benefits

- **Dedicated memory bandwidth**: Each workload gets full NUMA node bandwidth
- **Cache optimization**: Each node optimized for its access patterns
- **No architectural complexity**: Single tokio runtime with pinned worker pools
- **Minimal data transfer**: Only VectorIds and results cross NUMA boundaries

## Implementation Details

### 1. Tokio Runtime Configuration

**File: `iris-mpc-cpu/src/execution/hawk_main.rs`**

**Location: `HawkActor::new()` method initialization**

```rust
// Configure tokio runtime to pin to NUMA node 0
pub fn configure_numa_runtime() -> tokio::runtime::Runtime {
    use libc::{cpu_set_t, sched_setaffinity, CPU_SET, CPU_ZERO, pthread_self};
    
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(get_numa_node_core_count(0))  // Only cores from NUMA node 0
        .thread_name("tokio-numa0")
        .on_thread_start(move || {
            pin_thread_to_numa_node(0);
            tracing::info!("Tokio worker pinned to NUMA node 0, core: {}", unsafe { libc::sched_getcpu() });
        })
        .build()
        .expect("Failed to create NUMA-aware tokio runtime")
}

fn pin_thread_to_numa_node(numa_node: usize) {
    unsafe {
        let mut cpu_set: cpu_set_t = std::mem::zeroed();
        CPU_ZERO(&mut cpu_set);
        
        // Get cores for this NUMA node (assuming 16 cores per node, adjust as needed)
        let cores_per_node = 16;
        let start_core = numa_node * cores_per_node;
        let end_core = start_core + cores_per_node;
        
        for core in start_core..end_core {
            CPU_SET(core, &mut cpu_set);
        }
        
        let result = sched_setaffinity(0, std::mem::size_of::<cpu_set_t>(), &cpu_set);
        if result != 0 {
            eprintln!("Failed to set CPU affinity for NUMA node {}: {}", numa_node, result);
        }
    }
}
```

### 2. Iris Store NUMA Allocation

**File: `iris-mpc-cpu/src/hawkers/shared_irises.rs`**

**Location: `SharedIrises<I>::new()` method**

```rust
impl<I: Clone> SharedIrises<I> {
    // Add NUMA-aware constructor
    pub fn new_numa_allocated(
        points: HashMap<VectorId, I>,
        empty_iris: I,
        numa_node: usize,
    ) -> Self {
        // Set NUMA memory policy before allocation
        Self::set_numa_memory_policy(numa_node);
        
        let mut storage = Self::new(points, empty_iris);
        
        // Pre-allocate with NUMA policy in effect
        storage.points.reserve(INITIAL_CAPACITY);
        
        tracing::info!("Iris store allocated on NUMA node {}", numa_node);
        storage
    }
    
    fn set_numa_memory_policy(numa_node: usize) {
        use libc::{c_void, c_ulong};
        
        unsafe {
            // Set memory policy to allocate on specific NUMA node
            let numa_mask: c_ulong = 1 << numa_node;
            let result = libc::syscall(
                libc::SYS_set_mempolicy,
                2, // MPOL_BIND
                &numa_mask as *const c_ulong as *const c_void,
                std::mem::size_of::<c_ulong>() * 8,
            );
            
            if result != 0 {
                eprintln!("Failed to set NUMA memory policy for node {}: {}", numa_node, result);
            }
        }
    }
}
```

### 3. Iris Worker Pool Configuration

**File: `iris-mpc-cpu/src/execution/hawk_main/iris_worker.rs`**

**Location: `init_workers()` function**

```rust
pub fn init_workers_numa_pinned(
    store_id: StoreId,
    storage: Aby3SharedIrisesRef,
    numa_config: NumaConfig,
) -> IrisPoolHandle {
    let numa_node = numa_config.iris_worker_node; // Should be 1
    let core_count = get_numa_node_core_count(numa_node);
    
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(core_count)
        .thread_name(move |idx| format!("iris-worker-numa{}-{}-{}", numa_node, store_id, idx))
        .start_handler(move |_| {
            pin_thread_to_numa_node(numa_node);
            tracing::info!("Iris worker pinned to NUMA node {}, core: {}", numa_node, unsafe { libc::sched_getcpu() });
        })
        .build()
        .expect("Failed to create NUMA-pinned iris worker pool");

    IrisPoolHandle { pool, storage }
}

pub struct NumaConfig {
    pub tokio_runtime_node: usize,  // 0
    pub iris_worker_node: usize,    // 1
    pub iris_store_node: usize,     // 1
}
```

### 4. HawkActor Integration

**File: `iris-mpc-cpu/src/execution/hawk_main.rs`**

**Location: `HawkActor::new()` method, around line 440**

```rust
impl HawkActor {
    pub async fn new(mut args: HawkArgs) -> Result<Self> {
        // Configure NUMA layout
        let numa_config = NumaConfig {
            tokio_runtime_node: 0,
            iris_worker_node: 1, 
            iris_store_node: 1,
        };
        
        // ... existing network setup ...

        // Create NUMA-aware iris stores  
        let iris_store: BothEyes<Aby3SharedIrises> = [StoreId::Left, StoreId::Right].map(|side| {
            load_iris_store_with_numa(side, &args, numa_config.iris_store_node)
        });
        
        let iris_store = iris_store.map(SharedIrises::to_arc);
        
        // Create NUMA-pinned worker pools
        let workers_handle = [StoreId::Left, StoreId::Right]
            .map(|side| init_workers_numa_pinned(side, iris_store[side].clone(), numa_config));

        // ... rest of initialization ...
        
        tracing::info!("NUMA workload separation initialized: Tokio=Node{}, IrisWorkers=Node{}", 
                      numa_config.tokio_runtime_node, numa_config.iris_worker_node);
        
        Ok(Self {
            args,
            searcher,
            prf_key: None,
            loader_db_size: args.db_size,
            iris_store,
            graph_store,
            workers_handle,
            // ... rest of fields ...
        })
    }
}

fn load_iris_store_with_numa(side: StoreId, args: &HawkArgs, numa_node: usize) -> Aby3SharedIrises {
    // Set NUMA policy before loading
    SharedIrises::set_numa_memory_policy(numa_node);
    
    // Load iris store data - will allocate on specified NUMA node
    let store = if let Some(db_url) = &args.db_url {
        load_from_database(db_url, side)
    } else {
        Aby3SharedIrises::default()
    };
    
    tracing::info!("Loaded iris store for {:?} on NUMA node {}", side, numa_node);
    store
}
```

### 5. Cross-NUMA Task Coordination

**File: `iris-mpc-cpu/src/execution/hawk_main.rs`**

**Location: `handle_job()` method, around line 1650**

```rust
// Add NUMA-aware task dispatching
async fn handle_job(
    hawk_actor: &mut HawkActor,
    sessions: &mut SessionGroups,
    request: HawkRequest,
) -> Result<HawkResult> {
    // Track NUMA performance
    let job_start_numa = unsafe { libc::sched_getcpu() };
    metrics::histogram!("job_start_numa_node").record(job_start_numa as f64);
    
    tracing::info!("Processing Hawk job on NUMA node {}", job_start_numa);
    
    // ... existing code until search operations ...
    
    // The search operations will naturally dispatch to NUMA node 1 workers
    // through the existing IrisPoolHandle mechanism
    let ((search_normal, matches_normal), (_, matches_mirror)) = try_join!(
        do_search(Orientation::Normal),
        do_search(Orientation::Mirror),
    )?;
    
    // Track job completion NUMA node
    let job_end_numa = unsafe { libc::sched_getcpu() };
    if job_start_numa != job_end_numa {
        metrics::counter!("job_numa_migrations").increment(1);
        tracing::warn!("Job migrated from NUMA {} to {}", job_start_numa, job_end_numa);
    }
    
    // ... rest of existing function ...
}
```

## Implementation Timeline

### Day 1: Core Infrastructure (6-8 hours)

**Morning (3-4 hours):**
- Implement `pin_thread_to_numa_node()` helper function
- Add `NumaConfig` structure
- Modify tokio runtime creation with NUMA pinning

**Afternoon (3-4 hours):** 
- Implement NUMA-aware iris store allocation
- Modify `init_workers()` for NUMA pinning
- Add basic NUMA monitoring metrics

### Day 2: Integration and Testing (6-8 hours)

**Morning (3-4 hours):**
- Integrate NUMA configuration into `HawkActor::new()`
- Test basic functionality with NUMA separation
- Validate thread/memory pinning with system tools

**Afternoon (3-4 hours):**
- Performance testing and tuning
- Add comprehensive NUMA performance metrics
- Document configuration and verify improvements

## Configuration

### Environment Variables
```bash
# Optional: Override default NUMA configuration
export IRIS_MPC_NUMA_TOKIO_NODE=0
export IRIS_MPC_NUMA_WORKER_NODE=1
export IRIS_MPC_NUMA_MEMORY_NODE=1
```

### System Requirements
- Multi-NUMA system (2+ NUMA nodes)
- Sufficient cores per NUMA node for workload separation
- Linux system with NUMA support enabled

## Success Metrics

### Performance Indicators
- **Memory bandwidth utilization**: Each NUMA node >80% local access
- **Cross-NUMA traffic reduction**: <10% of memory access cross-node
- **Task execution latency**: Reduced variance in iris worker task times
- **Overall throughput**: 10-30% improvement in job processing rate

### Validation Commands
```bash
# Verify thread pinning
ps -eLo pid,tid,psr,comm | grep iris-mpc

# Monitor NUMA memory usage
numastat -p $(pgrep iris-mpc)

# Check memory bandwidth utilization  
perf stat -e node-loads,node-stores,node-load-misses,node-store-misses ./iris-mpc

# Verify CPU distribution
top -H -p $(pgrep iris-mpc)
```

## Risk Mitigation

### Potential Issues
1. **Cross-NUMA coordination overhead**: Monitor with metrics
2. **Uneven workload distribution**: Adjust core allocation per node
3. **Memory allocation failures**: Fallback to interleaved policy
4. **System portability**: Graceful degradation on single-NUMA systems

### Fallback Strategy
- Feature flag to disable NUMA optimizations
- Automatic detection of NUMA topology
- Graceful fallback to current behavior if NUMA APIs fail

This approach delivers significant NUMA optimization benefits with minimal architectural complexity, focusing on memory bandwidth optimization where it matters most.

## Hugepage Integration for Iris Workers

### Overview

Combine NUMA workload separation with explicit hugepage allocation for the iris worker pool. This provides triple optimization: NUMA locality, reduced TLB misses, and dedicated memory bandwidth for the memory-bound iris operations.

### Architecture Enhancement

```
NUMA Node 0: Tokio Runtime (Standard 4KB pages)
├── Graph topology & traversal
├── Network sessions & coordination  
├── Job orchestration
└── Small allocations, buffers

NUMA Node 1: Iris Workers (Explicit 2MB Hugepages)
├── Iris store allocation (hugepage-backed)
├── Distance computations (hugepage memory access)
├── ABY3 crypto operations 
└── Large memory operations optimized
```

### Kubernetes/EKS Configuration

#### Node-Level Hugepage Setup

**EKS Node Group Configuration:**
```yaml
# In EKS node group user data or launch template
#!/bin/bash
# Reserve hugepages on NUMA node 1 only
echo 512 > /sys/devices/system/node/node1/hugepages/hugepages-2048kB/nr_hugepages

# Verify hugepage allocation
hugeadm --list-all-mounts
cat /proc/meminfo | grep Huge

# Mount hugepage filesystem for application use
mkdir -p /mnt/hugepages-2MB
mount -t hugetlbfs -o pagesize=2M none /mnt/hugepages-2MB
```

**Pod Resource Configuration:**
```yaml
# In your Kubernetes deployment
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: iris-mpc-cpu
    resources:
      requests:
        memory: "4Gi"
        hugepages-2Mi: "2Gi"  # For iris store allocation
      limits:
        memory: "8Gi" 
        hugepages-2Mi: "2Gi"
    volumeMounts:
    - name: hugepages
      mountPath: /mnt/hugepages-2MB
  volumes:
  - name: hugepages
    emptyDir:
      medium: HugePages-2Mi
  nodeSelector:
    node.kubernetes.io/instance-type: "m5.4xlarge"  # NUMA-capable instance
```

### Implementation Details

#### 1. Hugepage Allocation Interface

**File: `iris-mpc-cpu/src/execution/hawk_main/numa_allocator.rs` (new file)**

```rust
use std::ptr;
use std::alloc::{GlobalAlloc, Layout};
use std::sync::Mutex;

pub struct NumaHugepageAllocator {
    hugepage_fd: Mutex<Option<std::os::unix::io::RawFd>>,
    hugepage_base: Mutex<Option<*mut u8>>,
    numa_node: usize,
}

impl NumaHugepageAllocator {
    pub fn new(numa_node: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let allocator = Self {
            hugepage_fd: Mutex::new(None),
            hugepage_base: Mutex::new(None),
            numa_node,
        };
        
        allocator.initialize_hugepages()?;
        Ok(allocator)
    }
    
    fn initialize_hugepages(&self) -> Result<(), Box<dyn std::error::Error>> {
        use std::ffi::CString;
        use std::os::unix::io::RawFd;
        
        // Open hugepage mount point (configured by Kubernetes)
        let hugepage_path = CString::new("/mnt/hugepages-2MB/iris-store")?;
        
        unsafe {
            let fd: RawFd = libc::open(
                hugepage_path.as_ptr(),
                libc::O_CREAT | libc::O_RDWR,
                0o600,
            );
            
            if fd == -1 {
                return Err("Failed to open hugepage file".into());
            }
            
            // Allocate hugepage region for iris store (e.g., 1GB)
            let hugepage_size = 1024 * 1024 * 1024; // 1GB
            let mapped_region = libc::mmap(
                ptr::null_mut(),
                hugepage_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            
            if mapped_region == libc::MAP_FAILED {
                libc::close(fd);
                return Err("Failed to mmap hugepage region".into());
            }
            
            // Bind to specific NUMA node
            self.bind_memory_to_numa_node(mapped_region, hugepage_size)?;
            
            *self.hugepage_fd.lock().unwrap() = Some(fd);
            *self.hugepage_base.lock().unwrap() = Some(mapped_region as *mut u8);
            
            tracing::info!("Hugepage allocator initialized: 1GB on NUMA node {}", self.numa_node);
        }
        
        Ok(())
    }
    
    fn bind_memory_to_numa_node(&self, ptr: *mut libc::c_void, size: usize) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            let numa_mask: libc::c_ulong = 1 << self.numa_node;
            let result = libc::syscall(
                libc::SYS_mbind,
                ptr,
                size,
                2, // MPOL_BIND
                &numa_mask as *const libc::c_ulong,
                std::mem::size_of::<libc::c_ulong>() * 8,
                0, // flags
            );
            
            if result != 0 {
                return Err(format!("Failed to bind hugepage memory to NUMA node {}", self.numa_node).into());
            }
        }
        Ok(())
    }
}

// Thread-local allocator for iris worker threads
thread_local! {
    static NUMA_HUGEPAGE_ALLOCATOR: Option<NumaHugepageAllocator> = None;
}

pub fn set_thread_hugepage_allocator(numa_node: usize) -> Result<(), Box<dyn std::error::Error>> {
    NUMA_HUGEPAGE_ALLOCATOR.with(|allocator| {
        let new_allocator = NumaHugepageAllocator::new(numa_node)?;
        *allocator.borrow_mut() = Some(new_allocator);
        Ok(())
    })
}
```

#### 2. Iris Store Hugepage Integration

**File: `iris-mpc-cpu/src/hawkers/shared_irises.rs`**

**Location: Add to `SharedIrises<I>` implementation**

```rust
impl<I: Clone> SharedIrises<I> {
    pub fn new_numa_hugepage_allocated(
        points: HashMap<VectorId, I>,
        empty_iris: I,
        numa_node: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize hugepage allocator for this thread
        crate::execution::hawk_main::numa_allocator::set_thread_hugepage_allocator(numa_node)?;
        
        // Create the iris store - allocations will now use hugepages
        let mut storage = Self::new(points, empty_iris);
        
        // Pre-allocate with hugepage backing
        let initial_capacity = std::env::var("IRIS_STORE_INITIAL_CAPACITY")
            .unwrap_or_else(|_| "100000".to_string())
            .parse::<usize>()
            .unwrap_or(100000);
            
        storage.points.reserve(initial_capacity);
        
        tracing::info!(
            "Iris store allocated with hugepages on NUMA node {}, capacity: {}", 
            numa_node, initial_capacity
        );
        
        Ok(storage)
    }
    
    // Override allocator for iris store operations
    pub fn insert_with_hugepage_alloc(&mut self, id: VectorId, version: VersionId, iris: I) {
        // This will use the thread-local hugepage allocator
        self.insert(id, version, iris);
    }
}
```

#### 3. Iris Worker Thread Initialization

**File: `iris-mpc-cpu/src/execution/hawk_main/iris_worker.rs`**

**Location: Modify `init_workers()` function**

```rust
pub fn init_workers_numa_hugepages(
    store_id: StoreId,
    storage: Aby3SharedIrisesRef,
    numa_config: NumaConfig,
) -> Result<IrisPoolHandle, Box<dyn std::error::Error>> {
    let numa_node = numa_config.iris_worker_node; // Should be 1
    let core_count = get_numa_node_core_count(numa_node);
    
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(core_count)
        .thread_name(move |idx| format!("iris-worker-hugepage-numa{}-{}-{}", numa_node, store_id, idx))
        .start_handler(move |_| {
            // Pin thread to NUMA node
            pin_thread_to_numa_node(numa_node);
            
            // Initialize hugepage allocator for this worker thread
            if let Err(e) = crate::execution::hawk_main::numa_allocator::set_thread_hugepage_allocator(numa_node) {
                tracing::error!("Failed to initialize hugepage allocator: {}", e);
            } else {
                tracing::info!("Iris worker initialized with hugepages on NUMA node {}, core: {}", 
                              numa_node, unsafe { libc::sched_getcpu() });
            }
        })
        .build()
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    Ok(IrisPoolHandle { pool, storage })
}
```

#### 4. HawkActor Integration

**File: `iris-mpc-cpu/src/execution/hawk_main.rs`**

**Location: Modify `HawkActor::new()` method**

```rust
impl HawkActor {
    pub async fn new(mut args: HawkArgs) -> Result<Self> {
        // ... existing setup ...
        
        // Configure NUMA + hugepage layout
        let numa_config = NumaConfig {
            tokio_runtime_node: 0,      // Standard pages
            iris_worker_node: 1,        // Hugepages
            iris_store_node: 1,         // Hugepages
        };
        
        // Validate hugepage availability in Kubernetes environment
        Self::validate_kubernetes_hugepage_config()?;
        
        // Create NUMA + hugepage aware iris stores
        let iris_store: BothEyes<Aby3SharedIrises> = [StoreId::Left, StoreId::Right]
            .map(|side| load_iris_store_with_numa_hugepages(side, &args, numa_config.iris_store_node))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .unwrap();
        
        let iris_store = iris_store.map(SharedIrises::to_arc);
        
        // Create hugepage-enabled worker pools
        let workers_handle: BothEyes<IrisPoolHandle> = [StoreId::Left, StoreId::Right]
            .map(|side| init_workers_numa_hugepages(side, iris_store[side].clone(), numa_config))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .unwrap();
        
        tracing::info!("NUMA + Hugepage workload separation initialized");
        
        Ok(Self {
            // ... existing fields ...
            iris_store,
            workers_handle,
        })
    }
    
    fn validate_kubernetes_hugepage_config() -> Result<(), Box<dyn std::error::Error>> {
        // Check if hugepage mount is available
        let hugepage_mount = std::path::Path::new("/mnt/hugepages-2MB");
        if !hugepage_mount.exists() {
            return Err("Hugepage mount not available - check Kubernetes pod configuration".into());
        }
        
        // Verify hugepage availability
        let hugepage_info = std::fs::read_to_string("/proc/meminfo")?;
        if !hugepage_info.contains("HugePages_Total:") {
            return Err("No hugepages configured on system".into());
        }
        
        tracing::info!("Kubernetes hugepage configuration validated");
        Ok(())
    }
}

fn load_iris_store_with_numa_hugepages(
    side: StoreId, 
    args: &HawkArgs, 
    numa_node: usize
) -> Result<Aby3SharedIrises, Box<dyn std::error::Error>> {
    // Load iris store data with hugepage allocation
    let store = if let Some(db_url) = &args.db_url {
        // Load from database using hugepage allocator
        let mut store = SharedIrises::new_numa_hugepage_allocated(HashMap::new(), Default::default(), numa_node)?;
        
        // Load data into hugepage-backed storage
        load_from_database_to_hugepage_store(db_url, side, &mut store)?;
        store
    } else {
        SharedIrises::new_numa_hugepage_allocated(HashMap::new(), Default::default(), numa_node)?
    };
    
    tracing::info!("Loaded iris store for {:?} with hugepages on NUMA node {}", side, numa_node);
    Ok(store)
}
```

### Performance Monitoring

Add hugepage-specific metrics:

```rust
// Add to periodic metrics collection
let hugepage_stats = std::fs::read_to_string("/proc/meminfo")?;
if let Some(free_hugepages) = parse_hugepage_stat(&hugepage_stats, "HugePages_Free:") {
    metrics::gauge!("hugepages_free_2mb").set(free_hugepages as f64);
}
if let Some(total_hugepages) = parse_hugepage_stat(&hugepage_stats, "HugePages_Total:") {
    metrics::gauge!("hugepages_total_2mb").set(total_hugepages as f64);
}

// Track hugepage utilization per NUMA node
metrics::gauge!("hugepage_utilization_numa1").set(
    ((total_hugepages - free_hugepages) as f64 / total_hugepages as f64) * 100.0
);
```

### EKS Deployment Considerations

1. **Instance Types**: Use NUMA-capable instances (m5.4xlarge, c5.4xlarge, etc.)
2. **Node Group Configuration**: Pre-configure hugepages in user data
3. **Resource Limits**: Set appropriate hugepage requests/limits in pod spec
4. **Volume Mounts**: Ensure hugepage volume is correctly mounted
5. **Node Affinity**: Consider pinning pods to hugepage-configured nodes

This enhancement provides significant memory performance improvements for the iris worker pool while maintaining simplicity for the tokio runtime operations.

## Potential Drawbacks and Counter-Arguments

### 1. Reduced Effective Parallelism

**Problem**: Pinning workloads to separate NUMA nodes artificially constrains available cores.

- **Tokio Runtime**: Limited to ~16 cores on NUMA node 0, even if iris workers on node 1 are idle
- **Iris Workers**: Limited to ~16 cores on NUMA node 1, even if tokio tasks are waiting
- **Work Stealing Elimination**: Current tokio can steal work across all cores - this removes that flexibility

**Real-world impact**: During bursty workloads, one NUMA node might be saturated while the other is underutilized, creating artificial bottlenecks.

### 2. Pipeline Stall Inefficiency  

**Problem**: Tokio runtime often waits for iris worker results anyway.

Current flow:
```
Tokio Task -> Dispatch to Iris Worker -> Wait for Result -> Continue
```

**Issue**: If tokio threads are sitting idle waiting for iris computations, dedicating an entire NUMA node to them wastes resources. The 36% tokio CPU usage might include significant wait time, making NUMA node 0 underutilized.

**Alternative**: Let tokio threads help with iris work during idle periods rather than sitting pinned to separate cores.

### 3. Cross-NUMA Coordination Overhead

**Problem**: Every iris operation now requires cross-NUMA communication.

- **Data marshaling**: VectorIds and results must cross NUMA boundaries
- **Cache misses**: Result data allocated on NUMA node 1 accessed from node 0
- **Memory latency**: Cross-NUMA memory access is ~2x slower than local access
- **Synchronization overhead**: Additional coordination between workloads

**Measurement risk**: The coordination overhead might exceed the memory bandwidth benefits, especially for smaller operations.

### 4. Load Balancing Degradation

**Problem**: Workload imbalance creates idle resources.

**Scenarios where this fails:**
- **Light iris load**: NUMA node 1 cores sit idle while node 0 is busy
- **Heavy graph traversal**: NUMA node 0 saturated while node 1 waits
- **Mixed workloads**: Some jobs are graph-heavy, others iris-heavy - fixed allocation is suboptimal

**Current advantage**: Dynamic work stealing naturally balances load across all cores based on actual demand.

### 5. Batching Already Optimizes Memory Access

**Problem**: Existing optimizations might make NUMA separation unnecessary.

- **128-item batches**: Already provide good memory locality and bandwidth utilization  
- **NUMA realloc**: `numa_realloc()` already places data optimally
- **Cache-friendly patterns**: Batch processing likely already optimizes cache usage

**Question**: If memory bandwidth isn't actually the bottleneck (CPU not saturated), then NUMA optimization provides no benefit while adding complexity.

### 6. Kubernetes/EKS Practical Constraints

**Problem**: Production deployment complexity and reliability risks.

**EKS challenges:**
- **Node heterogeneity**: Not all instance types have predictable NUMA topology
- **Resource management**: Hugepage allocation conflicts with other workloads
- **Deployment complexity**: Requires node-level configuration management
- **Debugging difficulty**: NUMA issues are hard to diagnose in production
- **Rollback risk**: Failed NUMA configuration could break deployments

**Operational overhead**: The approach adds significant deployment and operational complexity for uncertain performance gains.

### 7. Premature Optimization Risk

**Problem**: Optimizing based on assumptions rather than measured bottlenecks.

**Missing data:**
- **Actual memory bandwidth utilization**: Is it really the constraint?
- **Cross-NUMA access patterns**: How much currently happens?  
- **Work stealing frequency**: Is it actually beneficial or harmful?
- **Wait time analysis**: How much time is spent waiting vs. computing?

**Better approach**: Measure first with tokio-metrics, then optimize only if data shows clear NUMA bottlenecks.

### 8. Implementation Complexity vs. Benefit Ratio

**Problem**: High implementation and maintenance cost for uncertain gains.

**Complexity added:**
- Custom memory allocators
- NUMA-aware thread pinning  
- Kubernetes resource management
- Cross-NUMA coordination logic
- Platform-specific optimizations
- Debugging and monitoring complexity

**Alternative**: Simple optimizations (better batching, memory pools, compiler optimizations) might provide similar benefits with much less complexity.

### Recommendation: Measure First, Optimize Later

Instead of implementing NUMA separation based on theoretical benefits:

1. **Use tokio-metrics** to identify actual bottlenecks
2. **Profile memory bandwidth** utilization with `perf` 
3. **Measure cross-NUMA access** patterns in current implementation
4. **Quantify work stealing** impact on performance

**If measurements show:**
- High cross-NUMA memory access (>20%)
- Memory bandwidth saturation (>80%)  
- Work stealing causing performance degradation
- Clear correlation between NUMA placement and latency

**Then** the complexity of workload separation is justified.

**If measurements show** good NUMA locality and efficient work stealing, focus optimization efforts elsewhere (algorithmic improvements, better data structures, etc.).

## Using tokio-metrics to Evaluate NUMA Optimization Need

### Key Metrics for NUMA Decision Making

#### 1. Work Stealing Pattern Analysis

```rust
// Collect baseline metrics during normal load
let baseline_metrics = runtime_monitor.intervals().next().await;

// Key indicators to track:
let steal_rate = baseline_metrics.total_steal_operations as f64 / baseline_metrics.total_polls as f64;
let steal_efficiency = if baseline_metrics.total_steal_operations > 0 {
    baseline_metrics.total_steal_count as f64 / baseline_metrics.total_steal_operations as f64
} else { 0.0 };

// Decision thresholds:
// steal_rate > 0.15 (15%) = High stealing, potential NUMA issue
// steal_rate < 0.05 (5%) = Low stealing, current approach working well
// steal_efficiency < 0.3 = Steals often fail, suggesting NUMA memory locality issues
```

**What to look for:**
- **High steal rate + low efficiency**: Tasks are frequently moved but often can't access their data efficiently
- **Steal rate spikes during iris operations**: Indicates iris work causes cross-NUMA migration
- **Consistent low steal rate**: Current work distribution is already optimal

#### 2. Task Duration Variance Analysis

```rust
// Track poll duration patterns during iris-heavy workloads
let slow_poll_ratio = metrics.total_slow_poll_duration.as_nanos() as f64 
    / metrics.total_poll_duration.as_nanos() as f64;

// Correlate with steal events
let steal_vs_slow_correlation = steal_rate * slow_poll_ratio;

// Decision thresholds:
// slow_poll_ratio > 0.3 = 30% of time spent in slow polls (>1ms)
// steal_vs_slow_correlation > 0.05 = Steals correlate with slow polls (NUMA issue)
// steal_vs_slow_correlation < 0.02 = No correlation (NUMA not the issue)
```

**What this reveals:**
- **High correlation**: When tasks steal work, they become slow (NUMA memory access penalty)
- **Low correlation**: Task slowness is not related to work stealing (other bottleneck)

#### 3. Worker Thread Utilization Patterns

```rust
// Track per-worker metrics if available, or infer from system tools
async fn analyze_worker_patterns() {
    // During iris-heavy operations, monitor:
    let worker_busy_count = metrics.busy_duration_total.as_nanos();
    let worker_idle_count = metrics.total_poll_duration.as_nanos() - worker_busy_count;
    
    // Check for uneven utilization using system tools
    let cpu_usage_per_core = get_per_core_cpu_usage(); // External system call
    let numa_node_0_usage = calculate_numa_node_usage(cpu_usage_per_core, 0);
    let numa_node_1_usage = calculate_numa_node_usage(cpu_usage_per_core, 1);
    
    let usage_imbalance = (numa_node_0_usage - numa_node_1_usage).abs() / 
                         (numa_node_0_usage + numa_node_1_usage);
    
    // Decision threshold:
    // usage_imbalance > 0.3 = One NUMA node significantly more utilized
}
```

#### 4. Memory Access Pattern Detection

```rust
// Combine tokio-metrics with system-level NUMA stats
async fn detect_numa_memory_patterns() {
    // Before job processing
    let numa_stats_before = read_numa_stats()?; // /proc/*/numa_maps or numastat
    
    // Process job with tokio-metrics collection
    let job_start = Instant::now();
    let metrics_before = runtime_monitor.intervals().next().await;
    
    // Process actual workload
    handle_job(hawk_actor, sessions, request).await?;
    
    let metrics_after = runtime_monitor.intervals().next().await;
    let numa_stats_after = read_numa_stats()?;
    
    // Calculate NUMA access patterns
    let cross_numa_access_ratio = calculate_cross_numa_ratio(numa_stats_before, numa_stats_after);
    let task_migration_during_job = metrics_after.total_steal_count - metrics_before.total_steal_count;
    
    // Decision indicators:
    // cross_numa_access_ratio > 0.2 = 20% of memory access crosses NUMA boundaries  
    // High task_migration_during_job + High cross_numa_access = NUMA optimization beneficial
}
```

### Specific Test Scenarios

#### Scenario 1: Baseline Performance Test
```rust
async fn baseline_numa_test() -> NumaTestResult {
    // Run moderate workload (60% capacity) 
    // Measure: steal_rate, slow_poll_ratio, cross_numa_access
    // Expected: Low values = current approach is efficient
    
    let test_duration = Duration::from_secs(300); // 5 minutes
    let metrics = collect_metrics_during_period(test_duration).await;
    
    NumaTestResult {
        steal_rate: calculate_steal_rate(&metrics),
        slow_poll_ratio: calculate_slow_poll_ratio(&metrics),
        recommendation: if metrics.steal_rate < 0.05 && metrics.slow_poll_ratio < 0.2 {
            "Skip NUMA optimization - current performance is good"
        } else {
            "Consider NUMA optimization - investigate further"
        }
    }
}
```

#### Scenario 2: Stress Test with Forced Work Stealing
```rust
async fn stress_test_numa_impact() -> NumaTestResult {
    // Saturate system to force work stealing
    // Push load to 100%+ capacity
    // Measure: How much does performance degrade with increased stealing?
    
    let high_concurrency_jobs = generate_high_load_jobs(200); // 200% normal load
    
    let stressed_metrics = collect_metrics_during_load(high_concurrency_jobs).await;
    
    // Look for disproportionate degradation
    let performance_degradation = stressed_metrics.mean_poll_duration.as_millis() as f64 / 
                                 baseline_metrics.mean_poll_duration.as_millis() as f64;
    
    if performance_degradation > 3.0 && stressed_metrics.steal_rate > 0.2 {
        "NUMA optimization likely beneficial - high steal rate causes significant slowdown"
    } else {
        "Current work stealing is efficient even under stress"
    }
}
```

#### Scenario 3: Iris-Specific Workload Analysis
```rust
async fn iris_workload_numa_test() -> NumaTestResult {
    // Run jobs with varying iris/graph ratios
    // Measure: Which operations cause steal spikes?
    
    let iris_heavy_jobs = generate_iris_heavy_jobs(); // Lots of distance computations
    let graph_heavy_jobs = generate_graph_heavy_jobs(); // Lots of HNSW traversal
    
    let iris_metrics = collect_metrics_during_jobs(iris_heavy_jobs).await;
    let graph_metrics = collect_metrics_during_jobs(graph_heavy_jobs).await;
    
    // Compare steal patterns
    if iris_metrics.steal_rate > graph_metrics.steal_rate * 2.0 {
        "Iris operations cause work stealing - NUMA separation beneficial"
    } else {
        "Steal rate consistent across workload types - other bottlenecks"
    }
}
```

### Decision Matrix Based on Metrics

| Steal Rate | Slow Poll Ratio | Cross-NUMA Access | Recommendation |
|------------|-----------------|-------------------|----------------|
| < 5% | < 20% | < 10% | **Skip NUMA optimization** - Current approach optimal |
| 5-15% | 20-40% | 10-25% | **Investigate further** - Run stress tests |
| > 15% | > 40% | > 25% | **Implement NUMA optimization** - Clear benefit expected |

### Quick Implementation for Data Collection

```rust
// Add this to handle_job() for immediate data collection
async fn handle_job_with_numa_metrics(/* ... */) -> Result<HawkResult> {
    let job_start_cpu = unsafe { libc::sched_getcpu() };
    let metrics_start = get_runtime_metrics_snapshot().await;
    
    // Run existing job logic
    let result = handle_job_original(hawk_actor, sessions, request).await;
    
    let job_end_cpu = unsafe { libc::sched_getcpu() };
    let metrics_end = get_runtime_metrics_snapshot().await;
    
    // Calculate job-specific NUMA impact
    let job_steal_count = metrics_end.total_steal_count - metrics_start.total_steal_count;
    let job_slow_polls = metrics_end.total_slow_poll_duration - metrics_start.total_slow_poll_duration;
    let cpu_migration = job_start_cpu != job_end_cpu;
    
    // Log decision-relevant data
    tracing::info!(
        "Job NUMA metrics: steals={}, slow_polls={}ms, cpu_migration={}, steal_rate={:.3}",
        job_steal_count,
        job_slow_polls.as_millis(),
        cpu_migration,
        job_steal_count as f64 / (metrics_end.total_polls - metrics_start.total_polls) as f64
    );
    
    // Export to metrics system for analysis
    metrics::histogram!("job_steal_count").record(job_steal_count as f64);
    metrics::histogram!("job_slow_poll_duration_ms").record(job_slow_polls.as_millis() as f64);
    metrics::counter!("job_cpu_migrations").increment(if cpu_migration { 1 } else { 0 });
    
    result
}
```

### Expected Results That Justify NUMA Optimization

- **Steal rate > 15%** during iris operations
- **Slow poll ratio > 40%** correlating with steals  
- **Cross-NUMA memory access > 25%** of total
- **Performance degradation > 3x** under stress with high steal rate
- **Clear correlation** between task migration and increased latency

**If you don't see these patterns**, the current approach is already NUMA-efficient and the optimization complexity isn't justified.
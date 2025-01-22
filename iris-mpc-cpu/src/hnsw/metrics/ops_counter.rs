use std::{
    collections::HashMap,
    fmt::Debug,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, RwLock,
    },
};
use tracing::{
    field::{Field, Visit},
    Event, Subscriber,
};
use tracing_subscriber::layer::{Context, Layer};

#[derive(Debug, Clone, Copy)]
pub enum Operation {
    EvaluateDistance,
    CompareDistance,
    OpenNode,
    LayerSearch,
}

/// The number of enum variants of `Operation`
const NUM_OPS: usize = 4;

impl Operation {
    pub const fn tag(&self) -> &'static str {
        match self {
            Operation::EvaluateDistance => "evaluate_distance",
            Operation::CompareDistance => "compare_distance",
            Operation::OpenNode => "open_node",
            Operation::LayerSearch => "layer_search",
        }
    }

    pub const fn id(&self) -> usize {
        *self as usize
    }
}

#[derive(Default)]
struct EventVisitor {
    // which event was encountered
    event: Option<usize>,

    // how much to increment the associated counter
    amount: Option<usize>,
}

impl Visit for EventVisitor {
    fn record_u64(&mut self, field: &Field, value: u64) {
        match field.name() {
            "event_type" => {
                self.event = Some(value as usize);
            }
            "increment_amount" => {
                self.amount = Some(value as usize);
            }
            _ => {}
        }
    }

    fn record_debug(&mut self, _field: &Field, _value: &dyn Debug) {}
}

pub enum OpCounter {
    Static {
        counter: StaticCounter,
    },
    Dynamic {
        counter: Box<dyn DynamicCounter + Send + Sync>,
    },
}

pub type OpCounters = [Vec<OpCounter>; NUM_OPS];

pub type ParamCounterRef<K> = Arc<RwLock<HashMap<K, AtomicUsize>>>;
pub type CounterRef = Arc<AtomicUsize>;

#[derive(Default)]
pub struct StaticCounter {
    counter: CounterRef,
}

impl StaticCounter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_counter(&self) -> CounterRef {
        self.counter.clone()
    }

    #[inline(always)]
    pub fn increment(&self, increment_amount: usize) {
        self.counter.fetch_add(increment_amount, Ordering::Relaxed);
    }
}

pub trait DynamicCounter {
    /// Access relevant data from current Event dynamically, and increment
    /// counter based on this data.
    fn increment(&self, increment_amount: usize, event: &Event<'_>);
}

#[derive(Default)]
pub struct OpCountersLayer {
    counters: Arc<OpCounters>,
}

impl OpCountersLayer {
    pub fn new_builder() -> OpCountersLayerBuilder {
        OpCountersLayerBuilder::default()
    }
}

#[derive(Default)]
pub struct OpCountersLayerBuilder {
    counters: OpCounters,
}

impl OpCountersLayerBuilder {
    pub fn register_static(mut self, counter: StaticCounter, operation: Operation) -> Self {
        self.counters
            .get_mut(operation.id())
            .unwrap()
            .push(OpCounter::Static { counter });
        self
    }

    pub fn register_dynamic<T: DynamicCounter + Send + Sync + 'static>(
        mut self,
        counter: T,
        operation: Operation,
    ) -> Self {
        self.counters
            .get_mut(operation.id())
            .unwrap()
            .push(OpCounter::Dynamic {
                counter: Box::new(counter),
            });
        self
    }

    pub fn init(self) -> OpCountersLayer {
        OpCountersLayer {
            counters: Arc::new(self.counters),
        }
    }
}

impl<S: Subscriber> Layer<S> for OpCountersLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = EventVisitor::default();
        event.record(&mut visitor);

        let operation_id = visitor.event.unwrap() as usize;
        let increment_amount = visitor.amount.unwrap_or(1);
        let counters = self
            .counters
            .get(operation_id)
            .expect("attempted to count using invalid operation id");
        for counter in counters {
            match counter {
                OpCounter::Static { counter: c } => {
                    c.increment(increment_amount);
                }
                OpCounter::Dynamic { counter: c } => {
                    c.increment(increment_amount, event);
                }
            }
        }
    }
}

pub trait KeyVisitor: Visit + Default {
    type Key: Eq + std::hash::Hash + Default;
    fn get_key(&self) -> Option<Self::Key>;
}

#[derive(Default)]
pub struct ParameterizedCounter<K: KeyVisitor> {
    counter_map:  ParamCounterRef<K::Key>,
    missing_keys: StaticCounter,
}

impl<K: KeyVisitor> ParameterizedCounter<K> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return references to parameterized counter map and missing keys counter
    pub fn get_counters(&self) -> (ParamCounterRef<K::Key>, CounterRef) {
        (self.counter_map.clone(), self.missing_keys.counter.clone())
    }
}

impl<K: KeyVisitor> DynamicCounter for ParameterizedCounter<K> {
    fn increment(&self, increment_amount: usize, event: &Event<'_>) {
        let mut visitor = K::default();
        event.record(&mut visitor);
        if let Some(key) = visitor.get_key() {
            let counters_read = self.counter_map.read().unwrap();
            if let Some(counter) = counters_read.get(&key) {
                counter.fetch_add(increment_amount, Ordering::Release);
            } else {
                drop(counters_read);
                let new_counter = AtomicUsize::new(increment_amount);
                self.counter_map.write().unwrap().insert(key, new_counter);
            }
        } else {
            self.missing_keys.increment(increment_amount);
        }
    }
}

// /// Tracing library Layer for counting detailed HNSW layer search operations
// pub struct VertexOpeningsLayer {
//     // Measure number of vertex openings for different lc and ef values
//     counters:     Counters,
//     missing_keys: Arc<AtomicUsize>,
// }

// impl VertexOpeningsLayer {
//     pub fn new() -> Self {
//         Self {
//             counters:     Counters::default(),
//             missing_keys: Arc::default(),
//         }
//     }

//     pub fn get_counters(&self) -> Counters {
//         self.counters.clone()
//     }
// }

// impl<S: Subscriber> Layer<S> for VertexOpeningsLayer {
//     fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
//         let mut op = EventVisitor::default();
//         event.record(&mut op);

//         const OPEN_NODE_EVENT: usize = Operation::OpenNode.id();
//         if let Some(OPEN_NODE_EVENT) = op.event {
//             let increment_amount = op.amount.unwrap_or(1);

//             let mut key_fields = VertexOpeningsParams::default();
//             event.record(&mut key_fields);

//             if let Some(key) = key_fields.get_key() {
//                 let counters_read = self.counters.read().unwrap();
//                 if let Some(counter) = counters_read.get(&key) {
//                     counter.fetch_add(increment_amount, Ordering::Release);
//                 } else {
//                     drop(counters_read);
//                     let new_counter = AtomicUsize::new(increment_amount);
//                     self.counters.write().unwrap().insert(key, new_counter);
//                 }
//             } else {
//                 self.missing_keys
//                     .fetch_add(increment_amount, Ordering::Release);
//             }
//         }
//     }
// }

pub type ParamVertexOpeningsCounter = ParameterizedCounter<VertexOpeningsParams>;

#[derive(Default)]
pub struct VertexOpeningsParams {
    lc: Option<u64>,
    ef: Option<u64>,
}

impl KeyVisitor for VertexOpeningsParams {
    type Key = (u64, u64);

    fn get_key(&self) -> Option<(u64, u64)> {
        match self {
            Self {
                lc: Some(lc),
                ef: Some(ef),
            } => Some((*lc, *ef)),
            _ => None,
        }
    }
}

impl Visit for VertexOpeningsParams {
    fn record_u64(&mut self, field: &Field, value: u64) {
        match field.name() {
            "lc" => {
                self.lc = Some(value);
            }
            "ef" => {
                self.ef = Some(value);
            }
            _ => {}
        }
    }
    fn record_debug(&mut self, _field: &Field, _value: &dyn Debug) {}
}

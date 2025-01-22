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
pub const NUM_OPS: usize = 4;
pub type OpCounters = [Vec<OpCounter>; NUM_OPS];

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
struct OpVisitor {
    // which operation was encountered
    id: Option<usize>,

    // how much to increment the associated counter
    amount: Option<usize>,
}

impl Visit for OpVisitor {
    fn record_u64(&mut self, field: &Field, value: u64) {
        match field.name() {
            "event_type" => {
                self.id = Some(value as usize);
            }
            "increment_amount" => {
                self.amount = Some(value as usize);
            }
            _ => {}
        }
    }

    fn record_debug(&mut self, _field: &Field, _value: &dyn Debug) {}
}

/// `OpCountersLayer` maintains a list of counters for each variant of the
/// `Operations` enum, which are each incremented when a callsite specifying
/// the associated operation ID is encountered.
#[derive(Default)]
pub struct OpCountersLayer {
    counters: Arc<OpCounters>,
}

impl OpCountersLayer {
    pub fn new() -> OpCountersLayerBuilder {
        OpCountersLayerBuilder::default()
    }
}

impl<S: Subscriber> Layer<S> for OpCountersLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = OpVisitor::default();
        event.record(&mut visitor);

        let operation_id = visitor.id.unwrap() as usize;
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

/// Enum representing two types of counters, static counters which count
/// operations unconditionally, and dynamic counters which respond to fields
/// associated with an event in order to increment one or more counters.
pub enum OpCounter {
    Static {
        counter: StaticCounter,
    },
    Dynamic {
        counter: Box<dyn DynamicCounter + Send + Sync>,
    },
}

pub type StaticCounterRef = Arc<AtomicUsize>;

#[derive(Default)]
pub struct StaticCounter {
    counter: StaticCounterRef,
}

impl StaticCounter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_counter(&self) -> StaticCounterRef {
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

pub type KeyedCounterRef<K> = Arc<RwLock<HashMap<K, AtomicUsize>>>;

pub trait KeyVisitor: Visit + Default {
    type Key: Eq + std::hash::Hash + Default;
    fn get_key(&self) -> Option<Self::Key>;
}

/// Dynamic counter type which keeps separate counters for different key values
/// derived from Event field data.  The generic type `K` implementing the
/// `KeyVisitor` trait provides the logic for recording the fields of an Event
/// and producing a `KeyVisitor::Key` value used as the key for a `HashMap` of
/// `AtomicUsize` counters.
///
/// Events which don't properly correspond with a key(indicated by a return
/// value of `None` from the `get_key` function) are recorded separately in a
/// `missing_keys` counter.
#[derive(Default)]
pub struct KeyedCounter<K: KeyVisitor> {
    counter_map:  KeyedCounterRef<K::Key>,
    missing_keys: StaticCounter,
}

impl<K: KeyVisitor> KeyedCounter<K> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return references to counter map and missing keys counter
    pub fn get_counters(&self) -> (KeyedCounterRef<K::Key>, StaticCounterRef) {
        (self.counter_map.clone(), self.missing_keys.counter.clone())
    }
}

impl<K: KeyVisitor> DynamicCounter for KeyedCounter<K> {
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

/// An instance of `KeyedCounter` which can be used to count
/// `OpenVertex` events according to their `lc` and `ef` parameters.
pub type KeyedVertexOpeningsCounter = KeyedCounter<VertexOpeningsKeys>;

#[derive(Default)]
pub struct VertexOpeningsKeys {
    lc: Option<u64>,
    ef: Option<u64>,
}

impl KeyVisitor for VertexOpeningsKeys {
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

impl Visit for VertexOpeningsKeys {
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

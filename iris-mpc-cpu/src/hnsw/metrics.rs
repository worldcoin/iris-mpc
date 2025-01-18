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

pub type OpCounters = [AtomicUsize; NUM_OPS];

pub struct CounterLayer {
    counters: Arc<OpCounters>,
    // callsite_target: Option<String>,
}

impl CounterLayer {
    pub fn new() -> Self {
        Self {
            counters: Arc::new(OpCounters::default()),
        }
    }

    pub fn get_counters(&self) -> Arc<OpCounters> {
        self.counters.clone()
    }
}

impl<S: Subscriber> Layer<S> for CounterLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = EventVisitor::default();
        event.record(&mut visitor);

        let event_id = visitor.event.unwrap() as usize;
        let increment_amount = visitor.amount.unwrap_or(1);
        self.counters
            .get(event_id)
            .expect("attempted to count using invalid counter id")
            .fetch_add(increment_amount as usize, Ordering::Relaxed);
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

// #[derive(Default)]
pub type Counters = Arc<RwLock<HashMap<(u64, u64), AtomicUsize>>>;

/// Tracing library Layer for counting detailed HNSW layer search operations
pub struct VertexOpeningsLayer {
    // Measure number of vertex openings for different lc and ef values
    counters:     Counters,
    missing_keys: Arc<AtomicUsize>,
}

impl VertexOpeningsLayer {
    pub fn new() -> Self {
        Self {
            counters:     Counters::default(),
            missing_keys: Arc::default(),
        }
    }

    pub fn get_counters(&self) -> Counters {
        self.counters.clone()
    }
}

impl<S: Subscriber> Layer<S> for VertexOpeningsLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut op = EventVisitor::default();
        event.record(&mut op);

        const OPEN_NODE_EVENT: usize = Operation::OpenNode.id();
        if let Some(OPEN_NODE_EVENT) = op.event {
            let increment_amount = op.amount.unwrap_or(1);

            let mut key_fields = KeyFields::default();
            event.record(&mut key_fields);

            if let Some(key) = key_fields.get_key() {
                let counters_read = self.counters.read().unwrap();
                if let Some(counter) = counters_read.get(&key) {
                    counter.fetch_add(increment_amount, Ordering::Release);
                } else {
                    drop(counters_read);
                    let new_counter = AtomicUsize::new(increment_amount);
                    self.counters.write().unwrap().insert(key, new_counter);
                }
            } else {
                self.missing_keys
                    .fetch_add(increment_amount, Ordering::Release);
            }
        }
    }
}

#[derive(Default)]
pub struct KeyFields {
    lc: Option<u64>,
    ef: Option<u64>,
}

impl KeyFields {
    pub fn get_key(&self) -> Option<(u64, u64)> {
        match self {
            Self {
                lc: Some(lc),
                ef: Some(ef),
            } => Some((*lc, *ef)),
            _ => None,
        }
    }
}

impl Visit for KeyFields {
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

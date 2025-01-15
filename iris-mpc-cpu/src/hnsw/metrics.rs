use std::{
    collections::HashMap,
    fmt::Debug,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};
use tracing::{
    field::{Field, Visit}, span::Attributes, Event, Id, Subscriber
};
use tracing_subscriber::{
    layer::{Context, Layer},
    registry::LookupSpan,
};

/// Provides a basic scheme for deriving a list of control strings from the
/// `name` field of a `tracing` callsite's metadata, by splitting into a list
/// of whitespace separated substrings.  These substrings may be used to
/// determine callsite activation status in `tracing::Subscriber` or
/// `tracing_subscriber::Layer` implementations.
pub fn parse_callsite_name(name: &str) -> Vec<String> {
    name.split_whitespace().map(|s| s.to_string()).collect()
}

pub struct SingletonCounterLayer {
    pub counter: Arc<AtomicUsize>,
    target: String,
}

impl SingletonCounterLayer {
    pub fn new(target: &str) -> Self {
        Self {
            counter: Arc::new(AtomicUsize::default()),
            target: target.to_string(),
        }
    }

    pub fn get_counter(&self) -> Arc<AtomicUsize> {
        self.counter.clone()
    }

    pub fn get_target(&self) -> String {
        self.target.clone()
    }
}

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
        let counters_inner = OpCounters::default();
        Self {
            counters: Arc::new(counters_inner),
            // callsite_target: target.map(|s| s.to_string())
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
    amount: Option<i128>,
}

impl Visit for EventVisitor {
    fn record_u64(&mut self, field: &Field, value: u64) {
        match field.name() {
            "event_type" => {
                self.event = Some(value as usize);
            }
            "increment_amount" => {
                self.amount = Some(value as i128);
            }
            _ => {}
        }
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        match field.name() {
            "increment_amount" => {
                self.amount = Some(value as i128);
            }
            _ => {}
        }
    }

    fn record_debug(&mut self, _field: &Field, _value: &dyn Debug) {}
}

/// Tracing library Layer for counting detailed HNSW layer search operations
pub struct VertexOpeningsLayer {
    // Measure number of vertex openings for different lc and ef values
    pub counter_map: Arc<Mutex<HashMap<(usize, usize), usize>>>,
}

impl<S> Layer<S> for VertexOpeningsLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let span = ctx.span(id).unwrap();
        let mut visitor = LayerSearchFields::default();
        attrs.record(&mut visitor);
        span.extensions_mut().insert(visitor);
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        let mut visitor = EventVisitor::default();
        event.record(&mut visitor);

        const OPEN_NODE_EVENT: usize = Operation::OpenNode.id();
        if let Some(OPEN_NODE_EVENT) = visitor.event {
            // open node event must have parent span representing open node function
            let current_span = ctx.current_span();
            let span_id = current_span.id().unwrap();
            if let Some(LayerSearchFields {
                lc: Some(lc),
                ef: Some(ef),
            }) = ctx
                .span(span_id)
                .unwrap()
                .extensions()
                .get::<LayerSearchFields>()
            {
                let mut counter_map = self.counter_map.lock().unwrap();
                let increment_amount = visitor.amount.unwrap_or(1);
                *counter_map
                    .entry((*lc as usize, *ef as usize))
                    .or_insert(0usize) += increment_amount as usize;
            } else {
                panic!("Open node event is missing associated span fields");
            }
        }
    }
}

#[derive(Default)]
pub struct LayerSearchFields {
    lc: Option<u64>,
    ef: Option<u64>,
}

impl Visit for LayerSearchFields {
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

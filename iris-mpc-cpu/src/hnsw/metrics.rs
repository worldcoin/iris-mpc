use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tracing::{Event, Subscriber};
use tracing_subscriber::{layer::Context, Layer};

pub const LAYER_SEARCH_EVENT: u64 = 0;
pub const OPEN_NODE_EVENT: u64 = 1;
pub const EVAL_DIST_EVENT: u64 = 2;
pub const COMPARE_DIST_EVENT: u64 = 3;

const NUM_EVENT_TYPES: usize = 4;

#[derive(Default)]
pub struct EventCounter {
    pub counters: [AtomicUsize; NUM_EVENT_TYPES],
}

pub struct HnswEventCounterLayer {
    pub counters: Arc<EventCounter>,
}

impl<S: Subscriber> Layer<S> for HnswEventCounterLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = EventVisitor::default();
        event.record(&mut visitor);

        if let Some(event_type) = visitor.event {
            if let Some(counter) = self.counters.counters.get(event_type) {
                let increment_amount = visitor.amount.unwrap_or(1);
                counter.fetch_add(increment_amount, Ordering::Relaxed);
            } else {
                panic!("Invalid event type specified: {:?}", event_type);
            }
        }
    }
}

#[derive(Default)]
struct EventVisitor {
    // which event was encountered
    event: Option<usize>,

    // how much to increment the associated counter
    amount: Option<usize>,
}

impl tracing::field::Visit for EventVisitor {
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
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

    fn record_debug(&mut self, _field: &tracing::field::Field, _value: &dyn std::fmt::Debug) {}
}

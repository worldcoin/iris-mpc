use std::{
    collections::HashMap,
    fmt::Debug,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};
use tracing::{
    field::{Field, Visit},
    span::Attributes,
    Event, Id, Subscriber,
};
use tracing_subscriber::{
    layer::{Context, Layer},
    registry::LookupSpan,
};

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

impl<S> Layer<S> for HnswEventCounterLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = EventVisitor::default();
        event.record(&mut visitor);

        if let Some(event_type) = visitor.event {
            if let Some(counter) = self.counters.counters.get(event_type as usize) {
                let increment_amount = visitor.amount.unwrap_or(1);
                counter.fetch_add(increment_amount as usize, Ordering::Relaxed);
            } else {
                panic!("Invalid event type specified: {:?}", event_type);
            }
        }
    }
}

#[derive(Default)]
struct EventVisitor {
    // which event was encountered
    event: Option<u64>,

    // how much to increment the associated counter
    amount: Option<u64>,
}

impl Visit for EventVisitor {
    fn record_u64(&mut self, field: &Field, value: u64) {
        match field.name() {
            "event_type" => {
                self.event = Some(value);
            }
            "increment_amount" => {
                self.amount = Some(value);
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
    // fn register_callsite(&self, _metadata: &'static Metadata<'static>) ->
    // Interest {     Interest::sometimes()
    // }

    // fn enabled(&self, metadata: &Metadata<'_>, _ctx: Context<'_, S>) -> bool {
    //     let is_search_layer_span = metadata.is_span() && metadata.name() ==
    // "search_layer";     is_search_layer_span || metadata.is_event()
    // }

    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let span = ctx.span(id).unwrap();
        let mut visitor = LayerSearchFields::default();
        attrs.record(&mut visitor);
        span.extensions_mut().insert(visitor);
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        let mut visitor = EventVisitor::default();
        event.record(&mut visitor);

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

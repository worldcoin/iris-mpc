use std::collections::{hash_map::Entry, HashMap};

use tracing_forest::tree::{Field, Tree};

/// Tree of network events and spans similar to `Tree` from `tracing_forest`.
///
/// Each internal node of this tree contains the sum of the bytes and messages of
/// its children. These values should be written in the fields "bytes" and
/// "messages" of the corresponding event calls.
///
/// A child node is considered unique if it has the same `NodeTag` (see below).
/// The main difference with the tracing-forest `Tree` is that each internal
/// node in this tree (aka Span) contains only unique children. In particular,
/// if a child node is repeated, the `calls` field of the child node is
/// incremented.
///
/// Node children are sorted by the product of messages and calls in descending
/// order.
///
/// For example, tracing-forest returns the following tree
///
/// span1
/// ├── event1 [bytes: 10, messages: 1]
/// ├── event1 [bytes: 10, messages: 1]
/// ├── event2 [bytes: 20, messages: 2]
/// ├── span2
/// |    ├── event3 [bytes: 10, messages: 1]
/// |    └── event4 [bytes: 20, messages: 2]
/// └── span2
///      ├── event3 [bytes: 10, messages: 1]
///      └── event4 [bytes: 20, messages: 2]
///
/// Then, the corresponding `NetworkTree` will be
///
/// span1 [ 10 msgs | 100B per call, 100% msgs | 100% bytes of total ]
/// ├── span2 x 2 [ 3 msgs | 30B per call, 60% msgs | 60% bytes of total ]
/// |    ├── event4 [ 2 msgs | 20B per call, 20% msgs | 20% bytes of total ]
/// |    └── event3 [ 1 msgs | 10B per call, 10% msgs | 10% bytes of total ]
/// ├── event1 x 2 [ 1 msgs | 10B per call, 20% msgs | 20% bytes of total ]
/// └── event2 [ 2 msgs | 20B per call, 20% msgs | 20% bytes of total ]
///
/// Note that this tree omits events when formatted.
#[allow(clippy::large_enum_variant)] // See this bug https://github.com/rust-lang/rust-clippy/issues/9798
pub(crate) enum NetworkTree {
    Event(NetworkEvent),
    Span(NetworkSpan),
}

/// Leaf node of the network tree
///
/// Note that these nodes are not formatted when printing the tree.
pub(crate) struct NetworkEvent {
    pub message: Option<String>,
    pub fields: Vec<Field>,
    pub bytes: usize,    // bytes sent
    pub messages: usize, // network messages
    pub calls: usize,    // number of calls of this event in a parent span
}

/// Internal node of the network tree
pub(crate) struct NetworkSpan {
    pub name: String,
    pub fields: Vec<Field>,
    pub nodes: Vec<NetworkTree>, // unique children (span, events)
    pub bytes: usize,            // total bytes sent by all children
    pub messages: usize,         // total network messages of all children
    pub calls: usize,            // number of calls of this span in a parent span
}

impl NetworkTree {
    pub(crate) fn bytes(&self) -> usize {
        match self {
            NetworkTree::Event(event) => event.bytes,
            NetworkTree::Span(span) => span.bytes,
        }
    }

    pub(crate) fn messages(&self) -> usize {
        match self {
            NetworkTree::Event(event) => event.messages,
            NetworkTree::Span(span) => span.messages,
        }
    }

    pub(crate) fn name(&self) -> Option<String> {
        match self {
            NetworkTree::Event(event) => event.message.clone(),
            NetworkTree::Span(span) => Some(span.name.clone()),
        }
    }

    pub(crate) fn fields(&self) -> &[Field] {
        match self {
            NetworkTree::Event(event) => &event.fields,
            NetworkTree::Span(span) => &span.fields,
        }
    }

    pub(crate) fn calls(&self) -> usize {
        match self {
            NetworkTree::Event(event) => event.calls,
            NetworkTree::Span(span) => span.calls,
        }
    }

    fn increment_calls(&mut self) {
        match self {
            NetworkTree::Event(event) => event.calls += 1,
            NetworkTree::Span(span) => span.calls += 1,
        }
    }
}

/// Tag to identify unique nodes in the network tree.
/// When the tree is formatted, nodes having the same parent span and tag are
/// collapsed into one and the resulting `calls` field is set to the number of
/// these nodes. See `NetworkTree` for the context.
#[derive(Eq, PartialEq, Hash, Clone)]
struct NodeTag {
    // `message` for events, `name` for spans
    name: Option<String>,
    fields: Vec<Field>,
    bytes: usize,
    messages: usize,
    calls: usize,
}

impl NodeTag {
    fn from_tree(node: &NetworkTree) -> Self {
        let name = node.name().map(|s| s.to_owned());
        let fields = node.fields().to_vec();
        let bytes = node.bytes();
        let messages = node.messages();
        let calls = node.calls();
        NodeTag {
            name,
            fields,
            bytes,
            messages,
            calls,
        }
    }
}

/// Converts a tracing-forest `Tree` into a `NetworkTree` by propagating the
/// bytes and messages of the children to the parent and grouping unique node
/// children.
pub(crate) fn get_network_tree(tree: &Tree) -> NetworkTree {
    match tree {
        Tree::Event(event) => {
            let mut bytes = 0;
            let mut messages = 0;
            for field in event.fields().iter() {
                if field.key() == "bytes" {
                    bytes = field.value().parse().unwrap_or(0);
                }
                if field.key() == "messages" {
                    messages = field.value().parse().unwrap_or(0);
                }
            }
            NetworkTree::Event(NetworkEvent {
                message: event.message().map(|s| s.to_owned()),
                fields: event.fields().to_vec(),
                bytes,
                messages,
                calls: 1,
            })
        }
        Tree::Span(span) => {
            let mut bytes = 0;
            let mut messages = 0;
            // Unique children
            let mut nodes_map = HashMap::new();
            for node in span.nodes().iter() {
                // Trees are expected to be shallow, so this is not a performance issue
                let network_node = get_network_tree(node);
                bytes += network_node.bytes();
                messages += network_node.messages();
                // Check the current child for uniqueness and increment calls if repeated
                let node_tag = NodeTag::from_tree(&network_node);
                if let Entry::Vacant(e) = nodes_map.entry(node_tag.clone()) {
                    e.insert(network_node);
                } else {
                    let node: &mut NetworkTree = nodes_map.get_mut(&node_tag).unwrap();
                    node.increment_calls();
                }
            }
            let mut nodes: Vec<NetworkTree> = nodes_map.into_values().collect();
            // Sort nodes by the product of messages and calls in descending order
            nodes.sort_by_key(|node| std::cmp::Reverse(node.messages() * node.calls()));
            NetworkTree::Span(NetworkSpan {
                name: span.name().to_owned(),
                fields: span.fields().to_vec(),
                nodes,
                bytes,
                messages,
                calls: 1,
            })
        }
    }
}

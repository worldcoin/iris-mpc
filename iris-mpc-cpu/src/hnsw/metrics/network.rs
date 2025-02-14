use std::{
    collections::{hash_map::Entry, HashMap},
    fmt::{self, Write},
};
use tracing_forest::{
    printer::Formatter,
    tree::{Field, Tree},
};

/// Tree of network events and spans similar to `Tree` from `tracing_forest`.
///
/// Each internal node of this tree contains the sum of the bytes and rounds of
/// its children. These values should be written in the fields "bytes" and
/// "rounds" of the corresponding event calls.
///
/// A child node is considered unique if it has the same `NodeTag` (see below).
/// The main difference with the tracing-forest `Tree` is that each internal
/// node in this tree (aka Span) contains only unique children. In particular,
/// if a child node is repeated, the `calls` field of the child node is
/// incremented.
///
/// Node children are sorted by the product of rounds and calls in descending
/// order.
///
/// For example, tracing-forest returns the following tree
///
/// span1
/// ├── event1 [bytes: 10, rounds: 1]
/// ├── event1 [bytes: 10, rounds: 1]
/// ├── event2 [bytes: 20, rounds: 2]
/// ├── span2
/// |    ├── event3 [bytes: 10, rounds: 1]
/// |    └── event4 [bytes: 20, rounds: 2]
/// └── span2
///      ├── event3 [bytes: 10, rounds: 1]
///      └── event4 [bytes: 20, rounds: 2]
///
/// Then, the corresponding `NetworkTree` will be
///
/// span1 [ 10 rounds | 100B per call, 100% rounds | 100% bytes of total ]
/// ├── span2 x 2 [ 3 rounds | 30B per call, 60% rounds | 60% bytes of total ]
/// |    ├── event4 [ 2 rounds | 20B per call, 20% rounds | 20% bytes of total ]
/// |    └── event3 [ 1 rounds | 10B per call, 10% rounds | 10% bytes of total ]
/// ├── event1 x 2 [ 1 rounds | 10B per call, 20% rounds | 20% bytes of total ]
/// └── event2 [ 2 rounds | 20B per call, 20% rounds | 20% bytes of total ]
///
/// Note that this tree omits events when formatted.
#[allow(clippy::large_enum_variant)] // See this bug https://github.com/rust-lang/rust-clippy/issues/9798
enum NetworkTree {
    Event(NetworkEvent),
    Span(NetworkSpan),
}

/// Leaf node of the network tree
///
/// Note that these nodes are not formatted when printing the tree.
struct NetworkEvent {
    pub message: Option<String>,
    pub fields:  Vec<Field>,
    pub bytes:   usize, // bytes sent
    pub rounds:  usize, // communication rounds
    pub calls:   usize, // number of calls of this event in a parent span
}

/// Internal node of the network tree
struct NetworkSpan {
    pub name:   String,
    pub fields: Vec<Field>,
    pub nodes:  Vec<NetworkTree>, // unique children (span, events)
    pub bytes:  usize,            // total bytes sent by all children
    pub rounds: usize,            // total communication rounds of all children
    pub calls:  usize,            // number of calls of this span in a parent span
}

impl NetworkTree {
    fn bytes(&self) -> usize {
        match self {
            NetworkTree::Event(event) => event.bytes,
            NetworkTree::Span(span) => span.bytes,
        }
    }

    fn rounds(&self) -> usize {
        match self {
            NetworkTree::Event(event) => event.rounds,
            NetworkTree::Span(span) => span.rounds,
        }
    }

    fn name(&self) -> Option<String> {
        match self {
            NetworkTree::Event(event) => event.message.clone(),
            NetworkTree::Span(span) => Some(span.name.clone()),
        }
    }

    fn fields(&self) -> &[Field] {
        match self {
            NetworkTree::Event(event) => &event.fields,
            NetworkTree::Span(span) => &span.fields,
        }
    }

    fn calls(&self) -> usize {
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

/// Tag to identify unique nodes in the network tree
#[derive(Eq, PartialEq, Hash, Clone)]
struct NodeTag {
    name:   Option<String>, // `message` for events, `name` for spans
    fields: Vec<Field>,
    bytes:  usize,
    rounds: usize,
    calls:  usize,
}

impl NodeTag {
    fn from_tree(node: &NetworkTree) -> Self {
        let name = node.name().map(|s| s.to_owned());
        let fields = node.fields().to_vec();
        let bytes = node.bytes();
        let rounds = node.rounds();
        let calls = node.calls();
        NodeTag {
            name,
            fields,
            bytes,
            rounds,
            calls,
        }
    }
}

/// Converts a tracing-forest `Tree` into a `NetworkTree` by propagating the
/// bytes and rounds of the children to the parent and grouping unique node
/// children.
fn get_network_tree(tree: &Tree) -> NetworkTree {
    match tree {
        Tree::Event(event) => {
            let mut bytes = 0;
            let mut rounds = 0;
            for field in event.fields().iter() {
                if field.key() == "bytes" {
                    bytes = field.value().to_string().parse().unwrap();
                }
                if field.key() == "rounds" {
                    rounds = field.value().to_string().parse().unwrap();
                }
            }
            NetworkTree::Event(NetworkEvent {
                message: event.message().map(|s| s.to_owned()),
                fields: event.fields().to_vec(),
                bytes,
                rounds,
                calls: 1,
            })
        }
        Tree::Span(span) => {
            let mut bytes = 0;
            let mut rounds = 0;
            // Unique children
            let mut nodes_map = HashMap::new();
            for node in span.nodes().iter() {
                // Trees are expected to be shallow, so this is not a performance issue
                let network_node = get_network_tree(node);
                bytes += network_node.bytes();
                rounds += network_node.rounds();
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
            // Sort nodes by the product of rounds and calls in descending order
            nodes.sort_by_key(|node| -(node.rounds() as i64 * node.calls() as i64));
            NetworkTree::Span(NetworkSpan {
                name: span.name().to_owned(),
                fields: span.fields().to_vec(),
                nodes,
                bytes,
                rounds,
                calls: 1,
            })
        }
    }
}

/// Formatter for network trees similar to `PrettyPrinter` from
/// `tracing_forest`.
pub struct NetworkFormatter {}

impl Formatter for NetworkFormatter {
    type Error = fmt::Error;

    fn fmt(&self, tree: &Tree) -> Result<String, fmt::Error> {
        let network_tree = get_network_tree(tree);
        let mut tree_string = String::with_capacity(256);

        NetworkFormatter::format_tree(
            &network_tree,
            None,
            None,
            &mut IndentVec::new(),
            &mut tree_string,
        )?;

        Ok(tree_string)
    }
}

impl NetworkFormatter {
    fn format_tree(
        tree: &NetworkTree,
        bytes_root: Option<f64>,
        rounds_root: Option<f64>,
        indent: &mut IndentVec,
        writer: &mut String,
    ) -> fmt::Result {
        match tree {
            NetworkTree::Event(_) => Ok(()),
            NetworkTree::Span(span) => {
                NetworkFormatter::format_indent(indent, writer)?;
                NetworkFormatter::format_span(span, bytes_root, rounds_root, indent, writer)
            }
        }
    }

    fn format_indent(indent: &[Indent], writer: &mut String) -> fmt::Result {
        for indent in indent {
            writer.write_str(indent.repr())?;
        }
        Ok(())
    }

    fn format_span(
        span: &NetworkSpan,
        bytes_root: Option<f64>,
        rounds_root: Option<f64>,
        indent: &mut IndentVec,
        writer: &mut String,
    ) -> fmt::Result {
        let total_bytes = (span.bytes * span.calls) as f64;
        let bytes_root = bytes_root.unwrap_or(total_bytes);
        let percent_total_of_root_bytes = 100.0 * total_bytes / bytes_root;

        write!(writer, "{}", span.name)?;
        if span.calls > 1 {
            write!(writer, " x {}", span.calls)?;
        }

        let total_rounds = (span.rounds * span.calls) as f64;
        let rounds_root = rounds_root.unwrap_or(total_rounds);
        let percent_total_of_root_rounds = 100.0 * total_rounds / rounds_root;

        write!(
            writer,
            " [ {} rounds | {} per call, ",
            span.rounds,
            BytesDisplay(span.bytes as f64)
        )?;
        write!(
            writer,
            "{:.2}% rounds | {:.2}% bytes of total ] | ",
            percent_total_of_root_bytes, percent_total_of_root_rounds
        )?;

        for (n, field) in span.fields.iter().enumerate() {
            write!(
                writer,
                "{} {}: {}",
                if n == 0 { "" } else { " |" },
                field.key(),
                field.value()
            )?;
        }
        writeln!(writer)?;

        if let Some((last, remaining)) = span.nodes.split_last() {
            match indent.last_mut() {
                Some(edge @ Indent::Turn) => *edge = Indent::Null,
                Some(edge @ Indent::Fork) => *edge = Indent::Line,
                _ => {}
            }

            indent.push(Indent::Fork);

            for tree in remaining {
                if let Some(edge) = indent.last_mut() {
                    *edge = Indent::Fork;
                }
                NetworkFormatter::format_tree(
                    tree,
                    Some(bytes_root),
                    Some(rounds_root),
                    indent,
                    writer,
                )?;
            }

            if let Some(edge) = indent.last_mut() {
                *edge = Indent::Turn;
            }
            NetworkFormatter::format_tree(
                last,
                Some(bytes_root),
                Some(rounds_root),
                indent,
                writer,
            )?;

            indent.pop();
        }

        Ok(())
    }
}

enum Indent {
    Null,
    Line,
    Fork,
    Turn,
}

impl Indent {
    fn repr(&self) -> &'static str {
        match self {
            Self::Null => "   ",
            Self::Line => "│  ",
            Self::Fork => "┝━ ",
            Self::Turn => "┕━ ",
        }
    }
}

type IndentVec = Vec<Indent>;

struct BytesDisplay(f64);

impl fmt::Display for BytesDisplay {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut t = self.0;
        for unit in ["B", "kB", "MB", "GB"] {
            if t < 1000.0 && unit == "B" {
                return write!(f, "{:.0}{}", t, unit);
            }
            if t < 10.0 {
                return write!(f, "{:.2}{}", t, unit);
            } else if t < 100.0 {
                return write!(f, "{:.1}{}", t, unit);
            } else if t < 1000.0 {
                return write!(f, "{:.0}{}", t, unit);
            }
            t /= 1000.0;
        }
        write!(f, "{:.0}GB", t * 1000.0)
    }
}

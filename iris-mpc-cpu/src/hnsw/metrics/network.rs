use std::{
    collections::{hash_map::Entry, HashMap},
    fmt::{self, Write},
};
use tracing_forest::{
    printer::Formatter,
    tree::{Event, Field, Span, Tree},
};

enum NetworkTree {
    Event(Box<NetworkEvent>),
    Span(Box<NetworkSpan>),
}

struct NetworkEvent {
    pub event:  Event,
    pub bytes:  usize,
    pub rounds: usize,
    pub calls:  usize,
}

struct NetworkSpan {
    pub span:   Span,
    pub nodes:  Vec<NetworkTree>,
    pub bytes:  usize,
    pub rounds: usize,
    pub calls:  usize,
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

    fn name(&self) -> Option<&str> {
        match self {
            NetworkTree::Event(event) => event.event.message(),
            NetworkTree::Span(span) => Some(span.span.name()),
        }
    }

    fn fields(&self) -> &[Field] {
        match self {
            NetworkTree::Event(event) => event.event.fields(),
            NetworkTree::Span(span) => span.span.fields(),
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

#[derive(Eq, PartialEq, Hash, Clone)]
struct NodeTag {
    name:   Option<String>,
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
            NetworkTree::Event(Box::new(NetworkEvent {
                event: event.clone(),
                bytes,
                rounds,
                calls: 1,
            }))
        }
        Tree::Span(span) => {
            let mut bytes = 0;
            let mut rounds = 0;
            let mut nodes_map = HashMap::new();
            for node in span.nodes().iter() {
                let network_node = get_network_tree(node);
                bytes += network_node.bytes();
                rounds += network_node.rounds();
                let node_tag = NodeTag::from_tree(&network_node);
                if let Entry::Vacant(e) = nodes_map.entry(node_tag.clone()) {
                    e.insert(network_node);
                } else {
                    let node: &mut NetworkTree = nodes_map.get_mut(&node_tag).unwrap();
                    node.increment_calls();
                }
            }
            let nodes = nodes_map.into_values().collect();
            NetworkTree::Span(Box::new(NetworkSpan {
                span: span.clone(),
                nodes,
                bytes,
                rounds,
                calls: 1,
            }))
        }
    }
}

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
        // let total_duration = span.span.total_duration().as_nanos() as f64;
        // let inner_duration = span.span.inner_duration().as_nanos() as f64;
        // let root_duration = duration_root.unwrap_or(total_duration);
        // let percent_total_of_root_duration = 100.0 * total_duration / root_duration;
        //
        // write!(
        // writer,
        // "{} [ {} | ",
        // span.span.name(),
        // DurationDisplay(total_duration)
        // )?;
        //
        // if inner_duration > 0.0 {
        // let base_duration = span.span.base_duration().as_nanos() as f64;
        // let percent_base_of_root_duration = 100.0 * base_duration / root_duration;
        // write!(writer, "{:.2}% / ", percent_base_of_root_duration)?;
        // }
        //
        // write!(writer, "{:.2}% ]", percent_total_of_root_duration)?;

        let total_bytes = (span.bytes * span.calls) as f64;
        let bytes_root = bytes_root.unwrap_or(total_bytes);
        let percent_total_of_root_bytes = 100.0 * total_bytes / bytes_root;

        write!(writer, "{}", span.span.name())?;
        if span.calls > 1 {
            write!(writer, " x {}", span.calls)?;
        }
        write!(writer, " [ {} | ", BytesDisplay(total_bytes))?;
        write!(writer, "{:.2}% ]", percent_total_of_root_bytes)?;

        let total_rounds = (span.rounds * span.calls) as f64;
        let rounds_root = rounds_root.unwrap_or(total_rounds);
        let percent_total_of_root_rounds = 100.0 * total_rounds / rounds_root;

        write!(writer, "[ {} rounds | ", total_rounds)?;
        write!(writer, "{:.2}% ]", percent_total_of_root_rounds)?;

        for (n, field) in span.span.fields().iter().enumerate() {
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

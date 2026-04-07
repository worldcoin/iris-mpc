use std::{
    collections::{hash_map::Entry, HashMap},
    fmt::{self, Write},
    sync::{Arc, Mutex},
};
use tracing_forest::{
    printer::Formatter,
    tree::{Field, Tree},
};

/// Sort order for network statistics tables.
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum, serde::Deserialize)]
pub enum SortBy {
    /// Sort tables by total bytes (default)
    #[default]
    Bytes,
    /// Sort tables by number of messages
    Messages,
}

impl SortBy {
    pub fn key(self, node: &StatsTreeNode) -> u64 {
        match self {
            SortBy::Bytes => node.total_bytes(),
            SortBy::Messages => node.total_messages(),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            SortBy::Bytes => "total bytes",
            SortBy::Messages => "total messages",
        }
    }
}

/// Flat per-function row: totals across all call sites in the tree.
pub struct FlatFunctionStats {
    pub total_bytes: u64,
    pub total_messages: u64,
}

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
enum NetworkTree {
    Event(NetworkEvent),
    Span(NetworkSpan),
}

/// Leaf node of the network tree
///
/// Note that these nodes are not formatted when printing the tree.
struct NetworkEvent {
    pub message: Option<String>,
    pub fields: Vec<Field>,
    pub bytes: usize,    // bytes sent
    pub messages: usize, // network messages
    pub calls: usize,    // number of calls of this event in a parent span
}

/// Internal node of the network tree
struct NetworkSpan {
    pub name: String,
    pub fields: Vec<Field>,
    pub nodes: Vec<NetworkTree>, // unique children (span, events)
    pub bytes: usize,            // total bytes sent by all children
    pub messages: usize,         // total network messages of all children
    pub calls: usize,            // number of calls of this span in a parent span
}

impl NetworkTree {
    fn bytes(&self) -> usize {
        match self {
            NetworkTree::Event(event) => event.bytes,
            NetworkTree::Span(span) => span.bytes,
        }
    }

    fn messages(&self) -> usize {
        match self {
            NetworkTree::Event(event) => event.messages,
            NetworkTree::Span(span) => span.messages,
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
fn get_network_tree(tree: &Tree) -> NetworkTree {
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

/// A node in the accumulated stats tree.  Each node tracks direct
/// bytes/messages (from tracing events attributed to this span) and has
/// children keyed by span name.  This preserves the full call hierarchy,
/// avoiding the ambiguity that arose from the old `(function, parent)` flat
/// key approach.
#[derive(Debug, Clone, Default)]
pub struct StatsTreeNode {
    pub name: String,
    pub direct_bytes: u64,
    pub direct_messages: u64,
    pub children: Vec<StatsTreeNode>,
}

/// Walks a raw tracing-forest `Tree` and merges its stats into a tree of
/// `StatsTreeNode`s rooted in `roots`.
fn accumulate_tree_into(tree: &Tree, roots: &mut HashMap<String, StatsTreeNode>) {
    match tree {
        Tree::Event(_) => {} // top-level events outside any span are ignored
        Tree::Span(span) => {
            let name = span.name().to_string();
            let root = roots.entry(name.clone()).or_insert_with(|| StatsTreeNode {
                name,
                ..Default::default()
            });
            for child in span.nodes().iter() {
                accumulate_child(child, root);
            }
        }
    }
}

impl StatsTreeNode {
    /// Find or create a child node by name.
    /// Search is done via linear scan since we expect a small number of children per node.
    fn child_mut(&mut self, name: &str) -> &mut Self {
        let pos = self.children.iter().position(|c| c.name == name);
        match pos {
            Some(i) => &mut self.children[i],
            None => {
                self.children.push(StatsTreeNode {
                    name: name.to_string(),
                    ..Default::default()
                });
                self.children.last_mut().unwrap()
            }
        }
    }

    /// Recursively sort children by total_bytes descending.
    fn sort_recursive(&mut self) {
        for child in &mut self.children {
            child.sort_recursive();
        }
        self.children
            .sort_by_key(|c| std::cmp::Reverse(c.total_bytes()));
    }

    /// Total bytes (direct + all descendants).
    pub fn total_bytes(&self) -> u64 {
        self.direct_bytes + self.children.iter().map(|c| c.total_bytes()).sum::<u64>()
    }

    /// Total messages (direct + all descendants).
    pub fn total_messages(&self) -> u64 {
        self.direct_messages
            + self
                .children
                .iter()
                .map(|c| c.total_messages())
                .sum::<u64>()
    }

    /// Recursively sort tree nodes by the chosen key (descending).
    pub fn sort_roots(roots: &mut [Self], sort_by: SortBy) {
        for node in roots.iter_mut() {
            Self::sort_roots(&mut node.children, sort_by);
        }
        roots.sort_by_key(|b| std::cmp::Reverse(sort_by.key(b)));
    }

    /// Flatten the tree into a map keyed by function name, summing
    /// `total_bytes` / `total_messages` across all call sites.
    /// Entries with zero bytes are removed.
    pub fn flatten_all(roots: &[Self]) -> HashMap<String, FlatFunctionStats> {
        let mut map = HashMap::new();
        for root in roots {
            flatten_stats(root, &mut map);
        }
        map.retain(|_, v| v.total_bytes > 0);
        map
    }
}

fn flatten_stats(node: &StatsTreeNode, map: &mut HashMap<String, FlatFunctionStats>) {
    let entry = map.entry(node.name.clone()).or_insert(FlatFunctionStats {
        total_bytes: 0,
        total_messages: 0,
    });
    entry.total_bytes += node.total_bytes();
    entry.total_messages += node.total_messages();
    for child in &node.children {
        flatten_stats(child, map);
    }
}

fn accumulate_child(tree: &Tree, parent: &mut StatsTreeNode) {
    match tree {
        Tree::Event(event) => {
            for field in event.fields().iter() {
                if field.key() == "bytes" {
                    parent.direct_bytes += field.value().parse().unwrap_or(0);
                }
                if field.key() == "messages" {
                    parent.direct_messages += field.value().parse().unwrap_or(0);
                }
            }
        }
        Tree::Span(span) => {
            let child = parent.child_mut(span.name());
            for node in span.nodes().iter() {
                accumulate_child(node, child);
            }
        }
    }
}

/// Formatter for network trees similar to `PrettyPrinter` from
/// `tracing_forest`.
///
/// Implements the `tracing_forest::Formatter` trait: each time a tracing span
/// tree is completed, `fmt()` converts it into a [`NetworkTree`] (deduplicating
/// repeated children) and accumulates per-function byte/message counters into an
/// internal [`StatsTreeNode`] forest.
///
/// The accumulated stats can be rendered in several formats via the public
/// helper methods:
///
/// ## Hierarchical tree table (`format_tree_table`)
///
/// Preserves the call-tree structure with box-drawing indentation. Each row
/// shows direct and total (direct + descendants) bytes/messages, plus the
/// percentage of the global total:
///
/// ```text
///   Per-function breakdown (by total bytes):
///     function             direct bytes   direct msgs   total bytes  total messages  % bytes  % msgs
///     ──────────────────────────────────────────────────────────────────────────────────────────────────
///     search_to_insert         12.3 kB           420      18.5 kB            630     62.1%    58.3%
///     ├─ compare_batch         6.20 kB           210      6.20 kB            210     20.8%    19.4%
///     └─ greedy_search             0 B             0      6.20 kB            210     20.8%    19.4%
///     insert                       0 B             0      11.3 kB            450     37.9%    41.7%
///     └─ connect_peers         11.3 kB           450      11.3 kB            450     37.9%    41.7%
///     ──────────────────────────────────────────────────────────────────────────────────────────────────
///     TOTAL                                               29.8 kB           1080
/// ```
///
/// ## Flat per-function table (`format_flat_table`)
///
/// Collapses the tree: each function name appears once with its `total_bytes`
/// and `total_messages` summed across every call site in the tree:
///
/// ```text
///   Flat per-function totals (by total bytes, summed across all call sites):
///     function             bytes     messages  % bytes  % msgs
///     ─────────────────────────────────────────────────────────
///     search_to_insert     18.5 kB        630    62.1%   58.3%
///     insert               11.3 kB        450    37.9%   41.7%
///     connect_peers        11.3 kB        450    37.9%   41.7%
///     compare_batch        6.20 kB        210    20.8%   19.4%
///     ─────────────────────────────────────────────────────────
///     TOTAL                29.8 kB       1080
/// ```
///
/// ## JSON (`to_json`)
///
/// Returns a `serde_json::Value` with the full tree structure:
///
/// ```json
/// {
///   "total_bytes": 29800,
///   "total_messages": 1080,
///   "functions": [
///     { "name": "search_to_insert", "direct_bytes": 12300, "direct_messages": 420,
///       "total_bytes": 18500, "total_messages": 630,
///       "children": [ ... ] },
///     ...
///   ]
/// }
/// ```
///
/// ## CSV (`to_flat_csv`)
///
/// Flat CSV with columns `function,bytes,messages,pct_bytes,pct_messages`,
/// followed by a `TOTAL` row.
///
/// Both tables can be sorted by total bytes (default) or total messages via
/// [`SortBy`].  When `tracing_output` is enabled, `fmt()` additionally returns
/// a per-call tree string (the [`NetworkTree`] rendering with `x N` call
/// counts and per-call byte budgets); otherwise it returns an empty string for
/// efficiency.
pub struct NetworkFormatter {
    accumulator: Arc<Mutex<HashMap<String, StatsTreeNode>>>,
    tracing_output: bool,
}

impl Default for NetworkFormatter {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkFormatter {
    pub fn new() -> Self {
        Self {
            accumulator: Arc::new(Mutex::new(HashMap::new())),
            tracing_output: false,
        }
    }

    /// Enable or disable per-call tracing tree output from `fmt()`.
    /// When disabled (default), `fmt()` still accumulates stats but returns
    /// an empty string — more efficient than piping output to a null writer.
    pub fn with_tracing_output(mut self, enabled: bool) -> Self {
        self.tracing_output = enabled;
        self
    }

    /// Returns accumulated stats as a forest of `StatsTreeNode` roots.
    pub fn snapshot(&self) -> Vec<StatsTreeNode> {
        let map = self.accumulator.lock().unwrap_or_else(|e| e.into_inner());
        let mut roots: Vec<StatsTreeNode> = map.values().cloned().collect();
        for root in &mut roots {
            root.sort_recursive();
        }
        roots.sort_by_key(|b| std::cmp::Reverse(b.total_bytes()));
        roots
    }

    /// Resets accumulated stats.
    pub fn reset(&self) {
        self.accumulator
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }
}

impl Formatter for NetworkFormatter {
    type Error = fmt::Error;

    fn fmt(&self, tree: &Tree) -> Result<String, fmt::Error> {
        // Accumulate stats from the raw tree (cheap — no dedup/sort/format)
        accumulate_tree_into(
            tree,
            &mut self.accumulator.lock().unwrap_or_else(|e| e.into_inner()),
        );

        if !self.tracing_output {
            return Ok(String::new());
        }

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
        messages_root: Option<f64>,
        indent: &mut IndentVec,
        writer: &mut String,
    ) -> fmt::Result {
        match tree {
            NetworkTree::Event(_) => Ok(()),
            NetworkTree::Span(span) => {
                NetworkFormatter::format_indent(indent, writer)?;
                NetworkFormatter::format_span(span, bytes_root, messages_root, indent, writer)
            }
        }
    }

    fn format_indent(indent: &[Indent], writer: &mut String) -> fmt::Result {
        writer.write_str(&indent_string(indent))?;
        Ok(())
    }

    fn format_span(
        span: &NetworkSpan,
        bytes_root: Option<f64>,
        messages_root: Option<f64>,
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

        let total_messages = (span.messages * span.calls) as f64;
        let messages_root = messages_root.unwrap_or(total_messages);
        let percent_total_of_root_messages = 100.0 * total_messages / messages_root;

        let msgs = if span.messages == 1 {
            format!("{} msg", span.messages)
        } else {
            format!("{} msgs", span.messages)
        };
        let bytes = BytesDisplay(span.bytes as f64);
        write!(writer, " [ {msgs} | {bytes} per call, ",)?;
        write!(
            writer,
            "{:.2}% msgs | {:.2}% bytes of total ] | ",
            percent_total_of_root_messages, percent_total_of_root_bytes
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
                    Some(messages_root),
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
                Some(messages_root),
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
    /// Box-drawing character for horizontal rules / table separators.
    const RULE: &str = "─";

    fn repr(&self) -> &'static str {
        match self {
            Self::Null => "   ",
            Self::Line => "│  ",
            Self::Fork => "├─ ",
            Self::Turn => "└─ ",
        }
    }
}

type IndentVec = Vec<Indent>;

fn indent_string(indent: &[Indent]) -> String {
    indent.iter().map(|i| i.repr()).collect()
}

struct BytesDisplay(f64);

impl fmt::Display for BytesDisplay {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut t = self.0;
        for unit in ["B", "kB", "MB", "GB"] {
            if t < 1000.0 && unit == "B" {
                return write!(f, "{:.0} {}", t, unit);
            }
            if t < 10.0 {
                return write!(f, "{:.2} {}", t, unit);
            } else if t < 100.0 {
                return write!(f, "{:.1} {}", t, unit);
            } else if t < 1000.0 {
                return write!(f, "{:.0} {}", t, unit);
            }
            t /= 1000.0;
        }
        write!(f, "{:.0} GB", t * 1000.0)
    }
}

// ---------------------------------------------------------------------------
// Formatting helpers (private)
// ---------------------------------------------------------------------------

fn format_bytes(b: u64) -> String {
    format!("{}", BytesDisplay(b as f64))
}

fn display_width(s: &str) -> usize {
    s.chars().count()
}

fn pad_right(s: &str, width: usize) -> String {
    let dw = display_width(s);
    if dw >= width {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(width - dw))
    }
}

fn to_percentage(part: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        part as f64 * 100.0 / total as f64
    }
}

struct StatsNodeDisplay {
    formatted_name: String,
    direct_bytes: u64,
    direct_messages: u64,
    total_bytes: u64,
    total_messages: u64,
}

fn collect_display_names(
    node: &StatsTreeNode,
    indent: &mut IndentVec,
    names: &mut Vec<StatsNodeDisplay>,
) {
    let name = if indent.is_empty() {
        node.name.clone()
    } else {
        format!("{}{}", indent_string(indent), node.name)
    };
    names.push(StatsNodeDisplay {
        formatted_name: name,
        direct_bytes: node.direct_bytes,
        direct_messages: node.direct_messages,
        total_bytes: node.total_bytes(),
        total_messages: node.total_messages(),
    });

    match indent.last_mut() {
        Some(edge @ &mut Indent::Turn) => *edge = Indent::Null,
        Some(edge @ &mut Indent::Fork) => *edge = Indent::Line,
        _ => {}
    }

    for (i, child) in node.children.iter().enumerate() {
        if i == node.children.len() - 1 {
            indent.push(Indent::Turn);
        } else {
            indent.push(Indent::Fork);
        }
        collect_display_names(child, indent, names);
        indent.pop();
    }
}

fn tree_node_to_json(node: &StatsTreeNode) -> serde_json::Value {
    let children: Vec<serde_json::Value> = node.children.iter().map(tree_node_to_json).collect();
    let mut obj = serde_json::json!({
        "name": node.name,
        "direct_bytes": node.direct_bytes,
        "direct_messages": node.direct_messages,
        "total_bytes": node.total_bytes(),
        "total_messages": node.total_messages(),
    });
    if !children.is_empty() {
        obj["children"] = serde_json::Value::Array(children);
    }
    obj
}

// ---------------------------------------------------------------------------
// High-level formatting methods on NetworkFormatter
// ---------------------------------------------------------------------------

/// Minimum width of the function name column in both tables.
const MIN_NAME_WIDTH: usize = 20;

impl NetworkFormatter {
    /// Prepare a pruned and sorted snapshot with precomputed totals.
    fn prepared_snapshot(&self, sort_by: SortBy) -> (Vec<StatsTreeNode>, u64, u64) {
        let mut roots = self.snapshot();
        StatsTreeNode::sort_roots(&mut roots, sort_by);
        let total_bytes: u64 = roots.iter().map(|r| r.total_bytes()).sum();
        let total_messages: u64 = roots.iter().map(|r| r.total_messages()).sum();
        (roots, total_bytes, total_messages)
    }

    /// Format a hierarchical per-function table as a string.
    pub fn format_tree_table(&self, sort_by: SortBy) -> String {
        // column widths
        // direct bytes
        let w_db = 12;
        // direct messages
        let w_dm = 14;
        // total bytes
        let w_tb = 12;
        // total messages
        let w_tm = 14;
        // % bytes
        let w_pb = 8;
        // % messages
        let w_pm = 8;
        // column gaps (single spaces between columns)
        let cg = 6;

        let (roots, total_bytes, total_messages) = self.prepared_snapshot(sort_by);

        let mut all_names = vec![];
        for root in &roots {
            collect_display_names(root, &mut IndentVec::new(), &mut all_names);
        }
        let name_width = all_names
            .iter()
            .map(|n| display_width(&n.formatted_name))
            .max()
            .unwrap_or(MIN_NAME_WIDTH)
            .max(MIN_NAME_WIDTH);

        let mut out = String::new();

        writeln!(out, "  Per-function breakdown (by {}):", sort_by.label()).unwrap();
        writeln!(
            out,
            "    {} {:>w_db$} {:>w_dm$} {:>w_tb$} {:>w_tm$} {:>w_pb$} {:>w_pm$}",
            pad_right("function", name_width),
            "direct bytes",
            "direct msgs",
            "total bytes",
            "total messages",
            "% bytes",
            "% msgs"
        )
        .unwrap();
        let rule_len = name_width + w_db + w_dm + w_tb + w_tm + w_pb + w_pm + cg;
        writeln!(out, "    {}", Indent::RULE.repeat(rule_len)).unwrap();

        for n in all_names {
            writeln!(
                out,
                "    {} {:>w_db$} {:>w_dm$} {:>w_tb$} {:>w_tm$} {:>w_pb$.1}% {:>w_pm$.1}%",
                pad_right(&n.formatted_name, name_width),
                format_bytes(n.direct_bytes),
                n.direct_messages,
                format_bytes(n.total_bytes),
                n.total_messages,
                to_percentage(n.total_bytes, total_bytes),
                to_percentage(n.total_messages, total_messages),
                w_pb = w_pb - 1,
                w_pm = w_pm - 1,
            )
            .unwrap();
        }

        writeln!(out, "    {}", Indent::RULE.repeat(rule_len)).unwrap();
        writeln!(
            out,
            "    {} {:>w_db$} {:>w_dm$} {:>w_tb$} {:>w_tm$}",
            pad_right("TOTAL", name_width),
            "",
            "",
            format_bytes(total_bytes),
            total_messages
        )
        .unwrap();

        out
    }

    /// Format a flat per-function summary table as a string.
    pub fn format_flat_table(&self, sort_by: SortBy) -> String {
        // column widths
        // bytes
        let w_b = 12;
        // messages
        let w_m = 8;
        // % bytes
        let w_pb = 8;
        // % messages
        let w_pm = 8;
        // column gaps (single spaces between columns)
        let cg = 4;

        let (roots, total_bytes, total_messages) = self.prepared_snapshot(sort_by);

        let map = StatsTreeNode::flatten_all(&roots);

        let mut rows: Vec<_> = map.into_iter().collect();
        match sort_by {
            SortBy::Bytes => rows.sort_by_key(|r| std::cmp::Reverse(r.1.total_bytes)),
            SortBy::Messages => rows.sort_by_key(|r| std::cmp::Reverse(r.1.total_messages)),
        }

        let name_width = rows
            .iter()
            .map(|(n, _)| display_width(n))
            .chain(std::iter::once(8)) // width of "function" header
            .chain(std::iter::once(5)) // width of "TOTAL" row
            .max()
            .unwrap_or(MIN_NAME_WIDTH)
            .max(MIN_NAME_WIDTH);

        let mut out = String::new();

        writeln!(
            out,
            "  Flat per-function totals (by {}, summed across all call sites):",
            sort_by.label()
        )
        .unwrap();
        writeln!(
            out,
            "    {} {:>w_b$} {:>w_m$} {:>w_pb$} {:>w_pm$}",
            pad_right("function", name_width),
            "bytes",
            "messages",
            "% bytes",
            "% msgs"
        )
        .unwrap();
        let rule_len = name_width + w_b + w_m + w_pb + w_pm + cg;
        writeln!(out, "    {}", Indent::RULE.repeat(rule_len)).unwrap();

        for (name, stats) in &rows {
            let pct_bytes = to_percentage(stats.total_bytes, total_bytes);
            let pct_messages = to_percentage(stats.total_messages, total_messages);
            writeln!(
                out,
                "    {} {:>w_b$} {:>w_m$} {:>w_pb$.1}% {:>w_pm$.1}%",
                pad_right(name, name_width),
                format_bytes(stats.total_bytes),
                stats.total_messages,
                pct_bytes,
                pct_messages,
                w_pb = w_pb - 1,
                w_pm = w_pm - 1,
            )
            .unwrap();
        }

        writeln!(out, "    {}", Indent::RULE.repeat(rule_len)).unwrap();
        writeln!(
            out,
            "    {} {:>w_b$} {:>w_m$}",
            pad_right("TOTAL", name_width),
            format_bytes(total_bytes),
            total_messages
        )
        .unwrap();

        out
    }

    /// Return accumulated stats as a JSON value.
    pub fn to_json(&self, sort_by: SortBy) -> serde_json::Value {
        let (roots, total_bytes, total_messages) = self.prepared_snapshot(sort_by);

        let tree_json: Vec<serde_json::Value> = roots.iter().map(tree_node_to_json).collect();
        serde_json::json!({
            "total_bytes": total_bytes,
            "total_messages": total_messages,
            "functions": tree_json,
        })
    }

    /// Return accumulated stats as a CSV string.
    pub fn to_flat_csv(&self, sort_by: SortBy) -> String {
        let (roots, total_bytes, total_messages) = self.prepared_snapshot(sort_by);

        let map = StatsTreeNode::flatten_all(&roots);

        let mut rows: Vec<_> = map.into_iter().collect();
        match sort_by {
            SortBy::Bytes => rows.sort_by_key(|r| std::cmp::Reverse(r.1.total_bytes)),
            SortBy::Messages => rows.sort_by_key(|r| std::cmp::Reverse(r.1.total_messages)),
        }

        let mut out = String::from("function,bytes,messages,pct_bytes,pct_messages\n");
        for (name, stats) in &rows {
            let pct_bytes = to_percentage(stats.total_bytes, total_bytes);
            let pct_messages = to_percentage(stats.total_messages, total_messages);
            writeln!(
                out,
                "{},{},{},{:.1},{:.1}",
                name, stats.total_bytes, stats.total_messages, pct_bytes, pct_messages
            )
            .unwrap();
        }
        writeln!(out, "TOTAL,{},{},100.0,100.0", total_bytes, total_messages).unwrap();
        out
    }
}

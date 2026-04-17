use std::{
    collections::HashMap,
    fmt::{self, Write},
    sync::{Arc, Mutex},
};

use tracing_forest::{printer::Formatter, tree::Tree};

use super::{
    display::{
        collect_display_names, display_width, format_bytes, indent_string, pad_right,
        to_percentage, tree_node_to_json, BytesDisplay, Indent, IndentVec,
    },
    stats::{accumulate_tree_into, SortBy, StatsTreeNode},
    tree::{get_network_tree, NetworkSpan, NetworkTree},
};

/// Minimum width of the function name column in both tables.
const MIN_NAME_WIDTH: usize = 20;

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
        map.values().cloned().collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_forest::printer::Formatter;

    /// Capture tracing trees from a closure that emits spans/events.
    async fn capture_trees<F: std::future::Future<Output = ()>>(
        f: F,
    ) -> Vec<tracing_forest::tree::Tree> {
        tracing_forest::capture().build().on(f).await
    }

    /// Feed a known tree into the formatter for output format tests.
    async fn formatter_with_data() -> NetworkFormatter {
        let formatter = NetworkFormatter::new();

        let trees = capture_trees(async {
            let _span = tracing::info_span!("search").entered();
            tracing::info!(bytes = 100, messages = 10, "send");
            {
                let _inner = tracing::info_span!("compare").entered();
                tracing::info!(bytes = 50, messages = 5, "send");
            }
        })
        .await;
        for tree in &trees {
            formatter.fmt(tree).unwrap();
        }

        let trees = capture_trees(async {
            let _span = tracing::info_span!("insert").entered();
            tracing::info!(bytes = 200, messages = 20, "send");
        })
        .await;
        for tree in &trees {
            formatter.fmt(tree).unwrap();
        }

        formatter
    }

    // -----------------------------------------------------------------------
    // NetworkFormatter integration tests (via tracing_forest::capture)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn fmt_accumulates_single_span() {
        let trees = capture_trees(async {
            let _span = tracing::info_span!("my_func").entered();
            tracing::info!(bytes = 100, messages = 2, "send");
        })
        .await;

        let formatter = NetworkFormatter::new();
        for tree in &trees {
            formatter.fmt(tree).unwrap();
        }

        let snapshot = formatter.snapshot();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].name, "my_func");
        assert_eq!(snapshot[0].direct_bytes, 100);
        assert_eq!(snapshot[0].direct_messages, 2);
    }

    #[tokio::test]
    async fn fmt_accumulates_across_multiple_calls() {
        let formatter = NetworkFormatter::new();

        // First call
        let trees = capture_trees(async {
            let _span = tracing::info_span!("func_a").entered();
            tracing::info!(bytes = 50, messages = 1, "send");
        })
        .await;
        for tree in &trees {
            formatter.fmt(tree).unwrap();
        }

        // Second call — same function name, stats should sum
        let trees = capture_trees(async {
            let _span = tracing::info_span!("func_a").entered();
            tracing::info!(bytes = 30, messages = 2, "send");
        })
        .await;
        for tree in &trees {
            formatter.fmt(tree).unwrap();
        }

        let snapshot = formatter.snapshot();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].direct_bytes, 80);
        assert_eq!(snapshot[0].direct_messages, 3);
    }

    #[tokio::test]
    async fn fmt_nested_spans_produce_children() {
        let trees = capture_trees(async {
            let _outer = tracing::info_span!("outer").entered();
            {
                let _inner = tracing::info_span!("inner").entered();
                tracing::info!(bytes = 200, messages = 5, "send");
            }
        })
        .await;

        let formatter = NetworkFormatter::new();
        for tree in &trees {
            formatter.fmt(tree).unwrap();
        }

        let snapshot = formatter.snapshot();
        assert_eq!(snapshot.len(), 1);
        let outer = &snapshot[0];
        assert_eq!(outer.name, "outer");
        assert_eq!(outer.direct_bytes, 0);
        assert_eq!(outer.children.len(), 1);

        let inner = &outer.children[0];
        assert_eq!(inner.name, "inner");
        assert_eq!(inner.direct_bytes, 200);
        assert_eq!(inner.direct_messages, 5);
    }

    #[tokio::test]
    async fn fmt_tracing_output_disabled_returns_empty() {
        let trees = capture_trees(async {
            let _span = tracing::info_span!("f").entered();
            tracing::info!(bytes = 10, messages = 1, "send");
        })
        .await;

        let formatter = NetworkFormatter::new(); // tracing_output defaults false
        let result = formatter.fmt(&trees[0]).unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn fmt_tracing_output_enabled_returns_nonempty() {
        let trees = capture_trees(async {
            let _span = tracing::info_span!("my_span").entered();
            tracing::info!(bytes = 10, messages = 1, "send");
        })
        .await;

        let formatter = NetworkFormatter::new().with_tracing_output(true);
        let result = formatter.fmt(&trees[0]).unwrap();
        assert!(!result.is_empty());
        assert!(result.contains("my_span"));
    }

    #[tokio::test]
    async fn reset_clears_accumulator() {
        let trees = capture_trees(async {
            let _span = tracing::info_span!("f").entered();
            tracing::info!(bytes = 100, messages = 1, "send");
        })
        .await;

        let formatter = NetworkFormatter::new();
        formatter.fmt(&trees[0]).unwrap();
        assert!(!formatter.snapshot().is_empty());

        formatter.reset();
        assert!(formatter.snapshot().is_empty());
    }

    // -----------------------------------------------------------------------
    // Output format tests (tree table, flat table, JSON, CSV)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn format_tree_table_contains_expected_sections() {
        let formatter = formatter_with_data().await;
        let table = formatter.format_tree_table(SortBy::Bytes);

        assert!(table.contains("Per-function breakdown"));
        assert!(table.contains("function"));
        assert!(table.contains("direct bytes"));
        assert!(table.contains("total bytes"));
        assert!(table.contains("TOTAL"));
        assert!(table.contains("search"));
        assert!(table.contains("compare"));
        assert!(table.contains("insert"));
    }

    #[tokio::test]
    async fn format_flat_table_contains_expected_sections() {
        let formatter = formatter_with_data().await;
        let table = formatter.format_flat_table(SortBy::Bytes);

        assert!(table.contains("Flat per-function totals"));
        assert!(table.contains("function"));
        assert!(table.contains("bytes"));
        assert!(table.contains("TOTAL"));
        assert!(table.contains("search"));
        assert!(table.contains("insert"));
    }

    #[tokio::test]
    async fn to_json_has_correct_structure() {
        let formatter = formatter_with_data().await;
        let json = formatter.to_json(SortBy::Bytes);

        assert!(json["total_bytes"].as_u64().unwrap() > 0);
        assert!(json["total_messages"].as_u64().unwrap() > 0);
        assert!(json["functions"].is_array());

        let functions = json["functions"].as_array().unwrap();
        assert!(!functions.is_empty());

        // Each function node has expected keys
        let first = &functions[0];
        assert!(first["name"].is_string());
        assert!(first["direct_bytes"].is_u64());
        assert!(first["total_bytes"].is_u64());
        assert!(first["total_messages"].is_u64());
    }

    #[tokio::test]
    async fn to_json_totals_match_snapshot() {
        let formatter = formatter_with_data().await;
        let json = formatter.to_json(SortBy::Bytes);

        let snapshot = formatter.snapshot();
        let expected_bytes: u64 = snapshot.iter().map(|r| r.total_bytes()).sum();
        let expected_msgs: u64 = snapshot.iter().map(|r| r.total_messages()).sum();

        assert_eq!(json["total_bytes"].as_u64().unwrap(), expected_bytes);
        assert_eq!(json["total_messages"].as_u64().unwrap(), expected_msgs);
    }

    #[tokio::test]
    async fn to_flat_csv_is_parseable() {
        let formatter = formatter_with_data().await;
        let csv = formatter.to_flat_csv(SortBy::Bytes);

        let lines: Vec<&str> = csv.lines().collect();
        // header + at least one data row + TOTAL row
        assert!(lines.len() >= 3);
        assert_eq!(lines[0], "function,bytes,messages,pct_bytes,pct_messages");
        assert!(lines.last().unwrap().starts_with("TOTAL,"));
    }

    #[tokio::test]
    async fn to_flat_csv_total_row_sums_correctly() {
        let formatter = formatter_with_data().await;
        let csv = formatter.to_flat_csv(SortBy::Bytes);

        let total_line = csv.lines().last().unwrap();
        let parts: Vec<&str> = total_line.split(',').collect();
        assert_eq!(parts[0], "TOTAL");

        let total_bytes: u64 = parts[1].parse().unwrap();
        let total_messages: u64 = parts[2].parse().unwrap();

        let snapshot = formatter.snapshot();
        let expected_bytes: u64 = snapshot.iter().map(|r| r.total_bytes()).sum();
        let expected_msgs: u64 = snapshot.iter().map(|r| r.total_messages()).sum();

        assert_eq!(total_bytes, expected_bytes);
        assert_eq!(total_messages, expected_msgs);
    }
}

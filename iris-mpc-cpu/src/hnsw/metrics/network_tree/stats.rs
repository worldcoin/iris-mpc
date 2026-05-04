use std::collections::HashMap;

use tracing_forest::tree::Tree;

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
pub(crate) fn accumulate_tree_into(tree: &Tree, roots: &mut HashMap<String, StatsTreeNode>) {
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
    pub(crate) fn child_mut(&mut self, name: &str) -> &mut Self {
        let pos = self.children.iter().position(|c| c.name == name);
        match pos {
            Some(i) => &mut self.children[i],
            None => {
                let last_index = self.children.len();
                self.children.push(StatsTreeNode {
                    name: name.to_string(),
                    ..Default::default()
                });
                &mut self.children[last_index]
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a leaf StatsTreeNode (no children).
    fn leaf(name: &str, bytes: u64, messages: u64) -> StatsTreeNode {
        StatsTreeNode {
            name: name.to_string(),
            direct_bytes: bytes,
            direct_messages: messages,
            children: vec![],
        }
    }

    /// Build a branch StatsTreeNode with given children and zero direct stats.
    fn branch(name: &str, children: Vec<StatsTreeNode>) -> StatsTreeNode {
        StatsTreeNode {
            name: name.to_string(),
            direct_bytes: 0,
            direct_messages: 0,
            children,
        }
    }

    #[test]
    fn stats_tree_node_leaf_totals() {
        let node = leaf("f", 100, 5);
        assert_eq!(node.total_bytes(), 100);
        assert_eq!(node.total_messages(), 5);
    }

    #[test]
    fn stats_tree_node_totals_include_children() {
        let node = StatsTreeNode {
            name: "parent".to_string(),
            direct_bytes: 10,
            direct_messages: 1,
            children: vec![leaf("a", 20, 2), leaf("b", 30, 3)],
        };
        assert_eq!(node.total_bytes(), 60);
        assert_eq!(node.total_messages(), 6);
    }

    #[test]
    fn stats_tree_node_nested_totals() {
        let inner = StatsTreeNode {
            name: "mid".to_string(),
            direct_bytes: 5,
            direct_messages: 1,
            children: vec![leaf("leaf", 10, 2)],
        };
        let root = StatsTreeNode {
            name: "root".to_string(),
            direct_bytes: 1,
            direct_messages: 1,
            children: vec![inner],
        };
        // root: 1 + mid(5 + leaf(10)) = 16
        assert_eq!(root.total_bytes(), 16);
        assert_eq!(root.total_messages(), 4);
    }

    #[test]
    fn child_mut_creates_new_child() {
        let mut node = leaf("parent", 0, 0);
        assert!(node.children.is_empty());
        let child = node.child_mut("new_child");
        child.direct_bytes = 42;
        assert_eq!(node.children.len(), 1);
        assert_eq!(node.children[0].name, "new_child");
        assert_eq!(node.children[0].direct_bytes, 42);
    }

    #[test]
    fn child_mut_returns_existing_child() {
        let mut node = StatsTreeNode {
            name: "parent".to_string(),
            direct_bytes: 0,
            direct_messages: 0,
            children: vec![leaf("existing", 10, 1)],
        };
        let child = node.child_mut("existing");
        child.direct_bytes += 5;
        assert_eq!(node.children.len(), 1);
        assert_eq!(node.children[0].direct_bytes, 15);
    }

    #[test]
    fn sort_roots_by_bytes_descending() {
        let mut roots = vec![leaf("small", 10, 100), leaf("big", 1000, 1)];
        StatsTreeNode::sort_roots(&mut roots, SortBy::Bytes);
        assert_eq!(roots[0].name, "big");
        assert_eq!(roots[1].name, "small");
    }

    #[test]
    fn sort_roots_by_messages_descending() {
        let mut roots = vec![leaf("few_msgs", 1000, 1), leaf("many_msgs", 10, 100)];
        StatsTreeNode::sort_roots(&mut roots, SortBy::Messages);
        assert_eq!(roots[0].name, "many_msgs");
        assert_eq!(roots[1].name, "few_msgs");
    }

    #[test]
    fn sort_roots_recurses_into_children() {
        let mut roots = vec![StatsTreeNode {
            name: "parent".to_string(),
            direct_bytes: 0,
            direct_messages: 0,
            children: vec![leaf("child_small", 1, 0), leaf("child_big", 100, 0)],
        }];
        StatsTreeNode::sort_roots(&mut roots, SortBy::Bytes);
        assert_eq!(roots[0].children[0].name, "child_big");
        assert_eq!(roots[0].children[1].name, "child_small");
    }

    #[test]
    fn flatten_all_sums_across_call_sites() {
        // "compare" appears under two different parents
        let roots = vec![
            branch("search", vec![leaf("compare", 100, 10)]),
            branch("insert", vec![leaf("compare", 200, 20)]),
        ];
        let flat = StatsTreeNode::flatten_all(&roots);
        let compare = flat.get("compare").expect("compare should be present");
        assert_eq!(compare.total_bytes, 300);
        assert_eq!(compare.total_messages, 30);
    }

    #[test]
    fn flatten_all_removes_zero_byte_entries() {
        let roots = vec![branch("root", vec![leaf("zero_fn", 0, 5)])];
        let flat = StatsTreeNode::flatten_all(&roots);
        assert!(!flat.contains_key("zero_fn"));
        assert!(!flat.contains_key("root"));
    }

    #[test]
    fn flatten_all_includes_parent_totals() {
        let roots = vec![StatsTreeNode {
            name: "parent".to_string(),
            direct_bytes: 10,
            direct_messages: 1,
            children: vec![leaf("child", 20, 2)],
        }];
        let flat = StatsTreeNode::flatten_all(&roots);
        let parent = flat.get("parent").expect("parent should be present");
        // parent total_bytes = 10 (direct) + 20 (child) = 30
        assert_eq!(parent.total_bytes, 30);
    }

    // -----------------------------------------------------------------------
    // SortBy
    // -----------------------------------------------------------------------

    #[test]
    fn sort_by_key_bytes() {
        let node = leaf("x", 42, 7);
        assert_eq!(SortBy::Bytes.key(&node), 42);
    }

    #[test]
    fn sort_by_key_messages() {
        let node = leaf("x", 42, 7);
        assert_eq!(SortBy::Messages.key(&node), 7);
    }
}

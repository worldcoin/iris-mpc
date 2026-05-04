use std::fmt;

use super::stats::StatsTreeNode;

pub(crate) enum Indent {
    Null,
    Line,
    Fork,
    Turn,
}

impl Indent {
    /// Box-drawing character for horizontal rules / table separators.
    pub(crate) const RULE: &str = "─";

    pub(crate) fn repr(&self) -> &'static str {
        match self {
            Self::Null => "   ",
            Self::Line => "│  ",
            Self::Fork => "├─ ",
            Self::Turn => "└─ ",
        }
    }
}

pub(crate) type IndentVec = Vec<Indent>;

pub(crate) fn indent_string(indent: &[Indent]) -> String {
    indent.iter().map(|i| i.repr()).collect()
}

pub(crate) struct BytesDisplay(pub f64);

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

pub(crate) fn format_bytes(b: u64) -> String {
    format!("{}", BytesDisplay(b as f64))
}

pub(crate) fn display_width(s: &str) -> usize {
    s.chars().count()
}

pub(crate) fn pad_right(s: &str, width: usize) -> String {
    let dw = display_width(s);
    if dw >= width {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(width - dw))
    }
}

pub(crate) fn to_percentage(part: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        part as f64 * 100.0 / total as f64
    }
}

pub(crate) struct StatsNodeDisplay {
    pub formatted_name: String,
    pub direct_bytes: u64,
    pub direct_messages: u64,
    pub total_bytes: u64,
    pub total_messages: u64,
}

pub(crate) fn collect_display_names(
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

pub(crate) fn tree_node_to_json(node: &StatsTreeNode) -> serde_json::Value {
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

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // BytesDisplay
    // -----------------------------------------------------------------------

    #[test]
    fn bytes_display_zero() {
        assert_eq!(format!("{}", BytesDisplay(0.0)), "0 B");
    }

    #[test]
    fn bytes_display_small() {
        assert_eq!(format!("{}", BytesDisplay(42.0)), "42 B");
    }

    #[test]
    fn bytes_display_boundary_999() {
        assert_eq!(format!("{}", BytesDisplay(999.0)), "999 B");
    }

    #[test]
    fn bytes_display_one_kb() {
        // 1000 bytes → 1.00 kB (< 10, so 2 decimal places)
        assert_eq!(format!("{}", BytesDisplay(1000.0)), "1.00 kB");
    }

    #[test]
    fn bytes_display_mid_kb() {
        // 5500 → 5.50 kB
        assert_eq!(format!("{}", BytesDisplay(5500.0)), "5.50 kB");
    }

    #[test]
    fn bytes_display_tens_kb() {
        // 45_000 → 45.0 kB
        assert_eq!(format!("{}", BytesDisplay(45_000.0)), "45.0 kB");
    }

    #[test]
    fn bytes_display_hundreds_kb() {
        // 500_000 → 500 kB
        assert_eq!(format!("{}", BytesDisplay(500_000.0)), "500 kB");
    }

    #[test]
    fn bytes_display_one_mb() {
        assert_eq!(format!("{}", BytesDisplay(1_000_000.0)), "1.00 MB");
    }

    #[test]
    fn bytes_display_one_gb() {
        assert_eq!(format!("{}", BytesDisplay(1_000_000_000.0)), "1.00 GB");
    }

    #[test]
    fn bytes_display_large_gb() {
        // > 999 GB wraps to TB-scale but code prints as GB
        assert_eq!(format!("{}", BytesDisplay(5_000_000_000_000.0)), "5000 GB");
    }

    // -----------------------------------------------------------------------
    // format_bytes (wrapper)
    // -----------------------------------------------------------------------

    #[test]
    fn format_bytes_delegates_to_display() {
        assert_eq!(format_bytes(2048), "2.05 kB");
    }

    // -----------------------------------------------------------------------
    // pad_right / display_width
    // -----------------------------------------------------------------------

    #[test]
    fn pad_right_shorter_string() {
        assert_eq!(pad_right("hi", 6), "hi    ");
    }

    #[test]
    fn pad_right_exact_width() {
        assert_eq!(pad_right("hello", 5), "hello");
    }

    #[test]
    fn pad_right_longer_string() {
        assert_eq!(pad_right("toolong", 3), "toolong");
    }

    #[test]
    fn display_width_ascii() {
        assert_eq!(display_width("hello"), 5);
    }

    #[test]
    fn display_width_empty() {
        assert_eq!(display_width(""), 0);
    }

    // -----------------------------------------------------------------------
    // to_percentage
    // -----------------------------------------------------------------------

    #[test]
    fn to_percentage_normal() {
        let pct = to_percentage(25, 100);
        assert!((pct - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn to_percentage_zero_total() {
        assert!((to_percentage(10, 0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn to_percentage_full() {
        let pct = to_percentage(100, 100);
        assert!((pct - 100.0).abs() < f64::EPSILON);
    }
}

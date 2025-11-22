use std::{
    collections::BTreeMap,
    io::Cursor,
    path::{Path, PathBuf},
};

use clap::{Parser, Subcommand};
use iris_mpc_cpu::{
    hnsw::graph::graph_diff::{
        explicit::{ExplicitNeighborhoodDiffer, SortBy},
        jaccard::DetailedJaccardDiffer,
        node_equiv::ensure_node_equivalence,
        run_diff,
    },
    utils::serialization::graph::{
        check_valid_graph_formats, check_valid_graph_pair_formats, read_graph,
        read_graph_from_file, read_graph_pair, read_graph_pair_from_file, write_graph_pair_to_file,
        write_graph_to_file, GraphFormat, GRAPH_FORMAT_CURRENT,
    },
};

use eyre::{eyre, Result};

#[derive(Parser)]
#[command(name = "app")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Upgrade a graph or graph pair from an older serialization format to the
    /// current stable serialization format.  If a specific older format is
    /// known, it can be specified with the `--src-format` flag.  Otherwise, the
    /// utility will try all known graph serialization formats to find a format
    /// which works, if any.  If a graph (pair) can be deserialized with
    /// multiple formats, the utility will display which formats are possible
    /// and halt.  In this case, a specific serialization format must be
    /// provided to eliminate the ambiguity.
    UpgradeFormat {
        /// Source file
        src_file: PathBuf,

        /// Source graph file format, default behavior is to trial different formats
        #[arg(long)]
        src_format: Option<GraphFormat>,

        /// Destination file
        dst_file: PathBuf,

        /// Flag to upgrade a graph pair
        #[arg(long)]
        pair: bool,
    },
    /// Split a serialized graph pair into two individual graphs, and write these to
    /// file.
    SplitPair {
        /// Source file for graph pair
        src_file: PathBuf,

        /// Source graph file format for pair file, defaults to "current"
        #[arg(long, value_enum, default_value_t = GraphFormat::Current)]
        src_format: GraphFormat,

        /// Destination file for left graph
        dst_file_left: PathBuf,

        /// Destination file for right graph
        dst_file_right: PathBuf,
    },
    /// Combine two individual serialized graphs into a graph pair, and write
    /// this to file.
    MakePair {
        /// Source file for left graph
        src_file_left: PathBuf,

        /// Source graph file format for left graph, defaults to "current"
        #[arg(long, value_enum, default_value_t = GraphFormat::Current)]
        src_format_left: GraphFormat,

        /// Source file for right graph
        src_file_right: PathBuf,

        /// Source graph file format for right graph, defaults to "current"
        #[arg(long, value_enum, default_value_t = GraphFormat::Current)]
        src_format_right: GraphFormat,

        /// Destination file for graph pair
        dst_file: PathBuf,
    },
    /// Display statistics about a serialized graph.
    Stat {
        /// Source file
        src_file: PathBuf,

        /// Source graph file format, defaults to "current"
        #[arg(long, value_enum, default_value_t = GraphFormat::Current)]
        src_format: GraphFormat,
    },
    /// Run a diff utility between two serialized graphs.
    Diff {
        /// Source file for graph 1
        src_file_1: PathBuf,

        /// Source graph file format for graph 1, defaults to "current"
        #[arg(long, value_enum, default_value_t = GraphFormat::Current)]
        src_format_1: GraphFormat,

        /// Source file for graph 2
        src_file_2: PathBuf,

        /// Source graph file format for graph 2, defaults to "current"
        #[arg(long, value_enum, default_value_t = GraphFormat::Current)]
        src_format_2: GraphFormat,

        /// Diff specification subcommand
        #[command(subcommand)]
        diff_spec: Option<DiffSpec>,
    },
}

#[derive(Subcommand)]
enum DiffSpec {
    /// Use the detailed Jaccard differ
    Jaccard {
        /// The number of most dissimilar nodes to display per layer
        #[arg(short = 'n', long, default_value_t = 15)]
        num_display: usize,
    },

    /// Use the explicit neighborhood differ, sorted by node index
    Links {
        /// Sorting method for display
        #[arg(long, value_enum, default_value_t = SortBy::Index)]
        sort_by: SortBy,
    },
}

fn main() {
    let cli = Cli::parse();

    let res = match cli.command {
        Commands::UpgradeFormat {
            src_file,
            src_format,
            dst_file,
            pair,
        } => upgrade_format(src_file, src_format, dst_file, pair),
        Commands::SplitPair {
            src_file,
            src_format,
            dst_file_left,
            dst_file_right,
        } => split_graph_pair(src_file, src_format, dst_file_left, dst_file_right),
        Commands::MakePair {
            src_file_left,
            src_format_left,
            src_file_right,
            src_format_right,
            dst_file,
        } => make_graph_pair(
            src_file_left,
            src_format_left,
            src_file_right,
            src_format_right,
            dst_file,
        ),
        Commands::Stat {
            src_file,
            src_format,
        } => graph_statistics(src_file, src_format),
        Commands::Diff {
            src_file_1,
            src_format_1,
            src_file_2,
            src_format_2,
            diff_spec,
        } => diff_graphs(
            src_file_1,
            src_format_1,
            src_file_2,
            src_format_2,
            diff_spec,
        ),
    };

    if let Err(report) = res {
        eprintln!("{report}");
    }
}

/// Read a graph from file and write to file using the current standard serialization format
fn upgrade_format(
    src_file: PathBuf,
    src_format: Option<GraphFormat>,
    dst_file: PathBuf,
    pair: bool,
) -> Result<()> {
    println!("Reading data from file: {}", path_string(&src_file));
    let data =
        std::fs::read(&src_file).map_err(|e| eyre!("Unable to read source file :: {}", e))?;

    // Identify valid graph data formats

    let src_format = match src_format {
        Some(format) => Ok(format),
        None => {
            println!("Checking for valid graph formats for deserialization");
            let valid_formats = if pair {
                check_valid_graph_pair_formats(&data)
            } else {
                check_valid_graph_formats(&data)
            };

            match valid_formats.len() {
                0 => Err(eyre!("No graph format succesfully deserializes file data")),
                1 => Ok(*valid_formats.first().unwrap()),
                2.. => {
                    let valid_formats_string = valid_formats
                        .into_iter()
                        .map(|gr_fmt| gr_fmt.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    Err(eyre!("Multiple valid graph formats found: {valid_formats_string}\nPlease specify a particular deserialization format with the --src-format option"))
                }
            }
        }
    }?;

    // Deserialize and reserialize

    if pair {
        println!("Deserializing data with graph format {:?}", src_format);
        let mut reader = Cursor::new(&data);

        let graph_pair = read_graph_pair(&mut reader, src_format).map_err(|e| {
            eyre!(
                "Unable to deserialize graph pair using format {} :: {}",
                src_format,
                e
            )
        })?;

        println!(
            "Writing graph pair to file using current stable graph format {}: {}",
            GRAPH_FORMAT_CURRENT,
            path_string(&dst_file)
        );
        write_graph_pair_to_file(dst_file, graph_pair)
            .map_err(|e| eyre!("Unable to write graph pair to file :: {}", e))?;
    } else {
        println!("Deserializing data with graph format {:?}", src_format);
        let mut reader = Cursor::new(&data);

        let graph = read_graph(&mut reader, src_format).map_err(|e| {
            eyre!(
                "Unable to deserialize graph using format {} :: {}",
                src_format,
                e
            )
        })?;

        println!(
            "Writing graph to file using current stable graph format {}: {}",
            GRAPH_FORMAT_CURRENT,
            path_string(&dst_file)
        );
        write_graph_to_file(dst_file, graph)
            .map_err(|e| eyre!("Unable to write graph to file :: {}", e))?;
    }

    println!("Done!");

    Ok(())
}

/// Read a graph pair from file, and write the left and write graphs from the pair to file
fn split_graph_pair(
    src_file: PathBuf,
    src_format: GraphFormat,
    dst_file_left: PathBuf,
    dst_file_right: PathBuf,
) -> Result<()> {
    println!("Reading graph pair from file: {}", path_string(&src_file));
    let [graph_left, graph_right] =
        read_graph_pair_from_file(src_file, src_format).map_err(|e| {
            eyre!(
                "Unable to read graph pair from file using format {} :: {}",
                src_format,
                e
            )
        })?;

    println!(
        "Writing left graph to file: {}",
        path_string(&dst_file_left)
    );
    write_graph_to_file(dst_file_left, graph_left)
        .map_err(|e| eyre!("Unable to write left graph to file :: {}", e))?;

    println!(
        "Writing right graph to file: {}",
        path_string(&dst_file_right)
    );
    write_graph_to_file(dst_file_right, graph_right)
        .map_err(|e| eyre!("Unable to write right graph to file :: {}", e))?;

    println!("Done!");

    Ok(())
}

/// Read left and right graphs from file, and write a graph pair with these graphs to file
fn make_graph_pair(
    src_file_left: PathBuf,
    src_format_left: GraphFormat,
    src_file_right: PathBuf,
    src_format_right: GraphFormat,
    dst_file: PathBuf,
) -> Result<()> {
    println!(
        "Reading left graph from file: {}",
        path_string(&src_file_left)
    );
    let graph_left = read_graph_from_file(src_file_left, src_format_left).map_err(|e| {
        eyre!(
            "Unable to read left graph from file using format {} :: {}",
            src_format_left,
            e
        )
    })?;

    println!(
        "Reading right graph from file: {}",
        path_string(&src_file_right)
    );
    let graph_right = read_graph_from_file(src_file_right, src_format_right).map_err(|e| {
        eyre!(
            "Unable to read right graph from file using format {} :: {}",
            src_format_right,
            e
        )
    })?;

    println!("Writing graph pair to file: {}", path_string(&dst_file));
    write_graph_pair_to_file(dst_file, [graph_left, graph_right])
        .map_err(|e| eyre!("Unable to write graph pair to file :: {}", e))?;

    println!("Done!");

    Ok(())
}

fn graph_statistics(src_file: PathBuf, src_format: GraphFormat) -> Result<()> {
    println!("Reading graph from file: {}", path_string(&src_file));
    let graph = read_graph_from_file(src_file, src_format).map_err(|e| {
        eyre!(
            "Unable to read graph from file using format {} :: {}",
            src_format,
            e
        )
    })?;

    println!("Succesfully read graph from file.\n");

    println!("=== Graph Statistics ===");

    if matches!(src_format, GraphFormat::Current) {
        println!("File format: {}", GRAPH_FORMAT_CURRENT);
    } else {
        println!("File format: {}", src_format);
    }
    println!("Checksum: {}", graph.checksum());
    println!();

    let node_counts: Vec<_> = graph.layers.iter().map(|l| l.links.len()).collect();
    println!("Total nodes: {}", node_counts.iter().sum::<usize>());
    for (lc, count) in node_counts.iter().enumerate() {
        println!("Layer {} nodes: {}", lc, count);
    }
    println!();

    let eps_count = graph.entry_points.len();
    println!("Total entry points: {}", eps_count);

    let mut eps_layer_counts: BTreeMap<usize, usize> = BTreeMap::new();
    for ep in graph.entry_points.iter() {
        *eps_layer_counts.entry(ep.layer).or_insert(0) += 1;
    }

    for (key, value) in eps_layer_counts.iter() {
        println!("Layer {} entry points: {}", key, value);
    }
    if eps_layer_counts.len() > 1 {
        println!("WARNING: entry points present in multiple layers");
    }

    Ok(())
}

fn diff_graphs(
    src_file_1: PathBuf,
    src_format_1: GraphFormat,
    src_file_2: PathBuf,
    src_format_2: GraphFormat,
    diff_spec: Option<DiffSpec>,
) -> Result<()> {
    println!("Reading graph 1 from file: {}", path_string(&src_file_1));
    let graph_1 = read_graph_from_file(&src_file_1, src_format_1).map_err(|e| {
        eyre!(
            "Unable to read graph 1 from file using format {} :: {}",
            src_format_1,
            e
        )
    })?;

    println!("Reading graph 2 from file: {}", path_string(&src_file_2));
    let graph_2 = read_graph_from_file(&src_file_2, src_format_2).map_err(|e| {
        eyre!(
            "Unable to read graph 2 from file using format {} :: {}",
            src_format_2,
            e
        )
    })?;

    let diff_spec = diff_spec.unwrap_or(DiffSpec::Jaccard { num_display: 15 });

    println!();
    println!("=== Graph Diff ===");

    if graph_1 == graph_2 {
        println!("Graphs are identical, including ordering of neighborhoods");
        return Ok(());
    }

    let node_equiv_result = ensure_node_equivalence(&graph_1, &graph_2);
    if let Err(err) = node_equiv_result {
        println!("Graphs are not node-equivalent\n Reason: {:#?}", err);
    } else {
        match diff_spec {
            DiffSpec::Jaccard { num_display } => {
                println!(
                    "Using Jaccard differ displaying {} most dissimilar nodes per layer\n",
                    num_display
                );
                let differ = DetailedJaccardDiffer::new(num_display);
                let result = run_diff(&graph_1, &graph_2, differ);
                println!("{result}");
            }
            DiffSpec::Links { sort_by } => {
                println!(
                    "Using explicit neighborhood differ, sorting results by {} ordering\n",
                    sort_by
                );
                let differ = ExplicitNeighborhoodDiffer::new(sort_by);
                let result = run_diff(&graph_1, &graph_2, differ);
                println!("{result}");
            }
        };
    }

    Ok(())
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

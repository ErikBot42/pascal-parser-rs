#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::implicit_return)]
#![allow(clippy::print_stdout)]
#![allow(clippy::print_stderr)]
#![allow(clippy::use_debug)]
#![allow(clippy::no_effect_underscore_binding)]

use pascal_parser_rust::ProgParser;
use pascal_parser_rust::State;
use std::env::args;
use std::time::Instant;
fn main() -> Result<(), String> {
    let execute = args().find(|s| s == "-x").is_some();
    let compile = args().find(|s| s == "-c").is_some();
    let help = args().find(|s| s == "-h" || s == "--help").is_some();

    if help {
        eprintln!("Usage (any order):");
        eprintln!("   cargo run -- [filename] [-x] [-c] [-h] [--help]");
        eprintln!();
        eprintln!("   filename: input filename (can also read from stdin)");
        eprintln!("   -x: execute program");
        eprintln!("   -c: compile program to rust code");
        eprintln!("   -h or --help: print this help text");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("    cargo run -- example_programs/prim.pas -x");
        eprintln!("    cargo run -- example_programs/fib.pas -c");
        eprintln!("    cat example_programs/fib.pas | cargo run -- -x");
        Ok(())
    } else if let Some(program) = args()
        .skip(1)
        .map(std::fs::read_to_string)
        .map(Result::ok)
        .flatten()
        .next()
        .or_else(|| {
            eprintln!("Reading program from stdin... (Ctrl-D to send EOF)");
            std::io::read_to_string(std::io::stdin()).ok()
        })
    {
        let mut state = State::default();
        eprintln!("Program:\n{program}\n$");
        ProgParser::new()
            .parse(&mut state, &program)
            .map_err(|e| {
                eprintln!("COULD NOT PARSE!");
                eprintln!();
                e.to_string()
            })
            .map(|parsed| {
                eprintln!("PARSE OK!");
                eprintln!("{:?}", &state.variable_table);
                eprintln!("{:?}", &parsed);
                if execute {
                    eprintln!("executing...");
                    let start = Instant::now();
                    parsed.execute(&mut state);
                    let duration = start.elapsed();
                    eprintln!("done...");
                    eprintln!("{:?}", &state.variable_table);
                    eprintln!("Execution completed in {duration:?}");
                }
                if compile {
                    eprintln!("compiling...");
                    println!("{}", parsed.compile(&state));
                }
            })
    } else {
        Err("Error reading from file and standard input".to_string())
    }
}

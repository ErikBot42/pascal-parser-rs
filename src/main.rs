#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::implicit_return)]
#![allow(clippy::print_stdout)]
#![allow(clippy::print_stderr)]
#![allow(clippy::use_debug)]
#![allow(clippy::no_effect_underscore_binding)]

//use crate::ProgParser;
//use pascal_parser_rust::State;
use pascal_parser_rust::ProgParser;
use pascal_parser_rust::State;
use std::time::Instant;
fn main() {
    let _p0 = "
program testok1(input, output);  
var A, B, C: integer;  

begin
    A := B + C * 2;
    B := A + integer[B * 2 + C];
    real[3] := 3.4;
    C := 5;
    while C > 2 do
    begin
        C := C - 1
    end
end.
";
    let _fib = "
program fib(input, output);  
var A, B, C: integer;  

begin
    A := 0;
    B := 1;
    C := 1;
    while C < 1000 do
    begin
        C := A + B;
        A := B + C;
        B := C + A
    end
end.
";
    let prim = "
program prim(input, output);  
var i, j, maxNums: integer;  

begin
    maxNums := 10000;
    i := 2;
    while i < maxNums do
    begin
        while i < maxNums && boolean[i] do
        begin
            i := i + 1
        end;
        if i < maxNums then
        begin
            writeLn i;
            boolean[i] := true;
            j := 0;
            while j <= maxNums do
            begin
                boolean[j] := true;
                j := j + i
            end
        end
    end
end.
";

    let program = prim;

    let mut state = State::new();

    let execute = true;
    let compile = false;

    eprintln!("Program:\n{program}\n$");
    match ProgParser::new().parse(&mut state, program) {
        Err(e) => {
            eprintln!("{:?}", &state.variable_table);
            eprintln!("{e:?}");
        }
        Ok(parsed) => {
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
        }
    }
}

#![allow(clippy::use_self)]
#![allow(clippy::implicit_return)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::enum_glob_use)]
#![allow(clippy::pub_use)]

#[macro_use]
extern crate lalrpop_util;

mod pascal {
    #![allow(clippy::pedantic)]
    #![allow(clippy::restriction)]
    #![allow(clippy::cargo)]
    #![allow(clippy::style)]
    #![allow(clippy::perf)]
    #![allow(clippy::complexity)]
    #![allow(clippy::suspicious)]
    #![allow(clippy::correctness)]
    #![allow(clippy::nursery)]
    lalrpop_mod!(pub pascal);
}
pub mod ast;
pub use ast::State;
pub use pascal::pascal::ProgParser;

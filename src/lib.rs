

#![allow(clippy::use_self)]
#![allow(clippy::implicit_return)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::enum_glob_use)]

#[macro_use]
extern crate lalrpop_util;

lalrpop_mod!(pub pascal);
pub mod ast;
pub use ast::State;
pub use pascal::ProgParser;

#![allow(clippy::pattern_type_mismatch)]
#![allow(clippy::exhaustive_enums)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_inline_in_public_items)]
#![allow(clippy::integer_arithmetic)]
#![allow(clippy::float_arithmetic)]
#![allow(clippy::float_cmp)]
#![allow(clippy::arithmetic_side_effects)]
#![allow(clippy::modulo_arithmetic)]
#![allow(clippy::integer_division)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::default_numeric_fallback)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::exhaustive_structs)]
#![allow(clippy::shadow_reuse)]
#![allow(clippy::shadow_unrelated)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::missing_inline_in_public_items)]
// review later:
#![allow(clippy::print_stdout)]
#![allow(clippy::indexing_slicing)]
#![allow(clippy::map_err_ignore)]

use core::fmt::Write as _;
use std::collections::HashMap;
extern crate derive_more;
use derive_more::{Display, From};

const UNCHECKED_INDEXING: bool = true;

#[non_exhaustive]
#[derive(Debug)]
pub struct Prog {
    name: String,
    stat_list: StatList,
}
impl Prog {
    pub const fn new(name: String, stat_list: StatList) -> Self {
        Self { name, stat_list }
    }
    pub fn execute(&self, state: &mut State) {
        println!("Running program {}", self.name);
        self.stat_list.execute(state);
    }
    pub fn compile(&self, state: &State) -> String {
        format!(
            "fn main() {{

// Static prelude
use std::io::Write;
let __internal_start_time= std::time::Instant::now();
let mut __internal_stdout_print_lock = std::io::stdout().lock();
let mut real: Vec<f64> = (0..{:?}).map(|_| Default::default()).collect();
let mut integer: Vec<i32> = (0..{:?}).map(|_| Default::default()).collect();
let mut boolean: Vec<bool> = (0..{:?}).map(|_| Default::default()).collect();

// User defined variables
{}
// Code
{}
write!(__internal_stdout_print_lock, \"Done in {{:?}}\", __internal_start_time.elapsed());
}}
",
            state.max_size,
            state.max_size,
            state.max_size,
            state.variable_table.compile(),
            self.stat_list.compile(state),
        )
    }
}
#[derive(Debug)]
pub struct State {
    pub variable_table: VariableTable,
    mem_real: Vec<f64>,
    mem_bool: Vec<bool>,
    mem_int: Vec<i32>,
    pub executed_statements: usize,
    max_size: usize,
}
impl State {
    pub fn new() -> Self {
        let max_size = 256 * 1024;
        Self {
            variable_table: VariableTable::new(),
            mem_real: (0..max_size).map(|_| Default::default()).collect(),
            mem_bool: (0..max_size).map(|_| Default::default()).collect(),
            mem_int: (0..max_size).map(|_| Default::default()).collect(),
            executed_statements: 0,
            max_size,
        }
    }
    pub fn push_variable(
        &mut self,
        name: String,
        variable_type: Type,
    ) -> Option<VariableReference> {
        self.variable_table.push(name, variable_type)
    }
    pub fn set_variable_value(&mut self, v: VariableReference, val: Value) {
        self.variable_table.set_value(v, val);
    }
    pub fn get_variable_value(&self, v: VariableReference) -> Value {
        self.variable_table.get_value(v)
    }
    pub fn get_variable(&self, name: &String) -> Option<VariableReference> {
        self.variable_table.get(name)
    }
    pub fn get_variable_type(&self, v: VariableReference) -> Type {
        self.variable_table.get_type(v)
    }
    pub fn read_memory(&self, typ: Type, addr: Value) -> Value {
        use Type::*;
        let addr: usize = addr.into();
        match typ {
            Boolean => self.mem_bool[addr].into(),
            Integer => self.mem_int[addr].into(),
            Real => self.mem_real[addr].into(),
        }
    }
    pub fn write_memory(&mut self, typ: Type, addr: Value, val: Value) {
        use Type::*;
        let addr: usize = addr.into();
        match typ {
            Boolean => self.mem_bool[addr] = val.into(),
            Integer => self.mem_int[addr] = val.into(),
            Real => self.mem_real[addr] = val.into(),
        }
    }
    pub fn set_bool_variable_value(&mut self, v: VariableReference, val: bool) {
        self.set_variable_value(v, Value::Boolean(val));
    }
    pub fn set_int_variable_value(&mut self, v: VariableReference, val: i32) {
        self.set_variable_value(v, Value::Integer(val));
    }
    pub fn set_real_variable_value(&mut self, v: VariableReference, val: f64) {
        self.set_variable_value(v, Value::Real(val));
    }
    pub fn get_int_variable_value(&self, v: VariableReference) -> i32 {
        if let Value::Integer(i) = self.get_variable_value(v) {
            i
        } else {
            panic!("internal error")
        }
    }
    pub fn get_bool_variable_value(&self, v: VariableReference) -> bool {
        if let Value::Boolean(i) = self.get_variable_value(v) {
            i
        } else {
            panic!("internal error")
        }
    }
    pub fn get_real_variable_value(&self, v: VariableReference) -> f64 {
        if let Value::Real(i) = self.get_variable_value(v) {
            i
        } else {
            panic!("internal error")
        }
    }
    pub fn read_memory_bool(&self, addr: i32) -> bool {
        self.mem_bool[addr as usize]
    }
    pub fn read_memory_int(&self, addr: i32) -> i32 {
        self.mem_int[addr as usize]
    }
    pub fn read_memory_real(&self, addr: i32) -> f64 {
        self.mem_real[addr as usize]
    }
    pub fn write_memory_bool(&mut self, addr: i32, data: bool) {
        self.mem_bool[addr as usize] = data;
    }
    pub fn write_memory_int(&mut self, addr: i32, data: i32) {
        self.mem_int[addr as usize] = data;
    }
    pub fn write_memory_real(&mut self, addr: i32, data: f64) {
        self.mem_real[addr as usize] = data;
    }
}

#[derive(Debug)]
pub struct VariableTable {
    table: HashMap<String, VariableReference>,
    variables: Vec<(String, Value)>,
    reals_names: Vec<String>,
    reals: Vec<f64>,
    ints_names: Vec<String>,
    ints: Vec<i32>,
    bools_names: Vec<String>,
    bools: Vec<bool>,
}
impl VariableTable {
    fn new() -> Self {
        Self {
            table: HashMap::default(),
            variables: Vec::default(),
            reals: Vec::default(),
            reals_names: Vec::default(),
            ints: Vec::default(),
            ints_names: Vec::default(),
            bools: Vec::default(),
            bools_names: Vec::default(),
        }
    }
    fn push(&mut self, name: String, variable_type: Type) -> Option<VariableReference> {
        if self.table.get(&name).is_some() {
            None
        } else {
            let id = VariableReference(self.variables.len(), variable_type);
            self.table.insert(name.clone(), id);
            use Type::*;
            match variable_type {
                Boolean => {
                    self.bools.push(Default::default());
                    self.bools_names.push(name.clone())
                }
                Integer => {
                    self.ints.push(Default::default());
                    self.ints_names.push(name.clone())
                }
                Real => {
                    self.reals.push(Default::default());
                    self.reals_names.push(name.clone())
                }
            }
            self.variables
                .push((name, Value::default_from_type(variable_type)));
            Some(id)
        }
    }
    fn get(&self, name: &String) -> Option<VariableReference> {
        self.table.get(name).copied()
    }
    pub fn get_type(&self, v: VariableReference) -> Type {
        self.variables[v.0].1.get_type()
    }
    fn get_value(&self, v: VariableReference) -> Value {
        self.variables[v.0].1
    }
    fn get_name(&self, v: VariableReference) -> String {
        self.variables[v.0].0.clone()
    }
    fn set_value(&mut self, v: VariableReference, val: Value) {
        self.variables[v.0].1 = val.coerce_into(self.variables[v.0].1.get_type());
    }
    fn compile(&self) -> String {
        let mut buffer = String::default();
        for (variable, value) in &self.variables {
            writeln!(
                &mut buffer,
                "let mut {}: {} = {};",
                variable,
                value.get_type().as_rust_str(),
                value.compile()
            )
            .unwrap();
        }
        buffer
    }
}
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct VariableReference(usize, Type);
impl VariableReference {
    pub fn get_type(self, state: &State) -> Type {
        state.get_variable_type(self)
    }
    fn compile(self, state: &State) -> String {
        state.variable_table.variables[self.0].0.clone()
    }
}

#[derive(Debug)]
pub enum BoolAssignable {
    Mem(IntExpr),
    Variable(VariableReference),
}
impl BoolAssignable {
    fn assign(&self, state: &mut State, val: bool) {
        use BoolAssignable::*;
        match self {
            Mem(e) => state.write_memory_bool(e.evaluate(state), val),
            Variable(v) => state.set_bool_variable_value(*v, val),
        }
    }
    fn compile(&self, state: &State) -> String {
        use BoolAssignable::*;
        match self {
            Mem(e) => format!("boolean[{}]", e.compile(state)),
            Variable(v) => v.compile(state),
        }
    }
}
#[derive(Debug)]
pub enum IntAssignable {
    Mem(IntExpr),
    Variable(VariableReference),
}
impl IntAssignable {
    fn assign(&self, state: &mut State, val: i32) {
        use IntAssignable::*;
        match self {
            Mem(e) => state.write_memory_int(e.evaluate(state), val),
            Variable(v) => state.set_int_variable_value(*v, val),
        }
    }
    fn compile(&self, state: &State) -> String {
        use IntAssignable::*;
        match self {
            Mem(e) => format!("integer[{}]", e.compile(state)),
            Variable(v) => v.compile(state),
        }
    }
}
#[derive(Debug)]
pub enum RealAssignable {
    Mem(IntExpr),
    Variable(VariableReference),
}
impl RealAssignable {
    fn assign(&self, state: &mut State, val: f64) {
        use RealAssignable::*;
        match self {
            Mem(e) => state.write_memory_real(e.evaluate(state), val),
            Variable(v) => state.set_real_variable_value(*v, val),
        }
    }
    fn compile(&self, state: &State) -> String {
        use RealAssignable::*;
        match self {
            Mem(e) => format!("real[{}]", e.compile(state)),
            Variable(v) => v.compile(state),
        }
    }
}

#[derive(Debug, From)]
pub enum Assignable {
    Bool(BoolAssignable),
    Int(IntAssignable),
    Real(RealAssignable),
}
impl Assignable {
    pub fn variable(v: VariableReference, state: &State) -> Self {
        match state.get_variable_type(v) {
            Type::Boolean => BoolAssignable::Variable(v).into(),
            Type::Integer => IntAssignable::Variable(v).into(),
            Type::Real => RealAssignable::Variable(v).into(),
        }
    }
    pub fn memory(t: Type, e: Expr, state: &State) -> Self {
        match t {
            Type::Boolean => BoolAssignable::Mem(e.cast_int()).into(),
            Type::Integer => IntAssignable::Mem(e.cast_int()).into(),
            Type::Real => RealAssignable::Mem(e.cast_int()).into(),
        }
    }
    const fn get_type(&self) -> Type {
        match self {
            Assignable::Bool(_) => Type::Boolean,
            Assignable::Int(_) => Type::Integer,
            Assignable::Real(_) => Type::Real,
        }
    }
}

#[derive(Debug)]
pub enum Stat {
    AssignBool(BoolAssignable, BoolExpr),
    AssignReal(RealAssignable, RealExpr),
    AssignInt(IntAssignable, IntExpr),
    If(BoolExpr, StatList),
    While(BoolExpr, StatList),
    Print(Expr),
    PrintLn(Expr),
}
impl Stat {
    pub fn assign(a: Assignable, e: Expr) -> Result<Self, String> {
        match (a, e) {
            (Assignable::Bool(a), Expr::Bool(e)) => Ok(Stat::AssignBool(a, e)),
            (Assignable::Int(a), Expr::Int(e)) => Ok(Stat::AssignInt(a, e)),
            (Assignable::Real(a), Expr::Real(e)) => Ok(Stat::AssignReal(a, e)),
            (a, e) => Err(format!("Invalid assign {a:?} with {e:?}")),
        }
    }
    fn execute(&self, state: &mut State) {
        use Stat::*;
        match self {
            AssignBool(a, e) => a.assign(state, e.evaluate(state)),
            AssignReal(a, e) => a.assign(state, e.evaluate(state)),
            AssignInt(a, e) => a.assign(state, e.evaluate(state)),
            If(e, s) => {
                if e.evaluate(state) {
                    s.execute(state);
                }
            }
            While(e, s) => {
                while e.evaluate(state) {
                    s.execute(state);
                }
            }
            Print(e) => print!("{}", e.evaluate(state)),
            PrintLn(e) => println!("{}", e.evaluate(state)),
        }
    }
    fn compile(&self, state: &State) -> String {
        use Stat::*;
        match self {
            AssignBool(a, e) => format!("{} = {};\n", a.compile(state), e.compile(state)),
            AssignReal(a, e) => format!("{} = {};\n", a.compile(state), e.compile(state)),
            AssignInt(a, e) => format!("{} = {};\n", a.compile(state), e.compile(state)),
            If(e, s) => format!("if {} {{\n{}}}\n", e.compile(state), s.compile(state)),
            While(e, s) => format!("while {} {{\n{}}}\n", e.compile(state), s.compile(state)),
            Print(e) => format!("print!(\"{{:?}}\"{});\n", e.compile(state)),
            PrintLn(e) => format!("println!(\"{{:?}}\"{});\n", e.compile(state)),
        }
    }
}

#[derive(Debug, Default)]
pub struct StatList(pub Vec<Stat>);
impl StatList {
    pub fn new() -> Self {
        Self::default()
    }
    fn execute(&self, state: &mut State) {
        self.0.iter().for_each(|s| s.execute(state));
    }
    fn compile(&self, state: &State) -> String {
        self.0
            .iter()
            .map(|statement| statement.compile(state))
            .fold(String::new(), |mut a, b| {
                a.push_str(&b);
                a
            })
    }
}

#[derive(Debug, Copy, Clone, PartialOrd, Ord, Eq, PartialEq, Hash)]
pub enum Type {
    Boolean,
    Integer,
    Real,
}
impl Type {
    const fn as_pascal_string(self) -> &'static str {
        use Type::*;
        match self {
            Boolean => "boolean",
            Integer => "integer",
            Real => "real",
        }
    }
    const fn as_rust_str(self) -> &'static str {
        use Type::*;
        match self {
            Boolean => "bool",
            Integer => "i32",
            Real => "f64",
        }
    }
}
#[derive(Debug, Copy, Clone)]
pub enum ValuePair {
    Integer(i32, i32),
    Real(f64, f64),
    Boolean(bool, bool),
}
#[derive(Debug, Copy, Clone, From, Display)]
pub enum Value {
    Integer(i32),
    Real(f64),
    Boolean(bool),
}
#[rustfmt::skip]
impl From<Value> for usize { fn from(val: Value) -> Self { val.coerce_integer() as usize } }
#[rustfmt::skip]
impl From<Value> for i32 { fn from(val: Value) -> Self { val.coerce_integer() } }
#[rustfmt::skip]
impl From<Value> for f64 { fn from(val: Value) -> Self { val.coerce_real() } }
#[rustfmt::skip]
impl From<Value> for bool { fn from(val: Value) -> Self { val.coerce_bool() } }
impl Value {
    const fn default_from_type(t: Type) -> Value {
        use Type::*;
        match t {
            Boolean => Value::Boolean(false),
            Integer => Value::Integer(0),
            Real => Value::Real(0.0),
        }
    }
    const fn coerce_into(self, typ: Type) -> Self {
        use Type::*;
        match typ {
            Boolean => Value::Boolean(self.coerce_bool()),
            Integer => Value::Integer(self.coerce_integer()),
            Real => Value::Real(self.coerce_real()),
        }
    }
    const fn coerce_real(self) -> f64 {
        use Value::*;
        match self {
            Integer(i) => i as f64,
            Real(r) => r,
            Boolean(b) => b as i32 as f64,
        }
    }
    const fn coerce_integer(self) -> i32 {
        use Value::*;
        match self {
            Integer(i) => i,
            Real(r) => r as i32,
            Boolean(b) => b as i32,
        }
    }
    const fn coerce_bool(self) -> bool {
        use Value::*;
        match self {
            Integer(i) => i != 0,
            Real(r) => (r as i32) != 0,
            Boolean(b) => b,
        }
    }
    const fn get_type(self) -> Type {
        use Value::*;
        match self {
            Integer(_) => Type::Integer,
            Real(_) => Type::Real,
            Boolean(_) => Type::Boolean,
        }
    }
    fn compile(self) -> String {
        use Value::*;
        match self {
            Integer(v) => format!("{v}"),
            Real(v) => format!("{v}"),
            Boolean(v) => format!("{v}"),
        }
    }
}
#[derive(Debug, Copy, Clone)]
pub enum BinaryOpcode {
    Mul,
    Div,
    Mod,
    Add,
    Sub,
    Equal,
    NotEqual,
    GreaterThan,
    GreaterEqual,
    LessThan,
    LessThanEqual,
    And,
    Or,
    Xor,

    BitAnd,
    BitOr,
    BitXor,
    BitShiftRight,
    BitShiftLeft,
}

#[derive(Debug, From)]
pub enum Expr {
    Bool(BoolExpr),
    Int(IntExpr),
    Real(RealExpr),
}
#[derive(Debug)]
enum ExprPair {
    Bool(BoolExpr, BoolExpr),
    Int(IntExpr, IntExpr),
    Real(RealExpr, RealExpr),
}
impl Expr {
    fn evaluate(&self, state: &State) -> Value {
        match self {
            Expr::Bool(e) => e.evaluate(state).into(),
            Expr::Int(e) => e.evaluate(state).into(),
            Expr::Real(e) => e.evaluate(state).into(),
        }
    }
    fn compile(&self, state: &State) -> String {
        match self {
            Expr::Bool(e) => e.compile(state),
            Expr::Int(e) => e.compile(state),
            Expr::Real(e) => e.compile(state),
        }
    }
    pub fn variable(v: VariableReference, state: &State) -> Expr {
        match state.get_variable_type(v) {
            Type::Boolean => BoolExpr::Variable(v).into(),
            Type::Integer => IntExpr::Variable(v).into(),
            Type::Real => RealExpr::Variable(v).into(),
        }
    }
    pub fn mem(mem_type: Type, e: Expr) -> Expr {
        match mem_type {
            Type::Boolean => BoolExpr::Mem(Box::new(e.cast_int())).into(),
            Type::Integer => IntExpr::Mem(Box::new(e.cast_int())).into(),
            Type::Real => RealExpr::Mem(Box::new(e.cast_int())).into(),
        }
    }
    pub fn value(v: Value) -> Expr {
        match v {
            Value::Integer(i) => IntExpr::Value(i).into(),
            Value::Real(r) => RealExpr::Value(r).into(),
            Value::Boolean(b) => BoolExpr::Value(b).into(),
        }
    }
    pub fn binary(a: Expr, b: Expr, op: BinaryOpcode) -> Result<Expr, String> {
        match a.coerce_equal(b) {
            ExprPair::Bool(a, b) => op
                .try_into()
                .map(|op| BoolExpr::Binary(Box::new(a), op, Box::new(b)).into()),
            ExprPair::Int(a, b) => {
                let a = Box::new(a);
                let b = Box::new(b);
                match (op.try_into(), op.try_into()) {
                    (Ok(op), _) => Ok(BoolExpr::CompareInt(a, op, b).into()),
                    (_, Ok(op)) => Ok(IntExpr::Binary(a, op, b).into()),
                    (Err(e1), Err(e2)) => Err(format!("{e1}, {e2}")),
                }
            }
            ExprPair::Real(a, b) => {
                let a = Box::new(a);
                let b = Box::new(b);
                match (op.try_into(), op.try_into()) {
                    (Ok(op), _) => Ok(BoolExpr::CompareReal(a, op, b).into()),
                    (_, Ok(op)) => Ok(RealExpr::Binary(a, op, b).into()),
                    (Err(e1), Err(e2)) => Err(format!("{e1}, {e2}")),
                }
            }
        }
    }
    fn coerce_equal(self: Expr, b: Expr) -> ExprPair {
        match self.get_type().max(b.get_type()) {
            Type::Boolean => ExprPair::Bool(self.cast_bool(), b.cast_bool()),
            Type::Integer => ExprPair::Int(self.cast_int(), b.cast_int()),
            Type::Real => ExprPair::Real(self.cast_real(), b.cast_real()),
        }
    }
    pub fn cast_bool(self) -> BoolExpr {
        match self {
            Expr::Bool(e) => e,
            Expr::Int(e) => BoolExpr::FromInt(Box::new(e)),
            Expr::Real(e) => BoolExpr::FromInt(Box::new(IntExpr::FromReal(Box::new(e)))),
        }
    }
    pub fn cast_int(self) -> IntExpr {
        match self {
            Expr::Bool(e) => IntExpr::FromBool(Box::new(e)),
            Expr::Int(e) => e,
            Expr::Real(e) => IntExpr::FromReal(Box::new(e)),
        }
    }
    pub fn cast_real(self) -> RealExpr {
        match self {
            Expr::Bool(e) => RealExpr::FromInt(Box::new(IntExpr::FromBool(Box::new(e)))),
            Expr::Int(e) => RealExpr::FromInt(Box::new(e)),
            Expr::Real(e) => e,
        }
    }
    #[must_use]
    pub fn cast_type(self, new_type: Type) -> Self {
        match new_type {
            Type::Boolean => self.cast_bool().into(),
            Type::Integer => self.cast_int().into(),
            Type::Real => self.cast_real().into(),
        }
    }
    pub const fn get_type(&self) -> Type {
        match self {
            Expr::Bool(_) => Type::Boolean,
            Expr::Int(_) => Type::Integer,
            Expr::Real(_) => Type::Real,
        }
    }
}
#[derive(Debug)]
pub enum CmpOpcode {
    Equal,
    NotEqual,
    GreaterThan,
    GreaterEqual,
    LessThan,
    LessThanEqual,
}
impl CmpOpcode {
    const fn as_str(&self) -> &'static str {
        use CmpOpcode::*;
        match self {
            Equal => "==",
            NotEqual => "!=",
            GreaterThan => ">",
            GreaterEqual => ">=",
            LessThan => "<",
            LessThanEqual => "<=",
        }
    }
}
impl TryFrom<BinaryOpcode> for CmpOpcode {
    type Error = String;
    fn try_from(value: BinaryOpcode) -> Result<Self, Self::Error> {
        use CmpOpcode::*;
        match value {
            BinaryOpcode::Equal => Ok(Equal),
            BinaryOpcode::NotEqual => Ok(NotEqual),
            BinaryOpcode::GreaterThan => Ok(GreaterThan),
            BinaryOpcode::GreaterEqual => Ok(GreaterEqual),
            BinaryOpcode::LessThan => Ok(LessThan),
            BinaryOpcode::LessThanEqual => Ok(LessThanEqual),
            _ => Err(format!("{value:?} is not a comparasion operation")),
        }
    }
}
#[derive(Debug)]
pub enum BoolOpcode {
    And,
    Or,
    Xor,
}
impl TryFrom<BinaryOpcode> for BoolOpcode {
    type Error = String;
    fn try_from(value: BinaryOpcode) -> Result<Self, Self::Error> {
        use BoolOpcode::*;
        match value {
            BinaryOpcode::And => Ok(And),
            BinaryOpcode::Or => Ok(Or),
            BinaryOpcode::Xor => Ok(Xor),
            _ => Err(format!("{value:?} is not a binary operation")),
        }
    }
}
#[derive(Debug)]
pub enum BoolExpr {
    CompareInt(Box<IntExpr>, CmpOpcode, Box<IntExpr>),
    CompareReal(Box<RealExpr>, CmpOpcode, Box<RealExpr>),
    Binary(Box<BoolExpr>, BoolOpcode, Box<BoolExpr>),
    Mem(Box<IntExpr>),
    FromInt(Box<IntExpr>),
    Variable(VariableReference),
    Value(bool),
}
impl BoolExpr {
    fn evaluate(&self, state: &State) -> bool {
        match self {
            BoolExpr::CompareInt(a, op, b) => {
                let a = a.evaluate(state);
                let b = b.evaluate(state);
                use CmpOpcode::*;
                match op {
                    Equal => a == b,
                    NotEqual => a != b,
                    GreaterThan => a > b,
                    GreaterEqual => a >= b,
                    LessThan => a < b,
                    LessThanEqual => a <= b,
                }
            }
            BoolExpr::CompareReal(a, op, b) => {
                let a = a.evaluate(state);
                let b = b.evaluate(state);
                use CmpOpcode::*;
                match op {
                    Equal => a == b,
                    NotEqual => a != b,
                    GreaterThan => a > b,
                    GreaterEqual => a >= b,
                    LessThan => a < b,
                    LessThanEqual => a <= b,
                }
            }
            BoolExpr::Binary(a, op, b) => {
                let a = a.evaluate(state);
                let b = b.evaluate(state);
                use BoolOpcode::*;
                match op {
                    And => a && b,
                    Or => a || b,
                    Xor => a != b,
                }
            }
            BoolExpr::Mem(e) => state.read_memory_bool(e.evaluate(state)),
            BoolExpr::FromInt(i) => i.evaluate(state) != 0,
            BoolExpr::Variable(v) => state.get_bool_variable_value(*v),
            BoolExpr::Value(b) => *b,
        }
    }
    fn compile(&self, state: &State) -> String {
        use BoolExpr::*;
        use BoolOpcode::*;
        match self {
            CompareInt(a, op, b) => format!(
                "({}) {} ({})",
                a.compile(state),
                op.as_str(),
                b.compile(state)
            ),
            CompareReal(a, op, b) => format!(
                "({}) {} ({})",
                a.compile(state),
                op.as_str(),
                b.compile(state)
            ),
            Binary(a, op, b) => format!(
                "({}) {} ({})",
                a.compile(state),
                match op {
                    And => "&&",
                    Or => "||",
                    Xor => "!=",
                },
                b.compile(state),
            ),
            Mem(e) => format!("boolean[{}]", e.compile(state)),
            FromInt(e) => format!("({}) != 0", e.compile(state)),
            Variable(v) => state.variable_table.get_name(*v),
            Value(v) => format!("{v:?}"),
        }
    }
}
#[derive(Debug)]
pub enum IntOpcode {
    Mul,
    Div,
    Mod,
    Add,
    Sub,
    BitAnd,
    BitOr,
    BitXor,
    BitShiftRight,
    BitShiftLeft,
}
impl IntOpcode {
    const fn as_str(&self) -> &'static str {
        use IntOpcode::*;
        match self {
            Mul => "*",
            Div => "/",
            Mod => "%",
            Add => "+",
            Sub => "-",
            BitAnd => "&",
            BitOr => "|",
            BitXor => "^",
            BitShiftRight => ">>",
            BitShiftLeft => "<<",
        }
    }
}
impl TryFrom<BinaryOpcode> for IntOpcode {
    type Error = String;
    fn try_from(value: BinaryOpcode) -> Result<Self, Self::Error> {
        use IntOpcode::*;
        match value {
            BinaryOpcode::Mul => Ok(Mul),
            BinaryOpcode::Div => Ok(Div),
            BinaryOpcode::Mod => Ok(Mod),
            BinaryOpcode::Add => Ok(Add),
            BinaryOpcode::Sub => Ok(Sub),
            BinaryOpcode::BitAnd => Ok(BitAnd),
            BinaryOpcode::BitOr => Ok(BitOr),
            BinaryOpcode::BitXor => Ok(BitXor),
            BinaryOpcode::BitShiftRight => Ok(BitShiftRight),
            BinaryOpcode::BitShiftLeft => Ok(BitShiftLeft),
            _ => Err(format!("{value:?} is not an integer operation")),
        }
    }
}
#[derive(Debug)]
pub enum IntExpr {
    Binary(Box<IntExpr>, IntOpcode, Box<IntExpr>),
    Mem(Box<IntExpr>),
    FromBool(Box<BoolExpr>),
    FromReal(Box<RealExpr>),
    Variable(VariableReference),
    Value(i32),
}
impl IntExpr {
    fn evaluate(&self, state: &State) -> i32 {
        use IntExpr::*;
        match self {
            Binary(a, op, b) => {
                let a = a.evaluate(state);
                let b = b.evaluate(state);
                use IntOpcode::*;
                match op {
                    Mul => a * b,
                    Div => a / b,
                    Mod => a % b,
                    Add => a + b,
                    Sub => a - b,
                    BitAnd => a & b,
                    BitOr => a | b,
                    BitXor => a ^ b,
                    BitShiftRight => a >> b,
                    BitShiftLeft => a << b,
                }
            }
            Mem(e) => state.read_memory_int(e.evaluate(state)),
            FromBool(b) => i32::from(b.evaluate(state)),
            FromReal(r) => r.evaluate(state) as i32,
            Variable(v) => state.get_int_variable_value(*v),
            Value(i) => *i,
        }
    }
    fn compile(&self, state: &State) -> String {
        use IntExpr::*;
        match self {
            Binary(a, op, b) => {
                let a = a.compile(state);
                let b = b.compile(state);
                format!("({}) {} ({})", a, op.as_str(), b)
            }
            Mem(e) => format!("integer[{}]", e.compile(state)),
            FromBool(e) => format!("({}) as i32", e.compile(state)),
            FromReal(e) => format!("({}) as i32", e.compile(state)),
            Variable(v) => state.variable_table.get_name(*v),
            Value(v) => format!("{v:?}"),
        }
    }
}
#[derive(Debug)]
pub enum RealOpcode {
    Mul,
    Div,
    Mod,
    Add,
    Sub,
}
impl RealOpcode {
    const fn as_str(&self) -> &'static str {
        use RealOpcode::*;
        match self {
            Mul => "*",
            Div => "/",
            Mod => "%",
            Add => "+",
            Sub => "-",
        }
    }
}
impl TryFrom<BinaryOpcode> for RealOpcode {
    type Error = String;

    fn try_from(value: BinaryOpcode) -> Result<Self, Self::Error> {
        use RealOpcode::*;
        match value {
            BinaryOpcode::Mul => Ok(Mul),
            BinaryOpcode::Div => Ok(Div),
            BinaryOpcode::Mod => Ok(Mod),
            BinaryOpcode::Add => Ok(Add),
            BinaryOpcode::Sub => Ok(Sub),
            _ => Err(format!("{value:?} is not a real operation")),
        }
    }
}
#[derive(Debug)]
pub enum RealExpr {
    Binary(Box<RealExpr>, RealOpcode, Box<RealExpr>),
    Mem(Box<IntExpr>),
    FromInt(Box<IntExpr>),
    Variable(VariableReference),
    Value(f64),
}
impl RealExpr {
    fn evaluate(&self, state: &State) -> f64 {
        use RealExpr::*;
        match self {
            Binary(a, op, b) => {
                let a = a.evaluate(state);
                let b = b.evaluate(state);
                use RealOpcode::*;
                match op {
                    Mul => a * b,
                    Div => a / b,
                    Mod => a % b,
                    Add => a + b,
                    Sub => a - b,
                }
            }
            Mem(e) => state.read_memory_real(e.evaluate(state)),
            FromInt(i) => i.evaluate(state) as f64,
            Variable(v) => state.get_real_variable_value(*v),
            Value(r) => *r,
        }
    }

    fn compile(&self, state: &State) -> String {
        use RealExpr::*;
        match self {
            Binary(a, op, b) => {
                let a = a.compile(state);
                let b = b.compile(state);
                format!("({}) {} ({})", a, op.as_str(), b)
            }
            Mem(e) => format!("real[{}]", e.compile(state)),
            FromInt(e) => format!("({}) as f64", e.compile(state)),
            Variable(v) => state.variable_table.get_name(*v),
            Value(v) => format!("{v:?}"),
        }
    }
}

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
impl Default for State {
    fn default() -> Self {
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
}
impl State {
    pub(crate) fn push_variable(
        &mut self,
        name: String,
        variable_type: Type,
    ) -> Option<VariableReference> {
        self.variable_table.push(name, variable_type)
    }
    pub(crate) fn get_variable(&self, name: &String) -> Option<VariableReference> {
        self.variable_table.get(name)
    }
    fn get_variable_type(&self, v: VariableReference) -> Type {
        self.variable_table.get_type(v)
    }
    fn set_bool_variable_value(&mut self, v: VariableReference, val: bool) {
        debug_assert_eq!(v.1, Type::Bool);
        self.variable_table.set_bool(v.0, val);
    }
    fn set_int_variable_value(&mut self, v: VariableReference, val: i32) {
        debug_assert_eq!(v.1, Type::Int);
        self.variable_table.set_int(v.0, val);
    }
    fn set_real_variable_value(&mut self, v: VariableReference, val: f64) {
        debug_assert_eq!(v.1, Type::Real);
        self.variable_table.set_real(v.0, val);
    }
    fn get_int_variable_value(&self, v: VariableReference) -> i32 {
        debug_assert_eq!(v.1, Type::Int);
        self.variable_table.get_int(v.0)
    }
    fn get_bool_variable_value(&self, v: VariableReference) -> bool {
        debug_assert_eq!(v.1, Type::Bool);
        self.variable_table.get_bool(v.0)
    }
    fn get_real_variable_value(&self, v: VariableReference) -> f64 {
        debug_assert_eq!(v.1, Type::Real);
        self.variable_table.get_real(v.0)
    }
    fn read_memory_bool(&self, addr: i32) -> bool {
        self.mem_bool[addr as usize]
    }
    fn read_memory_int(&self, addr: i32) -> i32 {
        self.mem_int[addr as usize]
    }
    fn read_memory_real(&self, addr: i32) -> f64 {
        self.mem_real[addr as usize]
    }
    fn write_memory_bool(&mut self, addr: i32, data: bool) {
        self.mem_bool[addr as usize] = data;
    }
    fn write_memory_int(&mut self, addr: i32, data: i32) {
        self.mem_int[addr as usize] = data;
    }
    fn write_memory_real(&mut self, addr: i32, data: f64) {
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
            use Type::*;
            let index;
            match variable_type {
                Bool => {
                    index = self.bools.len();
                    self.bools.push(Default::default());
                    &mut self.bools_names
                }
                Int => {
                    index = self.ints.len();
                    self.ints.push(Default::default());
                    &mut self.ints_names
                }
                Real => {
                    index = self.reals.len();
                    self.reals.push(Default::default());
                    &mut self.reals_names
                }
            }
            .push(name.clone());
            let id = VariableReference(index, variable_type);
            self.table.insert(name.clone(), id);
            self.variables
                .push((name, Value::default_from_type(variable_type)));
            Some(id)
        }
    }
    fn get(&self, name: &String) -> Option<VariableReference> {
        self.table.get(name).copied()
    }
    fn get_type(&self, v: VariableReference) -> Type {
        v.1
    }
    fn get_int(&self, v: usize) -> i32 {
        self.ints[v]
    }
    fn get_real(&self, v: usize) -> f64 {
        self.reals[v]
    }
    fn get_bool(&self, v: usize) -> bool {
        self.bools[v]
    }
    fn set_int(&mut self, v: usize, val: i32) {
        self.ints[v] = val;
    }
    fn set_real(&mut self, v: usize, val: f64) {
        self.reals[v] = val;
    }
    fn set_bool(&mut self, v: usize, val: bool) {
        self.bools[v] = val;
    }
    fn get_name(&self, v: VariableReference) -> String {
        use Type::*;
        match v.1 {
            Bool => self.bools_names[v.0].clone(),
            Int => self.ints_names[v.0].clone(),
            Real => self.reals_names[v.0].clone(),
        }
    }
}
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct VariableReference(usize, Type);

#[derive(Debug)]
pub enum BoolAssignable {
    Mem(IntExpr),
    Variable(VariableReference),
}
#[derive(Debug)]
pub enum IntAssignable {
    Mem(IntExpr),
    Variable(VariableReference),
}
#[derive(Debug)]
pub enum RealAssignable {
    Mem(IntExpr),
    Variable(VariableReference),
}

#[derive(Debug, From)]
pub enum Assignable {
    Bool(BoolAssignable),
    Int(IntAssignable),
    Real(RealAssignable),
}
impl Assignable {
    pub(crate) fn new_variable(v: VariableReference, state: &State) -> Self {
        use Type::*;
        match state.get_variable_type(v) {
            Bool => BoolAssignable::Variable(v).into(),
            Int => IntAssignable::Variable(v).into(),
            Real => RealAssignable::Variable(v).into(),
        }
    }
    pub(crate) fn new_memory(t: Type, e: Expr) -> Self {
        use Type::*;
        match t {
            Bool => BoolAssignable::Mem(e.cast_int()).into(),
            Int => IntAssignable::Mem(e.cast_int()).into(),
            Real => RealAssignable::Mem(e.cast_int()).into(),
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
    pub(crate) fn new_assign(a: Assignable, e: Expr) -> Result<Self, String> {
        match (a, e) {
            (Assignable::Bool(a), Expr::Bool(e)) => Ok(Stat::AssignBool(a, e)),
            (Assignable::Int(a), Expr::Int(e)) => Ok(Stat::AssignInt(a, e)),
            (Assignable::Real(a), Expr::Real(e)) => Ok(Stat::AssignReal(a, e)),
            (a, e) => Err(format!("Invalid assign {a:?} with {e:?}")),
        }
    }
}

#[derive(Debug, Default)]
pub struct StatList(pub Vec<Stat>);
impl StatList {
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
    Bool,
    Int,
    Real,
}
impl Type {
    const fn as_pascal_string(self) -> &'static str {
        use Type::*;
        match self {
            Bool => "boolean",
            Int => "integer",
            Real => "real",
        }
    }
    const fn as_rust_str(self) -> &'static str {
        use Type::*;
        match self {
            Bool => "bool",
            Int => "i32",
            Real => "f64",
        }
    }
}
#[derive(Debug, Copy, Clone, From, Display)]
pub enum Value {
    Int(i32),
    Real(f64),
    Bool(bool),
}
impl Value {
    const fn default_from_type(t: Type) -> Value {
        use Type::*;
        match t {
            Bool => Value::Bool(false),
            Int => Value::Int(0),
            Real => Value::Real(0.0),
        }
    }
    const fn get_type(self) -> Type {
        use Value::*;
        match self {
            Int(_) => Type::Int,
            Real(_) => Type::Real,
            Bool(_) => Type::Bool,
        }
    }
    fn compile(self) -> String {
        use Value::*;
        match self {
            Int(v) => format!("{v}"),
            Real(v) => format!("{v}"),
            Bool(v) => format!("{v}"),
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
impl Expr {
    fn evaluate(&self, state: &State) -> Value {
        use Expr::*;
        match self {
            Bool(e) => e.evaluate(state).into(),
            Int(e) => e.evaluate(state).into(),
            Real(e) => e.evaluate(state).into(),
        }
    }
    fn compile(&self, state: &State) -> String {
        use Expr::*;
        match self {
            Bool(e) => e.compile(state),
            Int(e) => e.compile(state),
            Real(e) => e.compile(state),
        }
    }
    pub(crate) fn new_variable(v: VariableReference, state: &State) -> Expr {
        use Type::*;
        match state.get_variable_type(v) {
            Bool => BoolExpr::Variable(v).into(),
            Int => IntExpr::Variable(v).into(),
            Real => RealExpr::Variable(v).into(),
        }
    }
    pub(crate) fn new_mem(mem_type: Type, e: Expr) -> Expr {
        use Type::*;
        match mem_type {
            Bool => BoolExpr::Mem(Box::new(e.cast_int())).into(),
            Int => IntExpr::Mem(Box::new(e.cast_int())).into(),
            Real => RealExpr::Mem(Box::new(e.cast_int())).into(),
        }
    }
    pub(crate) fn new_value(v: Value) -> Expr {
        use Value::*;
        match v {
            Int(i) => IntExpr::Value(i).into(),
            Real(r) => RealExpr::Value(r).into(),
            Bool(b) => BoolExpr::Value(b).into(),
        }
    }
    pub(crate) fn new_binary(a: Expr, b: Expr, op: BinaryOpcode) -> Result<Expr, String> {
        use Type::*;
        match a.get_type().max(b.get_type()) {
            Bool => {
                let a = Box::new(a.cast_bool());
                let b = Box::new(b.cast_bool());
                op.try_into().map(|op| BoolExpr::Binary(a, op, b).into())
            }
            Int => {
                let a = Box::new(a.cast_int());
                let b = Box::new(b.cast_int());
                match (op.try_into(), op.try_into()) {
                    (Ok(op), _) => Ok(BoolExpr::CompareInt(a, op, b).into()),
                    (_, Ok(op)) => Ok(IntExpr::Binary(a, op, b).into()),
                    (Err(e1), Err(e2)) => Err(format!("{e1}, {e2}")),
                }
            }
            Real => {
                let a = Box::new(a.cast_real());
                let b = Box::new(b.cast_real());
                match (op.try_into(), op.try_into()) {
                    (Ok(op), _) => Ok(BoolExpr::CompareReal(a, op, b).into()),
                    (_, Ok(op)) => Ok(RealExpr::Binary(a, op, b).into()),
                    (Err(e1), Err(e2)) => Err(format!("{e1}, {e2}")),
                }
            }
        }
    }
    pub(crate) fn cast_bool(self) -> BoolExpr {
        use Expr::*;
        match self {
            Bool(e) => e,
            Int(e) => BoolExpr::FromInt(Box::new(e)),
            Real(e) => BoolExpr::FromInt(Box::new(IntExpr::FromReal(Box::new(e)))),
        }
    }
    fn cast_int(self) -> IntExpr {
        use Expr::*;
        match self {
            Bool(e) => IntExpr::FromBool(Box::new(e)),
            Int(e) => e,
            Real(e) => IntExpr::FromReal(Box::new(e)),
        }
    }
    fn cast_real(self) -> RealExpr {
        use Expr::*;
        match self {
            Bool(e) => RealExpr::FromInt(Box::new(IntExpr::FromBool(Box::new(e)))),
            Int(e) => RealExpr::FromInt(Box::new(e)),
            Real(e) => e,
        }
    }
    pub(crate) fn cast_type(self, new_type: Type) -> Self {
        use Type::*;
        match new_type {
            Bool => self.cast_bool().into(),
            Int => self.cast_int().into(),
            Real => self.cast_real().into(),
        }
    }
    const fn get_type(&self) -> Type {
        use Expr::*;
        match self {
            Bool(_) => Type::Bool,
            Int(_) => Type::Int,
            Real(_) => Type::Real,
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
        use BoolExpr::*;
        match self {
            CompareInt(a, op, b) => {
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
            CompareReal(a, op, b) => {
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
            Binary(a, op, b) => {
                let a = a.evaluate(state);
                let b = b.evaluate(state);
                use BoolOpcode::*;
                match op {
                    And => a && b,
                    Or => a || b,
                    Xor => a != b,
                }
            }
            Mem(e) => state.read_memory_bool(e.evaluate(state)),
            FromInt(i) => i.evaluate(state) != 0,
            Variable(v) => state.get_bool_variable_value(*v),
            Value(b) => *b,
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
            Mem(e) => format!("{}[{}]", Type::Bool.as_pascal_string(), e.compile(state)),
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
            Mem(e) => format!("{}[{}]", Type::Int.as_pascal_string(), e.compile(state)),
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
            Mem(e) => format!("{}[{}]", Type::Real.as_pascal_string(), e.compile(state)),
            FromInt(e) => format!("({}) as f64", e.compile(state)),
            Variable(v) => state.variable_table.get_name(*v),
            Value(v) => format!("{v:?}"),
        }
    }
}

mod interpret {
    use super::*;
    impl Prog {
        pub fn execute(&self, state: &mut State) {
            println!("Running program {}", self.name);
            self.stat_list.execute(state);
        }
    }
    impl BoolAssignable {
        pub(crate) fn assign(&self, state: &mut State, val: bool) {
            use BoolAssignable::*;
            match self {
                Mem(e) => state.write_memory_bool(e.evaluate(state), val),
                Variable(v) => state.set_bool_variable_value(*v, val),
            }
        }
    }
    impl IntAssignable {
        pub(crate) fn assign(&self, state: &mut State, val: i32) {
            use IntAssignable::*;
            match self {
                Mem(e) => state.write_memory_int(e.evaluate(state), val),
                Variable(v) => state.set_int_variable_value(*v, val),
            }
        }
    }
    impl RealAssignable {
        pub(crate) fn assign(&self, state: &mut State, val: f64) {
            use RealAssignable::*;
            match self {
                Mem(e) => state.write_memory_real(e.evaluate(state), val),
                Variable(v) => state.set_real_variable_value(*v, val),
            }
        }
    }
    impl Stat {
        pub(crate) fn execute(&self, state: &mut State) {
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
    }
}

mod compile {
    use super::*;
    impl Prog {
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
    impl VariableTable {
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
    impl VariableReference {
        pub(crate) fn compile(self, state: &State) -> String {
            state.variable_table.variables[self.0].0.clone()
        }
    }
    impl BoolAssignable {
        pub(crate) fn compile(&self, state: &State) -> String {
            use BoolAssignable::*;
            match self {
                Mem(e) => format!("{}[{}]", Type::Bool.as_pascal_string(), e.compile(state)),
                Variable(v) => v.compile(state),
            }
        }
    }
    impl IntAssignable {
        pub(crate) fn compile(&self, state: &State) -> String {
            use IntAssignable::*;
            match self {
                Mem(e) => format!("{}[{}]", Type::Int.as_pascal_string(), e.compile(state)),
                Variable(v) => v.compile(state),
            }
        }
    }
    impl RealAssignable {
        pub(crate) fn compile(&self, state: &State) -> String {
            use RealAssignable::*;
            match self {
                Mem(e) => format!("{}[{}]", Type::Real.as_pascal_string(), e.compile(state)),
                Variable(v) => v.compile(state),
            }
        }
    }
    impl Stat {
        pub(crate) fn compile(&self, state: &State) -> String {
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
}

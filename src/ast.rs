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
// review later:
#![allow(clippy::print_stdout)]
#![allow(clippy::indexing_slicing)]

use core::fmt::Write as _;
use std::collections::HashMap;

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
}
impl VariableTable {
    fn new() -> Self {
        Self {
            table: HashMap::default(),
            variables: Vec::default(),
        }
    }
    fn push(&mut self, name: String, variable_type: Type) -> Option<VariableReference> {
        if self.table.get(&name).is_some() {
            None
        } else {
            let id = VariableReference(self.variables.len());
            self.table.insert(name.clone(), id);
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
pub struct VariableReference(usize);
impl VariableReference {
    pub fn get_type(self, state: &State) -> Type {
        state.get_variable_type(self)
    }
    fn compile(self, state: &State) -> String {
        state.variable_table.variables[self.0].0.clone()
    }
}

#[derive(Debug)]
pub enum Operand {
    Value(Value),
    Variable(VariableReference),
    UnaryOp(UnaryOpcode, Box<Expr>),
}
impl Operand {
    pub fn get_type(&self, state: &State) -> Type {
        use Operand::*;
        match self {
            Value(v) => v.get_type(),
            Variable(v) => v.get_type(state),
            UnaryOp(opcode, _) => opcode.get_type(),
        }
    }
    fn evaluate(&self, state: &State) -> Value {
        use Operand::*;
        match self {
            Value(v) => *v,
            Variable(v) => state.get_variable_value(*v),
            UnaryOp(opcode, e) => opcode.evaluate(e.evaluate(state), state),
        }
    }
    fn compile(&self, state: &State) -> String {
        use Operand::*;
        match self {
            Value(v) => v.compile(),
            Variable(v) => v.compile(state),
            UnaryOp(op, e) => op.compile(e, state),
        }
    }
}

#[derive(Debug)]
pub enum Assignable {
    Variable(VariableReference),
    Memory(Type, Expr),
}
impl Assignable {
    pub fn get_type(&self, state: &State) -> Type {
        use Assignable::*;
        match self {
            Variable(v) => v.get_type(state),
            Memory(t, _) => *t,
        }
    }
    fn assign(&self, val: Value, state: &mut State) {
        use Assignable::*;
        match self {
            Variable(v) => state.set_variable_value(*v, val),
            Memory(t, e) => state.write_memory(*t, e.evaluate(state), val),
        }
    }
    fn compile(&self, state: &State) -> String {
        use Assignable::*;
        match self {
            Variable(v) => v.compile(state),
            Memory(t, e) => {
                if UNCHECKED_INDEXING {
                    format!(
                        "*unsafe {{ {}.get_unchecked(({}) as usize) }}",
                        t.as_pascal_string(),
                        e.compile(state)
                    )
                } else {
                    format!("{}[({}) as usize]", t.as_pascal_string(), e.compile(state))
                }
            }
        }
    }
}

#[derive(Debug)]
pub enum Stat {
    Assign(Assignable, Expr),
    If(Expr, StatList),
    While(Expr, StatList),
    Print(Expr),
    PrintLn(Expr),
}
impl Stat {
    fn execute(&self, state: &mut State) {
        use Stat::*;
        use Value::*;
        state.executed_statements += 1;
        match self {
            Assign(a, e) => a.assign(e.evaluate(state), state),
            If(e, s) => {
                if e.evaluate(state).coerce_bool() {
                    s.execute(state);
                }
            }
            While(e, s) => {
                while e.evaluate(state).coerce_bool() {
                    s.execute(state);
                }
            }
            Print(e) => match e.evaluate(state) {
                Integer(v) => print!("{v}"),
                Real(v) => print!("{v}"),
                Boolean(v) => print!("{v}"),
            },
            PrintLn(e) => match e.evaluate(state) {
                Integer(v) => println!("{v}"),
                Real(v) => println!("{v}"),
                Boolean(v) => println!("{v}"),
            },
        }
    }
    fn compile(&self, state: &State) -> String {
        use Stat::*;
        match self {
            Assign(a, e) => format!("{} = {};\n", a.compile(state), e.compile(state)),
            If(e, s) => format!("if {} {}", e.compile(state), s.compile(state)),
            While(e, s) => {
                format!("while {} {}", e.compile(state), s.compile(state))
            }
            Print(e) => format!(
                "write!(__internal_stdout_print_lock,\"{{}}\", {});\n",
                e.compile(state)
            ),
            PrintLn(e) => format!(
                "writeln!(__internal_stdout_print_lock,\"{{}}\", {});\n",
                e.compile(state)
            ),
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
        format!(
            "{}{}{}",
            "{\n",
            self.0
                .iter()
                .map(|statement| statement.compile(state))
                .fold(String::new(), |mut a, b| {
                    a.push_str(&b);
                    a
                }),
            "}\n"
        )
    }
}

#[derive(Debug, Copy, Clone, PartialOrd, Ord, Eq, PartialEq)]
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
#[derive(Debug, Copy, Clone)]
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
#[rustfmt::skip]
impl From<i32> for Value { fn from(value: i32) -> Self { Self::Integer(value) } }
#[rustfmt::skip]
impl From<f64> for Value { fn from(value: f64) -> Self { Self::Real(value) } }
#[rustfmt::skip]
impl From<bool> for Value { fn from(value: bool) -> Self { Self::Boolean(value) } }
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
    fn coerce_equal(self, other: Value) -> ValuePair {
        use Type::*;
        let typ = self.get_type().max(other.get_type());
        match typ {
            Boolean => ValuePair::Boolean(self.coerce_bool(), other.coerce_bool()),
            Integer => ValuePair::Integer(self.coerce_integer(), other.coerce_integer()),
            Real => ValuePair::Real(self.coerce_real(), other.coerce_real()),
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
pub enum UnaryOpcode {
    CastBool,
    CastReal,
    CastInt,
    MemBool,
    MemReal,
    MemInt,
}
impl UnaryOpcode {
    const fn get_type(self) -> Type {
        use Type::*;
        use UnaryOpcode::*;
        match self {
            CastBool | MemBool => Boolean,
            CastReal | MemReal => Real,
            CastInt | MemInt => Integer,
        }
    }
    fn evaluate(self, a: Value, state: &State) -> Value {
        use UnaryOpcode::*;
        match self {
            CastBool | CastReal | CastInt => a.coerce_into(self.get_type()),
            MemInt | MemReal | MemBool => state.read_memory(self.get_type(), a),
        }
    }
    fn compile(self, a: &Expr, state: &State) -> String {
        use UnaryOpcode::*;
        match self {
            CastBool | CastReal | CastInt => a.compile_cast_type(self.get_type(), state),
            MemBool | MemReal | MemInt => format!(
                "{}[({}) as usize]",
                self.get_type().as_pascal_string(),
                a.compile_cast_integer(state)
            ),
        }
    }
}
#[derive(Debug, Copy, Clone)]
pub enum Opcode {
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
impl Opcode {
    fn evaluate(self, a: Value, b: Value) -> Value {
        macro_rules! op_equal_type {( $p:tt ) => {
            match a.coerce_equal(b) {
                ValuePair::Integer(a, b) => Integer(a $p b),
                ValuePair::Real(a, b) => Real(a $p b),
                ValuePair::Boolean(a, b) => Boolean(((a as u8) $p (b as u8)) != 0),
            }};
        }
        macro_rules! op_cmp {( $p:tt ) => {
            match a.coerce_equal(b) {
                ValuePair::Integer(a, b) => Boolean(a $p b),
                ValuePair::Real(a, b) => Boolean(a $p b),
                ValuePair::Boolean(a, b) => Boolean(a $p b),
            }};
        }
        use Opcode::*;
        use Value::*;
        match self {
            Mul => op_equal_type!(*),
            Div => op_equal_type!(/),
            Mod => op_equal_type!(%),
            Add => op_equal_type!(+),
            Sub => op_equal_type!(-),
            Equal => op_cmp!(==),
            NotEqual => op_cmp!(!=),
            GreaterThan => op_cmp!(>),
            GreaterEqual => op_cmp!(>=),
            LessThan => op_cmp!(<),
            LessThanEqual => op_cmp!(<=),
            BitAnd => Integer(a.coerce_integer() & b.coerce_integer()),
            BitOr => Integer(a.coerce_integer() | b.coerce_integer()),
            BitXor => Integer(a.coerce_integer() ^ b.coerce_integer()),
            BitShiftRight => Integer(a.coerce_integer() >> b.coerce_integer()),
            BitShiftLeft => Integer(a.coerce_integer() << b.coerce_integer()),
            And => Boolean(a.coerce_bool() && b.coerce_bool()),
            Or => Boolean(a.coerce_bool() || b.coerce_bool()),
            Xor => Boolean(a.coerce_bool() != b.coerce_bool()),
        }
    }
    fn get_type(self, a: Type, b: Type) -> Type {
        use Opcode::*;
        use Type::*;
        match self {
            Mul | Div | Mod | Add | Sub => a.max(b),
            Equal | NotEqual | GreaterThan | GreaterEqual | LessThan | LessThanEqual | And | Or
            | Xor => Boolean,
            BitAnd | BitOr | BitXor | BitShiftRight | BitShiftLeft => Integer,
        }
    }
    fn compile(self, state: &State, a: &Expr, b: &Expr) -> String {
        macro_rules! op_equal_type {
            ( $p:tt ) => {{
                let (typ, (ca, cb)) = a.compile_coerce_equal(b, state);
                match typ {
                    Boolean => format!("({}) as u8 {} ({}) as u8", ca, $p, cb),
                    Integer => format!("({}) {} ({})", ca, $p, cb),
                    Real => format!("({}) {} ({})", ca, $p, cb),
                }
            }};
        }
        macro_rules! op_cmp {
            ( $p:tt ) => {{
                let (typ, (ca, cb)) = a.compile_coerce_equal(b, state);
                match typ {
                    Boolean => format!("({}) {} ({})", ca, $p, cb),
                    Integer => format!("({}) {} ({})", ca, $p, cb),
                    Real => format!("({}) {} ({})", ca, $p, cb),
                }
            }};
        }
        use Opcode::*;
        use Type::*;
        match self {
            Mul => op_equal_type!("*"),
            Div => op_equal_type!("/"),
            Mod => op_equal_type!("%"),
            Add => op_equal_type!("+"),
            Sub => op_equal_type!("-"),
            Equal => op_cmp!("=="),
            NotEqual => op_cmp!("!="),
            GreaterThan => op_cmp!(">"),
            GreaterEqual => op_cmp!(">="),
            LessThan => op_cmp!("<"),
            LessThanEqual => op_cmp!("<="),
            BitAnd => format!(
                "({}) & ({})",
                a.compile_cast_integer(state),
                b.compile_cast_integer(state)
            ),
            BitOr => format!(
                "({}) | ({})",
                a.compile_cast_integer(state),
                b.compile_cast_integer(state)
            ),
            BitXor => format!(
                "({}) ^ ({})",
                a.compile_cast_integer(state),
                b.compile_cast_integer(state)
            ),
            BitShiftRight => format!(
                "({}) >> ({})",
                a.compile_cast_integer(state),
                b.compile_cast_integer(state)
            ),
            BitShiftLeft => format!(
                "({}) << ({})",
                a.compile_cast_integer(state),
                b.compile_cast_integer(state)
            ),
            And => format!(
                "({}) && ({})",
                a.compile_cast_bool(state),
                b.compile_cast_bool(state)
            ),
            Or => format!(
                "({}) || ({})",
                a.compile_cast_bool(state),
                b.compile_cast_bool(state)
            ),
            Xor => format!(
                "({}) != ({})",
                a.compile_cast_bool(state),
                b.compile_cast_bool(state)
            ),
        }
    }
}

mod prototype {
    use super::State;
    use super::Type;
    use super::Value;
    use super::VariableReference;
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

    pub enum Expr {
        Bool(BoolExpr),
        Int(IntExpr),
        Real(RealExpr),
    }
    impl From<BoolExpr> for Expr {
        fn from(value: BoolExpr) -> Self {
            Self::Bool(value)
        }
    }
    impl From<IntExpr> for Expr {
        fn from(value: IntExpr) -> Self {
            Self::Int(value)
        }
    }
    impl From<RealExpr> for Expr {
        fn from(value: RealExpr) -> Self {
            Self::Real(value)
        }
    }
    enum ExprPair {
        Bool(BoolExpr, BoolExpr),
        Int(IntExpr, IntExpr),
        Real(RealExpr, RealExpr),
    }
    impl Expr {
        pub fn from_variable(v: VariableReference, state: &State) -> Expr {
            match state.get_variable_type(v) {
                Type::Boolean => BoolExpr::Variable(v).into(),
                Type::Integer => IntExpr::Variable(v).into(),
                Type::Real => RealExpr::Variable(v).into(),
            }
        }
        pub fn from_value(v: Value) -> Expr {
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
        fn cast_bool(self) -> BoolExpr {
            match self {
                Expr::Bool(e) => e,
                Expr::Int(e) => BoolExpr::FromInt(Box::new(e)),
                Expr::Real(e) => BoolExpr::FromInt(Box::new(IntExpr::FromReal(Box::new(e)))),
            }
        }
        fn cast_int(self) -> IntExpr {
            match self {
                Expr::Bool(e) => IntExpr::FromBool(Box::new(e)),
                Expr::Int(e) => e,
                Expr::Real(e) => IntExpr::FromReal(Box::new(e)),
            }
        }
        fn cast_real(self) -> RealExpr {
            match self {
                Expr::Bool(e) => RealExpr::FromInt(Box::new(IntExpr::FromBool(Box::new(e)))),
                Expr::Int(e) => RealExpr::FromInt(Box::new(e)),
                Expr::Real(e) => e,
            }
        }
        fn get_type(&self) -> Type {
            match self {
                Expr::Bool(_) => Type::Boolean,
                Expr::Int(_) => Type::Integer,
                Expr::Real(_) => Type::Real,
            }
        }
    }
    pub enum CmpOpcode {
        Equal,
        NotEqual,
        GreaterThan,
        GreaterEqual,
        LessThan,
        LessThanEqual,
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
    }
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
            match self {
                IntExpr::Binary(a, op, b) => {
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
                IntExpr::Mem(e) => state.read_memory_int(e.evaluate(state)),
                IntExpr::FromBool(b) => b.evaluate(state) as i32,
                IntExpr::FromReal(r) => r.evaluate(state) as i32,
                IntExpr::Variable(v) => state.get_int_variable_value(*v), 
                IntExpr::Value(i) => *i,
            }
        }
    }
    pub enum RealOpcode {
        Mul,
        Div,
        Mod,
        Add,
        Sub,
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
    pub enum RealExpr {
        Binary(Box<RealExpr>, RealOpcode, Box<RealExpr>),
        Mem(Box<IntExpr>),
        FromInt(Box<IntExpr>),
        Variable(VariableReference),
        Value(f64),
    }
    impl RealExpr {
        fn evaluate(&self, state: &State) -> f64 {
            match self {
                RealExpr::Binary(a, op, b) => {
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
                RealExpr::Mem(e) => state.read_memory_real(e.evaluate(state)),
                RealExpr::FromInt(i) => i.evaluate(state) as f64,
                RealExpr::Variable(v) => state.get_real_variable_value(*v),
                RealExpr::Value(r) => *r,
            }
        }
    }
}

#[derive(Debug)]
pub enum Expr {
    Operand(Operand),
    Binary(Box<Expr>, Opcode, Box<Expr>),
}
impl Expr {
    pub fn get_type(&self, state: &State) -> Type {
        use Expr::*;
        match self {
            Operand(op) => op.get_type(state),
            Binary(e, op, operand) => op.get_type(e.get_type(state), operand.get_type(state)),
        }
    }
    fn evaluate(&self, state: &State) -> Value {
        use Expr::*;
        match self {
            Operand(operand) => operand.evaluate(state),
            Binary(e, opcode, operand) => {
                opcode.evaluate(e.evaluate(state), operand.evaluate(state))
            }
        }
    }
    fn compile(&self, state: &State) -> String {
        use Expr::*;
        match self {
            Operand(operand) => operand.compile(state),
            Binary(a, op, b) => op.compile(state, a, b),
        }
    }
    fn compile_cast_type(&self, typ: Type, state: &State) -> String {
        use Type::*;
        match typ {
            Boolean => self.compile_cast_bool(state),
            Integer => self.compile_cast_integer(state),
            Real => self.compile_cast_real(state),
        }
    }
    fn compile_cast_bool(&self, state: &State) -> String {
        use Type::*;
        match self.get_type(state) {
            Boolean => self.compile(state),
            Integer => format!("({}) != 0", self.compile(state)),
            Real => format!("({}) != 0", self.compile_cast_integer(state)),
        }
    }
    fn compile_cast_integer(&self, state: &State) -> String {
        use Type::*;
        match self.get_type(state) {
            Boolean | Real => format!("({}) as i32", self.compile(state)),
            Integer => self.compile(state),
        }
    }
    fn compile_cast_real(&self, state: &State) -> String {
        use Type::*;
        match self.get_type(state) {
            Boolean => format!("({}) as f64", self.compile_cast_integer(state)),
            Integer => format!("({}) as f64", self.compile(state)),
            Real => self.compile(state),
        }
    }
    fn compile_coerce_equal(&self, other: &Expr, state: &State) -> (Type, (String, String)) {
        let typ = self.get_type(state).max(other.get_type(state));
        (
            typ,
            (
                self.compile_cast_type(typ, state),
                other.compile_cast_type(typ, state),
            ),
        )
    }
}

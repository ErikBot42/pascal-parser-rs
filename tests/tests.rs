use pascal_parser_rust::ProgParser;
use pascal_parser_rust::State;

macro_rules! good_case {
    ($name:ident) => {
        #[test]
        fn $name() {
            let program = include_str!(concat!("test_files/", stringify!($name), ".pas"));
            let mut state = State::default();
            ProgParser::new().parse(&mut state, program).unwrap();
        }
    };
}
macro_rules! bad_case {
    ($name:ident) => {
        #[test]
        fn $name() {
            let program = include_str!(concat!("test_files/", stringify!($name), ".pas"));
            let mut state = State::default();
            ProgParser::new().parse(&mut state, program).unwrap_err();
        }
    };
}

// white box cases
bad_case!(fun1);
bad_case!(fun2);
bad_case!(fun3);
good_case!(fun4); // 2 vardec lists is valid
bad_case!(fun5);

// incorrect semantic cases
bad_case!(sem1);
bad_case!(sem2);
bad_case!(sem3);
bad_case!(sem4);
bad_case!(sem5);

// syntax tests
bad_case!(testa);
bad_case!(testb);
bad_case!(testc);
bad_case!(testd);
bad_case!(teste);
bad_case!(testf);
bad_case!(testg);
bad_case!(testh);
bad_case!(testi);
bad_case!(testj);
bad_case!(testk);
bad_case!(testl);
bad_case!(testm);
bad_case!(testn);
bad_case!(testo);
bad_case!(testp);
bad_case!(testq);
bad_case!(testr);
bad_case!(tests);
bad_case!(testt);
bad_case!(testu);
bad_case!(testv);
good_case!(testw); // subtraction is implemented
bad_case!(testx);
bad_case!(testy);
bad_case!(testz);

// simple, valid tests
good_case!(testok1);
good_case!(testok2);
good_case!(testok3);
good_case!(testok4);
good_case!(testok5);
good_case!(testok6);
good_case!(testok7);


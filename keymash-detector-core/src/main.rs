use std::env;

use keymash_detector_core::*;

fn main() {
    let input = env::args().skip(1).next().unwrap();
    let input = preprocess(input.as_bytes());
    assert!(input.iter().all(u8::is_ascii));
    println!("{}", str::from_utf8(&input).unwrap());
    println!("{}", fit_keymash_model(&input) - eval_english_model(&input));
}

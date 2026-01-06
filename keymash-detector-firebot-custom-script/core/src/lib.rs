use keymash_detector_core::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn test_fragment(input: String) -> f64 {
    let mut ret = f64::NEG_INFINITY;

    let mut txt = input.into_bytes();
    txt.retain(|&c| c == b' ' || c.is_ascii_graphic());

    let words: Vec<_> = txt.split(|&c| c == b' ').collect();
    for i in 0..words.len() {
        let mut word = words[i].to_owned();
        if word.len() < 6 {
            // try joining with next word
            word.push(b' ');
            word.extend_from_slice(words.get(i + 1).cloned().unwrap_or(&[]));
            if word.len() < 7 {
                continue;
            }
        }

        let txt = preprocess(&word);
        let llr = fit_keymash_model(&txt) - eval_english_model(&txt);
        ret = ret.max(llr);
    }
    ret
}

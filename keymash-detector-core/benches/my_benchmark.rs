use criterion::{criterion_group, criterion_main, Criterion, BatchSize};
use std::hint::black_box;
use keymash_detector_core::*;

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

const TEXT_1: &str = "The atmosphere of Jupiter lacks a clear lower boundary, and gradually transitions into the liquid interior of the planet. From lowest to highest, the atmospheric layers are the troposphere, stratosphere, thermosphere and exosphere. Each layer has characteristic temperature gradients. The lowest layer, the troposphere, has a complicated system of clouds and hazes composed of layers of ammonia, ammonium hydrosulfide, and water.";
const TEXT_2: &str = "Basalt is a dark colored extrusive volcanic rock. The weathering of iron-bearing minerals within it results in a rusty brown color when exposed to air and water, lending it a distinct reddish or purple–red hue. Basalt frequently breaks into octagonal and pentagonal columns, creating a unique \"postpile\" appearance. Extensive slopes made of fractured basalt talus are visible at the base of many of the cliffs along the Metacomet Ridge.";
const MSG_1: &str = "Francis Bacon had advocated the advancement of knowledge through observation and experiment";
const MASH_1: &str = "aksjdflaskdjflkasdjfasdf";

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("text 1", |b| b.iter_batched(||TEXT_1.to_string(), |v| test_fragment(black_box(v)), BatchSize::LargeInput));
    c.bench_function("text 2", |b| b.iter_batched(||TEXT_2.to_string(), |v| test_fragment(black_box(v)), BatchSize::LargeInput));
    c.bench_function("msg 1", |b| b.iter_batched(||MSG_1.to_string(), |v| test_fragment(black_box(v)), BatchSize::LargeInput));
    c.bench_function("mash 1", |b| b.iter_batched(||MASH_1.to_string(), |v| test_fragment(black_box(v)), BatchSize::LargeInput));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
